#!/usr/bin/env python3
"""Two-lane long-context continuation/compaction soak for the Thor server."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


MODEL = "Qwen/Qwen3.6-27B-NVFP4"
BOOK_URL = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
SYSTEM = (
    "You are continuing a long literary draft for a server stability test. "
    "Preserve character continuity and prose style. Return prose only, without "
    "analysis, headings, or comments about the test."
)


class Recorder:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def write(self, event: str, **fields):
        row = {"ts": time.time(), "event": event, **fields}
        line = json.dumps(row, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        print(line, flush=True)


def get_json(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def post_json(url: str, payload: dict, timeout: int = 1800):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def read_meminfo():
    values = {}
    with open("/proc/meminfo", encoding="ascii") as f:
        for line in f:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    return values


def monitor(base_url: str, rec: Recorder, stop: threading.Event):
    while not stop.wait(2):
        try:
            mem = read_meminfo()
            loads = get_json(base_url + "/v1/loads", timeout=3)["loads"][0]
            rec.write(
                "sample",
                mem_available_kib=mem["MemAvailable"],
                swap_free_kib=mem["SwapFree"],
                running=loads["num_running_reqs"],
                waiting=loads["num_waiting_reqs"],
                used_tokens=loads["num_used_tokens"],
                total_tokens=loads["num_total_tokens"],
                active_tokens=loads["num_active_tokens"],
                token_usage=loads["token_usage"],
                gen_tps=loads["gen_throughput"],
            )
        except Exception as exc:
            rec.write("sample_error", error=repr(exc))


def chat_tokens(tokenizer, messages):
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "shape"):
        return int(encoded.shape[-1])
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return len(encoded)


def fit_source(tokenizer, token_pool, start, target_prompt_tokens, instruction):
    overhead_messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction},
    ]
    overhead = chat_tokens(tokenizer, overhead_messages)
    take = max(1, target_prompt_tokens - overhead - 32)
    n = len(token_pool)
    ids = [token_pool[(start + i) % n] for i in range(take)]
    source = tokenizer.decode(ids, skip_special_tokens=True)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": source + "\n\n" + instruction,
        },
    ]
    # Decode/encode boundaries can shift a few tokens. Trim until safely below target.
    while chat_tokens(tokenizer, messages) > target_prompt_tokens:
        ids = ids[:-256]
        source = tokenizer.decode(ids, skip_special_tokens=True)
        messages[1]["content"] = source + "\n\n" + instruction
    return source, messages, chat_tokens(tokenizer, messages)


def call_chat(base_url, lane, phase, cycle, messages, max_tokens, rec):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.time()
    rec.write(
        "request_start",
        lane=lane,
        phase=phase,
        cycle=cycle,
        max_tokens=max_tokens,
    )
    response = post_json(base_url + "/v1/chat/completions", payload)
    elapsed = time.time() - started
    choice = response["choices"][0]
    usage = response.get("usage", {})
    text = choice["message"].get("content") or ""
    rec.write(
        "request_done",
        lane=lane,
        phase=phase,
        cycle=cycle,
        elapsed_s=round(elapsed, 3),
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        output_chars=len(text),
    )
    return text, response


def lane_run(args, lane, tokenizer, book_ids, rec):
    cursor = (lane - 1) * (len(book_ids) // 2)
    summaries = []
    results = []
    for cycle in range(1, args.cycles + 1):
        continue_instruction = (
            "Continue this draft seamlessly from its final scene. Advance the plot "
            "with dialogue and concrete action."
        )
        source, continue_messages, estimated = fit_source(
            tokenizer, book_ids, cursor, args.target_prompt_tokens, continue_instruction
        )
        rec.write(
            "prompt_ready",
            lane=lane,
            phase="continue",
            cycle=cycle,
            estimated_prompt_tokens=estimated,
        )
        continuation, cont_response = call_chat(
            args.base_url,
            lane,
            "continue",
            cycle,
            continue_messages,
            args.continue_tokens,
            rec,
        )

        compact_instruction = (
            "Create a dense continuity record for the first half of the draft below. "
            "Preserve every plot dependency, named character, relationship, location, "
            "unresolved promise, and stylistic constraint needed to continue it. Do not "
            "reproduce long passages. Return only the continuity record.\n\nDRAFT:\n"
        )
        compact_messages = [
            {"role": "system", "content": "You compact long literary contexts faithfully."},
            {
                "role": "user",
                "content": compact_instruction + source + "\n\nLATEST CONTINUATION:\n" + continuation,
            },
        ]
        # The continuation can push the prompt over target; trim the tail of source only.
        while chat_tokens(tokenizer, compact_messages) > args.target_prompt_tokens:
            source_ids = tokenizer.encode(source, add_special_tokens=False)[:-256]
            source = tokenizer.decode(source_ids, skip_special_tokens=True)
            compact_messages[1]["content"] = (
                compact_instruction + source + "\n\nLATEST CONTINUATION:\n" + continuation
            )
        rec.write(
            "prompt_ready",
            lane=lane,
            phase="compact",
            cycle=cycle,
            estimated_prompt_tokens=chat_tokens(tokenizer, compact_messages),
        )
        summary, compact_response = call_chat(
            args.base_url,
            lane,
            "compact",
            cycle,
            compact_messages,
            args.compact_tokens,
            rec,
        )
        source_ids = tokenizer.encode(source, add_special_tokens=False)
        retained = tokenizer.decode(source_ids[len(source_ids) // 2 :], skip_special_tokens=True)
        compacted_messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": "PRIOR CONTINUITY RECORD:\n" + summary + "\n\nRECENT HALF:\n" + retained,
            },
            {"role": "assistant", "content": continuation},
            {"role": "user", "content": "Continue seamlessly."},
        ]
        compacted_tokens = chat_tokens(tokenizer, compacted_messages)
        rec.write(
            "context_compacted",
            lane=lane,
            cycle=cycle,
            before_tokens=compact_response.get("usage", {}).get("prompt_tokens"),
            after_tokens=compacted_tokens,
            ratio=round(
                compacted_tokens
                / max(1, compact_response.get("usage", {}).get("prompt_tokens", 1)),
                4,
            ),
        )
        summaries.append(summary)
        results.append(
            {
                "cycle": cycle,
                "continuation_usage": cont_response.get("usage"),
                "compaction_usage": compact_response.get("usage"),
                "compacted_tokens": compacted_tokens,
            }
        )
        cursor = (cursor + args.target_prompt_tokens) % len(book_ids)
    return {"lane": lane, "cycles": results, "summary_chars": sum(map(len, summaries))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--target-prompt-tokens", type=int, default=250000)
    parser.add_argument("--continue-tokens", type=int, default=1024)
    parser.add_argument("--compact-tokens", type=int, default=4096)
    parser.add_argument("--output-dir", default="soak-results")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.output_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=False)
    rec = Recorder(out_dir / "events.jsonl")
    rec.write("test_start", args=vars(args), pid=os.getpid())

    book_path = out_dir.parent / "pg1184.txt"
    if not book_path.exists():
        urllib.request.urlretrieve(BOOK_URL, book_path)
    book = book_path.read_text(encoding="utf-8-sig")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    book_ids = tokenizer.encode(book, add_special_tokens=False)
    rec.write("corpus_ready", chars=len(book), tokens=len(book_ids), source=BOOK_URL)

    stop = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor, args=(args.base_url, rec, stop), daemon=True
    )
    monitor_thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(lane_run, args, lane, tokenizer, book_ids, rec)
                for lane in (1, 2)
            ]
            results = [future.result() for future in futures]
        (out_dir / "summary.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rec.write("test_done", results=results)
    except Exception as exc:
        rec.write("test_failed", error=repr(exc))
        raise
    finally:
        stop.set()
        monitor_thread.join(timeout=5)


if __name__ == "__main__":
    main()
