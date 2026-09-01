#!/usr/bin/env python3
"""Exercise SGLang auto-truncation and an output-limit finish near context_len."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


MODEL = "Qwen/Qwen3.6-27B-NVFP4"


def get_json(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, payload: dict, timeout: int = 1800):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def token_count(tokenizer, messages) -> int:
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


def write_event(path: Path, event: str, **fields):
    row = {"ts": time.time(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def monitor(base_url: str, events: Path, stop: threading.Event):
    while not stop.wait(2):
        try:
            loads = get_json(base_url + "/v1/loads", timeout=3)["loads"][0]
            meminfo = {}
            with open("/proc/meminfo", encoding="ascii") as source:
                for line in source:
                    key, value = line.split(":", 1)
                    meminfo[key] = int(value.strip().split()[0])
            write_event(
                events,
                "sample",
                running=loads["num_running_reqs"],
                waiting=loads["num_waiting_reqs"],
                used_tokens=loads["num_used_tokens"],
                active_tokens=loads["num_active_tokens"],
                mem_available_kib=meminfo["MemAvailable"],
                swap_free_kib=meminfo["SwapFree"],
            )
        except Exception as exc:
            write_event(events, "sample_error", error=repr(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=263000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--context-length", type=int, default=262144)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / time.strftime("%Y%m%dT%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    events = output_dir / "events.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    corpus = Path(args.corpus).read_text(encoding="utf-8-sig")
    corpus_ids = tokenizer.encode(corpus, add_special_tokens=False)

    system = (
        "This is a server boundary regression test. Continue the supplied literary "
        "text indefinitely with plain prose. Do not emit an end marker or conclude."
    )
    overhead = token_count(
        tokenizer,
        [{"role": "system", "content": system}, {"role": "user", "content": "x"}],
    )
    take = args.target_prompt_tokens - overhead + 32
    ids = [corpus_ids[index % len(corpus_ids)] for index in range(take)]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    while token_count(tokenizer, messages) > args.target_prompt_tokens:
        ids = ids[:-128]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        messages[1]["content"] = text
    estimated_prompt_tokens = token_count(tokenizer, messages)
    write_event(
        events,
        "test_start",
        estimated_prompt_tokens=estimated_prompt_tokens,
        requested_max_tokens=args.max_tokens,
        context_length=args.context_length,
    )

    stop = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor, args=(args.base_url, events, stop), daemon=True
    )
    monitor_thread.start()
    started = time.time()
    try:
        response = post_json(
            args.base_url + "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "ignore_eos": True,
                "temperature": 0.0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        choice = response["choices"][0]
        usage = response["usage"]
        result = {
            "elapsed_s": round(time.time() - started, 3),
            "finish_reason": choice.get("finish_reason"),
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "output_chars": len(choice["message"].get("content") or ""),
        }
        failures = []
        if result["finish_reason"] != "length":
            failures.append("finish_reason is not length")
        if result["completion_tokens"] != args.max_tokens:
            failures.append("completion token budget was not fully returned")
        if not result["prompt_tokens"] < estimated_prompt_tokens:
            failures.append("server usage does not prove input truncation")
        if result["total_tokens"] > args.context_length:
            failures.append("response usage exceeds context length")
        result["assertions"] = "passed" if not failures else failures
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_event(events, "request_done", **result)
        if failures:
            raise AssertionError("; ".join(failures))
    except Exception as exc:
        write_event(events, "test_failed", error=repr(exc))
        raise
    finally:
        stop.set()
        monitor_thread.join(timeout=5)

    deadline = time.time() + 30
    while time.time() < deadline:
        loads = get_json(args.base_url + "/v1/loads")["loads"][0]
        if loads["num_running_reqs"] == 0 and loads["num_used_tokens"] == 0:
            write_event(events, "kv_released", loads=loads)
            break
        time.sleep(1)
    else:
        raise RuntimeError("KV did not return to zero within 30 seconds")
    write_event(events, "test_done")


if __name__ == "__main__":
    main()
