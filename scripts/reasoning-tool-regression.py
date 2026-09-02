#!/usr/bin/env python3
"""Capture Qwen3.8 thinking/tool-parser regressions through the OpenAI API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


TOOL = {
    "type": "function",
    "function": {
        "name": "shell_probe",
        "description": "Run a harmless diagnostic command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def get_json(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, payload: dict, timeout: int = 600):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"http_error": exc.code, "body": body}
    except Exception as exc:
        return {"client_error": repr(exc)}


def longest_equal_run(values) -> int:
    best = current = 0
    previous = object()
    for value in values or []:
        if value == previous:
            current += 1
        else:
            previous = value
            current = 1
        best = max(best, current)
    return best


def summarize(response: dict) -> dict:
    if "choices" not in response:
        return {"error": response}
    choice = response["choices"][0]
    message = choice.get("message") or {}
    token_ids = response.get("token_ids") or choice.get("token_ids") or []
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    return {
        "finish_reason": choice.get("finish_reason"),
        "reasoning_content": reasoning,
        "content": content,
        "tool_calls": message.get("tool_calls"),
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "longest_identical_token_run": longest_equal_run(token_ids),
        "bang_count": reasoning.count("!") + content.count("!"),
        "usage": response.get("usage"),
    }


def base_payload(model: str, messages: list[dict], *, thinking: bool, tools: bool,
                 preserve_thinking: bool, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "return_token_ids": True,
        "chat_template_kwargs": {
            "enable_thinking": thinking,
            "preserve_thinking": preserve_thinking,
        },
    }
    if thinking:
        payload["reasoning_effort"] = "low"
    if tools:
        payload["tools"] = [TOOL]
        payload["tool_choice"] = "auto"
    return payload


def run_case(base_url: str, model: str, name: str, messages: list[dict], **kwargs):
    payload = base_payload(model, messages, **kwargs)
    before = get_json(base_url + "/v1/loads")
    started = time.time()
    response = post_json(base_url + "/v1/chat/completions", payload)
    ended = time.time()
    after = get_json(base_url + "/v1/loads")
    return {
        "name": name,
        "started": started,
        "ended": ended,
        "elapsed_s": round(ended - started, 3),
        "payload": payload,
        "response": response,
        "summary": summarize(response),
        "loads_before": before,
        "loads_after": after,
    }


def synthetic_refeed_messages() -> list[dict]:
    return [
        {"role": "user", "content": "Use shell_probe to obtain the number four."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should call the supplied tool once.",
            "tool_calls": [
                {
                    "id": "call_previous",
                    "type": "function",
                    "function": {
                        "name": "shell_probe",
                        "arguments": {"command": "printf 4"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_previous", "content": "4"},
        {"role": "user", "content": "State the observed number in one sentence."},
    ]


def append_assistant_and_tool(messages: list[dict], response: dict, turn: int) -> None:
    if "choices" not in response:
        return
    message = response["choices"][0].get("message") or {}
    assistant = {
        "role": "assistant",
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
    }
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    messages.append(assistant)
    for call in tool_calls:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id", f"call_{turn}"),
                "content": f"probe-ok-{turn}",
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--agent-turns", type=int, default=8)
    args = parser.parse_args()

    cases = []
    simple = [{"role": "user", "content": "Call shell_probe with command echo parser-ok"}]
    for thinking in (True, False):
        for tools in (True, False):
            cases.append(
                run_case(
                    args.base_url,
                    args.model,
                    f"thinking_{thinking}_tools_{tools}",
                    simple,
                    thinking=thinking,
                    tools=tools,
                    preserve_thinking=True,
                    max_tokens=64,
                )
            )

    for preserve in (True, False):
        cases.append(
            run_case(
                args.base_url,
                args.model,
                f"refeed_preserve_{preserve}",
                synthetic_refeed_messages(),
                thinking=True,
                tools=True,
                preserve_thinking=preserve,
                max_tokens=128,
            )
        )

    messages: list[dict] = []
    agent_results = []
    for turn in range(1, args.agent_turns + 1):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Agent turn {turn}: call shell_probe with command "
                    f"echo agent-turn-{turn}. Do not call any other tool."
                ),
            }
        )
        result = run_case(
            args.base_url,
            args.model,
            f"agent_turn_{turn}",
            messages,
            thinking=True,
            tools=True,
            preserve_thinking=True,
            max_tokens=128,
        )
        agent_results.append(result)
        append_assistant_and_tool(messages, result["response"], turn)

    document = {
        "label": args.label,
        "started": min(case["started"] for case in cases),
        "ended": time.time(),
        "server_info": get_json(args.base_url + "/get_server_info"),
        "matrix": cases,
        "agent": agent_results,
        "agent_messages": messages,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.label}.json"
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "matrix": [case["summary"] for case in cases],
        "agent": [case["summary"] for case in agent_results],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


