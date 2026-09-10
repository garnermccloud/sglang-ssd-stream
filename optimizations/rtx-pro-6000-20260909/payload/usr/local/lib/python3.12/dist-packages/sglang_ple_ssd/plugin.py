"""SGLang plugin registration."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from .config import add_cli_args, configure_cli
from .graph import after_load_batch, around_execute

SUPPORTED_SGLANG_COMMIT = "d91c3682b0b429e4c70df63cd57f819588ce29b0"
SUPPORTED_MODULES = {
    "sglang.srt.models.qwen4_exp": (
        "f406977eb2373937393241f453477867f7dc943bd4839216db8fe66fa9f921d8"
    ),
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner": (
        "551493108ae0bb816cf64b7418e3e4b8d6458e8e8dd93263963eeef0f67e635d"
    ),
    "sglang.srt.server_args": (
        "177600230a33a7badf94f49c3dff7d5aae3762f03a1152c8ea62302d188734d8"
    ),
}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for module_name, expected_hash in SUPPORTED_MODULES.items():
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot locate required SGLang module {module_name}")
        actual_hash = _sha256(spec.origin)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"sglang-ple-ssd requires SGLang commit {SUPPORTED_SGLANG_COMMIT}; "
                f"{module_name} has SHA-256 {actual_hash}, expected {expected_hash}"
            )

    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.add_cli_args",
        add_cli_args,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.from_cli_args",
        configure_cli,
        HookType.BEFORE,
    )

    from .qwen4 import Qwen4PLELayer, around_load_weights

    HookRegistry.register(
        "sglang.srt.models.qwen4_exp.Qwen4ExpPLELayer",
        Qwen4PLELayer,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.models.qwen4_exp.Qwen4ExpForConditionalGeneration.load_weights",
        around_load_weights,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
        "DecodeCudaGraphRunner.execute",
        around_execute,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
        "DecodeCudaGraphRunner.load_batch",
        after_load_batch,
        HookType.AFTER,
    )
