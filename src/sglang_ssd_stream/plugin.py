"""SGLang plugin registration."""

from __future__ import annotations

import hashlib
import importlib.util
import platform
from pathlib import Path

from .config import configure_cli
from .graph import after_load_batch, around_execute

SUPPORTED_SGLANG = {
    "x86_64": (
        "3df8e1e7dbc5807696622afe2929b6c33c185ca3",
        {
            "sglang.srt.layers.attention.qsa.kernel": (
                "df4de87278688c65bd7d325c146b874180adbb469b2337ad3e7913d94f9ad189"
            ),
            "sglang.srt.layers.attention.qsa.sparse_attn": (
                "09f4df649d593021ff7d1b4cece8ada057434984e0e7d064a85f40f1121e9840"
            ),
            "sglang.srt.layers.attention.qwen_sparse_attn_backend": (
                "5be3cf21adf64b965d7b40b224faebdc04d15ab3f6593829425b665494847ae2"
            ),
            "sglang.srt.models.qwen4_exp": (
                "f406977eb2373937393241f453477867f7dc943bd4839216db8fe66fa9f921d8"
            ),
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner": (
                "3554b172d18be110e32b6140ac97154025962daceca454f417dde4e15862b74d"
            ),
            "sglang.srt.server_args": (
                "177600230a33a7badf94f49c3dff7d5aae3762f03a1152c8ea62302d188734d8"
            ),
        },
    ),
    "aarch64": (
        "0a79825b7baa3e2aafd54e89097a5aba83d00b4e",
        {
            "sglang.kernels.kda_kernels.qwen38_qsa_sm121.kernel": (
                "a419104f3a49e402c5b4d3297dd73d4784d3b41106cb989b201dcdace9487a53"
            ),
            "sglang.kernels.ops.attention": (
                "64b7dd1608acd9a6314f58bb10c29b22ecf4b368562edb83b1f198472fe6dec0"
            ),
            "sglang.srt.layers.attention.qsa.kernel": (
                "7e369f09293fb9b0872c21f0010247ec1e3a696b5ad4809f04d9a730b1031095"
            ),
            "sglang.srt.layers.attention.qsa.sparse_attn": (
                "f3801cc37453278e884873a821350def23c58453eb91c56f2c96d8f62a3709f5"
            ),
            "sglang.srt.layers.attention.qwen_sparse_attn_backend": (
                "7cb54a4440a3f6f9619227138398c173993a9f8deec8e6a6be50ce9067d50153"
            ),
            "sglang.srt.models.qwen4_exp": (
                "f406977eb2373937393241f453477867f7dc943bd4839216db8fe66fa9f921d8"
            ),
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner": (
                "3554b172d18be110e32b6140ac97154025962daceca454f417dde4e15862b74d"
            ),
            "sglang.srt.server_args": (
                "177600230a33a7badf94f49c3dff7d5aae3762f03a1152c8ea62302d188734d8"
            ),
        },
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

    architecture = platform.machine()
    if architecture == "arm64":
        architecture = "aarch64"
    try:
        commit, modules = SUPPORTED_SGLANG[architecture]
    except KeyError as exc:
        raise RuntimeError(
            f"sglang-ssd-stream does not support {platform.machine()}"
        ) from exc

    for module_name, expected_hash in modules.items():
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot locate required SGLang module {module_name}")
        actual_hash = _sha256(spec.origin)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"sglang-ssd-stream requires SGLang commit {commit}; "
                f"{module_name} has SHA-256 {actual_hash}, expected {expected_hash}"
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
