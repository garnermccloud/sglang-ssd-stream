import sys
from types import SimpleNamespace

from sglang_ssd_stream import cli


def _prepare_serve(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    cuda_home = tmp_path / "cuda"
    cuda_home.mkdir()
    monkeypatch.setattr(cli, "_require_compiler", lambda: None)
    monkeypatch.setattr(cli, "_cuda_home", lambda: cuda_home)
    monkeypatch.setattr(cli, "_pinned_revision", lambda _data_dir: "a" * 40)
    monkeypatch.setattr(cli, "_download_model", lambda _data_dir, _revision: snapshot)
    monkeypatch.setattr(
        cli,
        "_detect_hardware",
        lambda: cli.Hardware("x86_64", "RTX PRO 6000 Blackwell", 97_887),
    )
    return snapshot, cuda_home


def test_model_revision_stays_pinned_across_software_runs(monkeypatch, tmp_path):
    resolved = []
    monkeypatch.setattr(
        cli,
        "_latest_model_revision",
        lambda data_dir: resolved.append(data_dir) or "a" * 40,
    )

    assert cli._pinned_revision(tmp_path) == "a" * 40
    assert cli._pinned_revision(tmp_path) == "a" * 40
    assert resolved == [tmp_path]


def test_rtx_profile_uses_stable_context_and_native_mtp(tmp_path):
    args = cli._profile_args(
        cli.Hardware("x86_64", "NVIDIA RTX PRO 6000 Blackwell", 97_887),
        tmp_path,
        None,
    )

    assert args[args.index("--context-length") + 1] == "131072"
    assert args[args.index("--max-total-tokens") + 1] == "131072"
    assert args[args.index("--mem-fraction-static") + 1] == "0.985"
    assert args[args.index("--speculative-draft-model-path") + 1] == str(
        tmp_path / "mtp"
    )
    assert args[args.index("--speculative-num-draft-tokens") + 1] == "4"
    assert "--disable-flashinfer-autotune" not in args


def test_explicit_context_overrides_profile_default(tmp_path):
    args = cli._profile_args(
        cli.Hardware("x86_64", "NVIDIA RTX PRO 6000 Blackwell", 97_887),
        tmp_path,
        131_072,
    )

    assert args[args.index("--context-length") + 1] == "131072"


def test_dgx_spark_profile_matches_experimental_sglang_shape(tmp_path):
    args = cli._profile_args(
        cli.Hardware("aarch64", "NVIDIA GB10", 122_880),
        tmp_path,
        None,
    )

    assert args[args.index("--context-length") + 1] == "262144"
    assert args[args.index("--max-total-tokens") + 1] == "262144"
    assert args[args.index("--mem-fraction-static") + 1] == "0.85"
    assert args[args.index("--kv-cache-dtype") + 1] == "bfloat16"
    assert args[args.index("--chunked-prefill-size") + 1] == "8192"
    assert args[args.index("--mamba-ssm-dtype") + 1] == "float32"
    assert args[args.index("--speculative-num-steps") + 1] == "3"
    assert args[args.index("--speculative-eagle-topk") + 1] == "1"
    assert args[args.index("--speculative-num-draft-tokens") + 1] == "4"
    assert "--disable-prefill-cuda-graph" in args
    assert "--disable-cuda-graph" not in args


def test_dgx_spark_profile_requires_full_unified_memory(tmp_path):
    hardware = cli.Hardware("aarch64", "NVIDIA GB10", 64_000)

    try:
        cli._profile_args(hardware, tmp_path, None)
    except RuntimeError as exc:
        assert "no validated automatic profile" in str(exc)
    else:
        raise AssertionError("undersized GB10 was accepted as a DGX Spark")


def test_serve_uses_own_python_environment(monkeypatch, tmp_path):
    _snapshot, cuda_home = _prepare_serve(monkeypatch, tmp_path)
    args = SimpleNamespace(
        data_dir=str(tmp_path),
        context=131_072,
        host="127.0.0.1",
        port=30000,
        api_key=None,
        sglang_args=["--", "--log-level", "info"],
    )
    executed = {}

    def capture(path, command, environment):
        executed.update(path=path, command=command, environment=environment)

    monkeypatch.setattr(cli.os, "execve", capture)

    cli._serve(args)

    assert executed["path"] == sys.executable
    assert executed["command"][:3] == [
        sys.executable,
        "-m",
        "sglang.launch_server",
    ]
    assert executed["command"][-2:] == ["--log-level", "info"]
    assert "--" not in executed["command"]
    assert executed["environment"]["PATH"].split(cli.os.pathsep)[0] == str(
        cli.Path(sys.executable).parent
    )
    assert executed["environment"]["MAX_JOBS"] == "1"
    assert executed["environment"]["FLASHINFER_NVCC_THREADS"] == "1"
    assert executed["environment"]["CMAKE_BUILD_PARALLEL_LEVEL"] == "1"
    assert executed["environment"]["CUDA_HOME"] == str(cuda_home)
    assert executed["environment"]["SGLANG_PLUGINS"] == "ssd_stream"


def test_serve_preserves_explicit_compiler_parallelism(monkeypatch, tmp_path):
    _prepare_serve(monkeypatch, tmp_path)
    args = SimpleNamespace(
        data_dir=str(tmp_path),
        context=None,
        host="127.0.0.1",
        port=30000,
        api_key=None,
        sglang_args=[],
    )
    monkeypatch.setenv("MAX_JOBS", "4")
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "3")
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "2")
    executed = {}
    monkeypatch.setattr(
        cli.os,
        "execve",
        lambda _path, _command, environment: executed.update(environment),
    )

    cli._serve(args)

    assert executed["MAX_JOBS"] == "4"
    assert executed["FLASHINFER_NVCC_THREADS"] == "3"
    assert executed["CMAKE_BUILD_PARALLEL_LEVEL"] == "2"


def test_cuda_view_uses_packaged_toolkit_without_copying_it(monkeypatch, tmp_path):
    packaged = tmp_path / "site-packages/nvidia/cu13"
    (packaged / "bin").mkdir(parents=True)
    (packaged / "include").mkdir()
    (packaged / "lib").mkdir()
    (packaged / "bin/nvcc").touch()
    (packaged / "lib/libcudart.so.13").touch()
    runtime = tmp_path / "runtime/venv"
    runtime.mkdir(parents=True)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(cli, "_packaged_cuda_root", lambda: packaged)
    monkeypatch.setattr(cli.sys, "prefix", str(runtime))

    view = cli._cuda_home()

    assert view == tmp_path / "runtime/cuda"
    assert (view / "bin").resolve() == (packaged / "bin").resolve()
    assert (view / "include").resolve() == (packaged / "include").resolve()
    assert (view / "lib64/libcudart.so").resolve() == (
        packaged / "lib/libcudart.so.13"
    ).resolve()


def test_explicit_cuda_home_is_preserved(monkeypatch, tmp_path):
    configured = tmp_path / "cuda"
    configured.mkdir()
    monkeypatch.setenv("CUDA_HOME", str(configured))

    assert cli._cuda_home() == configured.resolve()


def test_missing_compiler_fails_before_model_loading(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda _compiler: None)

    try:
        cli._require_compiler()
    except RuntimeError as exc:
        assert "C++ compiler" in str(exc)
    else:
        raise AssertionError("missing compiler was accepted")
