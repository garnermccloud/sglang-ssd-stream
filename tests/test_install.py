"""Installer plumbing tests; these do not validate the GPU runtime."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.3.0"
PINS = {
    "x86_64": "3df8e1e7dbc5807696622afe2929b6c33c185ca3",
    "aarch64": "0a79825b7baa3e2aafd54e89097a5aba83d00b4e",
}

MOCK = r'''
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
def stage(value):
    with open(os.environ["INSTALL_LOG"], "a") as log:
        log.write(json.dumps([value, args]) + "\n")
    if os.environ.get("FAIL_STAGE") == value:
        sys.exit(42)
if name == "uname":
    print(os.environ.get("MOCK_OS", "Linux") if args == ["-s"] else os.environ["MOCK_ARCH"])
elif name == "uv":
    if args[0] == "venv":
        stage("venv")
        root = pathlib.Path(args[-1])
        if list(root.iterdir()):
            assert "--allow-existing" in args, "uv rejects a nonempty destination without --allow-existing"
        target = root / "bin"
        target.mkdir()
        (target / "python").write_bytes(pathlib.Path(__file__).read_bytes())
        (target / "python").chmod(0o755)
    else:
        step = "wheel" if args[-1].endswith(".whl") else "cache" if "jit-cache" in args[-1] else "cubin" if "cubin" in args[-1] else "dependencies"
        stage(step)
        if step == "wheel":
            python = args[args.index("--python") + 1]
            launcher = pathlib.Path(python).with_name("sglang-ssd-stream")
            # macOS falls back to sh for a script-based shebang interpreter.
            launcher.write_text("#!" + python + '\nexec "' + python + '" --mock-help\n')
            launcher.chmod(0o755)
elif name == "python":
    stage("metadata" if args[0] == "-c" else "help")
    if args[0] == "-c":
        import importlib.metadata, types
        importlib.metadata.version = lambda name: os.environ["MOCK_INSTALLED_VERSION"]
        package = types.ModuleType("sglang_ssd_stream")
        if os.environ.get("MOCK_NATIVE_IMPORT_FAIL") != "1":
            package._io = types.ModuleType("_io")
        sys.modules["sglang_ssd_stream"] = package
        sys.argv = ["-c", *args[2:]]
        exec(args[1])
elif name == "ln":
    stage("link")
    os.symlink(args[-2], args[-1])
elif name == "mv":
    stage("promote")
    assert args[0] == "-Tf"
    os.replace(args[-2], args[-1])
elif name == "curl":
    stage("bootstrap-download")
    assert args[1] == "https://astral.sh/uv/install.sh"
    pathlib.Path(args[-1]).write_text("exit 42\n" if os.environ.get("FAIL_STAGE") == "bootstrap-install" else "exit 0\n")
    if os.environ.get("FAIL_STAGE") != "bootstrap-install":
        uv = pathlib.Path(os.environ["HOME"]) / ".local/bin/uv"
        uv.write_bytes(pathlib.Path(__file__).read_bytes())
        uv.chmod(0o755)
else:
    raise AssertionError(name)
'''


@pytest.fixture
def installer(tmp_path):
    home = tmp_path / "home"
    bin_dir = home / ".local/bin"
    bin_dir.mkdir(parents=True)
    old = home / "old-env/bin/sglang-ssd-stream"
    old.parent.mkdir(parents=True)
    old.write_text("old launcher\n")
    launcher = bin_dir / "sglang-ssd-stream"
    launcher.symlink_to(old)
    mocks = tmp_path / "mocks"
    mocks.mkdir()
    for command in ("uname", "uv", "ln", "mv", "curl"):
        path = mocks / command
        path.write_text(f"#!{sys.executable}\n" + MOCK)
        path.chmod(0o755)
    log = tmp_path / "install.jsonl"
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_DATA_HOME": str(home / "data"),
        "PATH": f"{mocks}:/usr/bin:/bin",
        "MOCK_ARCH": "x86_64",
        "INSTALL_LOG": str(log),
        "FAIL_STAGE": "",
        "SGLANG_SSD_STREAM_WHEEL_URL": "",
        "MOCK_INSTALLED_VERSION": VERSION,
        "MOCK_NATIVE_IMPORT_FAIL": "",
    }

    def run(**overrides):
        if overrides.pop("WITHOUT_UV", False):
            (mocks / "uv").unlink()
        return subprocess.run(
            ["/bin/sh", str(ROOT / "install.sh")],
            env={**env, **overrides}, capture_output=True, text=True,
        )

    return run, launcher, old, log


@pytest.mark.parametrize("stage", ["bootstrap-download", "bootstrap-install"])
def test_bootstrap_failure_preserves_launcher(installer, stage):
    run, launcher, old, _ = installer
    assert run(WITHOUT_UV=True, FAIL_STAGE=stage).returncode != 0
    assert launcher.resolve() == old
    assert old.read_text() == "old launcher\n"


def test_bootstrap_success(installer):
    run, launcher, old, _ = installer
    result = run(WITHOUT_UV=True)
    assert result.returncode == 0, result.stderr
    assert launcher.resolve(strict=True) != old
    assert old.exists()


@pytest.mark.parametrize("stage", [
    "venv", "dependencies", "cubin", "cache", "wheel", "metadata", "help",
    "link", "promote",
])
def test_failure_preserves_launcher(installer, stage):
    run, launcher, old, _ = installer
    result = run(FAIL_STAGE=stage)
    assert result.returncode != 0
    assert launcher.resolve() == old
    assert old.read_text() == "old launcher\n"
    assert not list(launcher.parent.glob(".sglang-ssd-stream.*"))


@pytest.mark.parametrize("arch", PINS)
def test_success_uses_immutable_version_and_final_shebang(installer, arch):
    run, launcher, old, log = installer
    result = run(MOCK_ARCH=arch)
    assert result.returncode == 0, result.stderr
    target = launcher.resolve(strict=True)
    assert target.parent.parent.name.startswith(f"venv-{VERSION}.")
    assert target.read_text().splitlines()[0] == f"#!{target.parent / 'python'}"
    assert old.read_text() == "old launcher\n"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call[0] for call in calls] == [
        "venv", "dependencies", "cubin", "cache", "wheel", "metadata", "help",
        "link", "promote",
    ]
    assert f"sglang @ https://github.com/sgl-project/sglang/archive/{PINS[arch]}.tar.gz#subdirectory=python" in calls[1][1]
    assert "--allow-existing" in calls[0][1]
    assert calls[4][1][-1] == (
        "https://github.com/garnermccloud/sglang-ssd-stream/releases/download/"
        f"v{VERSION}/sglang_ssd_stream-{VERSION}-cp312-cp312-manylinux_2_28_{arch}.whl"
    )
    assert calls[5][1][-1] == VERSION
    assert run(MOCK_ARCH=arch).returncode == 0
    assert launcher.resolve() != target
    assert target.exists()


@pytest.mark.parametrize("arch", PINS)
@pytest.mark.parametrize("validation", ["valid", "wrong-version", "native-import-failure"])
def test_candidate_wheel_override(installer, arch, validation):
    run, launcher, old, log = installer
    url = f"file:///opt/candidate/sglang_ssd_stream-{VERSION}-cp312-cp312-manylinux_2_28_{arch}.whl"
    result = run(
        MOCK_ARCH=arch,
        SGLANG_SSD_STREAM_WHEEL_URL=url,
        MOCK_INSTALLED_VERSION="0.2.0" if validation == "wrong-version" else VERSION,
        MOCK_NATIVE_IMPORT_FAIL="1" if validation == "native-import-failure" else "",
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert next(args for stage, args in calls if stage == "wheel")[-1] == url
    assert next(args for stage, args in calls if stage == "metadata")[-1] == VERSION
    assert old.read_text() == "old launcher\n"
    if validation == "valid":
        assert result.returncode == 0, result.stderr
        assert launcher.resolve(strict=True) != old
    else:
        assert result.returncode != 0
        assert launcher.resolve(strict=True) == old
        assert calls[-1][0] == "metadata"
        assert not list(launcher.parent.glob(".sglang-ssd-stream.*"))


@pytest.mark.parametrize("overrides", [{"MOCK_ARCH": "unsupported"}, {"MOCK_OS": "Darwin"}])
def test_unsupported_platform_preserves_launcher(installer, overrides):
    run, launcher, old, log = installer
    assert run(**overrides).returncode != 0
    assert launcher.resolve() == old
    assert not log.exists()


def test_release_versions_match():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]
    assert project["version"] == cargo["version"] == VERSION
    assert f'VERSION="{VERSION}"' in (ROOT / "install.sh").read_text().splitlines()
    for filename in ("Cargo.lock", "uv.lock"):
        packages = tomllib.loads((ROOT / filename).read_text())["package"]
        assert next(p["version"] for p in packages if p["name"] == project["name"]) == VERSION
