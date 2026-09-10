"""Process-isolated RTX runtime view; never modify the installed SGLang tree."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

PROFILE_ENV = 'SGLANG_SSD_STREAM_RTX_PROFILE'
ROOT_ENV = 'SGLANG_SSD_STREAM_RTX_ROOT'
PREVIOUS_MODE_ENV = 'SGLANG_SSD_STREAM_PREVIOUS_RAGGED_MODE'
PACKAGE = Path(__file__).resolve().parent


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict:
    data = json.loads((PACKAGE / 'rtx_manifest.json').read_text())
    if data.get('version') != 1 or len(data.get('files', [])) != 15:
        raise RuntimeError('Invalid packaged RTX runtime manifest')
    seen = set()
    for entry in data['files']:
        path = PurePosixPath(entry['path'])
        if path.is_absolute() or '..' in path.parts or str(path) in seen:
            raise RuntimeError('Invalid RTX runtime path')
        if not (str(path).startswith('sglang/') or str(path) == 'qsa_pack.py'):
            raise RuntimeError('RTX runtime entry is outside its module scope')
        seen.add(str(path))
    return data


def module_hashes() -> dict[str, str]:
    return {entry['path'][:-3].replace('/', '.'): entry['sha256']
            for entry in _manifest()['files']}


def clear_inherited_profile(environment: dict[str, str]) -> None:
    root = environment.pop(ROOT_ENV, None)
    environment.pop(PROFILE_ENV, None)
    previous_mode = environment.pop(PREVIOUS_MODE_ENV, '')
    if root is None:
        return
    paths = [path for path in environment.get('PYTHONPATH', '').split(os.pathsep) if path != root]
    if paths and paths != ['']:
        environment['PYTHONPATH'] = os.pathsep.join(paths)
    else:
        environment.pop('PYTHONPATH', None)
    if previous_mode:
        environment['SGLANG_RAGGED_VERIFY_MODE'] = previous_mode
    else:
        environment.pop('SGLANG_RAGGED_VERIFY_MODE', None)


def _validate(root: Path, entries: list[dict], *, baseline: bool) -> None:
    for entry in entries:
        expected = entry['base_sha256'] if baseline else entry['sha256']
        if baseline and expected is None:
            continue
        path = root / entry['path']
        if not path.is_file() or _digest(path) != expected:
            state = 'installed base' if baseline else 'RTX runtime'
            raise RuntimeError(f'Incompatible {state} file: {entry["path"]}')


def _installed_root() -> Path:
    spec = importlib.util.find_spec('sglang')
    if spec is None or spec.origin is None:
        raise RuntimeError('The pinned SGLang package is not installed')
    return Path(spec.origin).resolve().parent.parent


def prepare_runtime(data_dir: Path) -> Path:
    """Return a validated import root inherited by all spawned server workers."""
    data = _manifest()
    entries = data['files']
    installed = _installed_root()
    payload = PACKAGE / 'rtx_payload'
    _validate(installed, entries, baseline=True)
    _validate(payload, entries, baseline=False)
    key = hashlib.sha256((str(installed) + json.dumps(data, sort_keys=True)).encode()).hexdigest()[:24]
    cache = data_dir / 'rtx-runtimes'
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / key
    with (cache / '.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists():
            _validate(target, entries, baseline=False)
            return target
        temporary = Path(tempfile.mkdtemp(prefix='.building-', dir=cache))
        try:
            # Unchanged files stay linked to this versioned environment. Changed
            # files are separate copies, selected only through this import root.
            shutil.copytree(installed / 'sglang', temporary / 'sglang',
                            copy_function=os.symlink, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            for entry in entries:
                destination = temporary / entry['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.unlink(missing_ok=True)
                shutil.copyfile(payload / entry['path'], destination)
            _validate(temporary, entries, baseline=False)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return target
