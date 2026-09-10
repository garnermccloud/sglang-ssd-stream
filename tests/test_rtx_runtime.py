import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sglang_ssd_stream import rtx_runtime as runtime


def sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def installation(monkeypatch, tmp_path):
    package = tmp_path / 'plugin'
    installed = tmp_path / 'site-packages'
    (installed / 'sglang').mkdir(parents=True)
    (installed / 'sglang/__init__.py').write_text('')
    (installed / 'sglang/unchanged.py').write_text('VALUE = "unchanged"\n')
    manifest = runtime._manifest()
    for index, entry in enumerate(manifest['files']):
        before = f'VALUE = "before-{index}"\n'.encode()
        after = f'VALUE = "after-{index}"\n'.encode()
        source = package / 'rtx_payload' / entry['path']
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(after)
        entry['sha256'] = sha(after)
        if entry['base_sha256'] is not None:
            target = installed / entry['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(before)
            entry['base_sha256'] = sha(before)
    (package / 'rtx_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(runtime, 'PACKAGE', package)
    monkeypatch.setattr(runtime, '_installed_root', lambda: installed)
    return package, installed, manifest


def test_packaged_payload_matches_qualified_manifest():
    data = runtime._manifest()
    runtime._validate(runtime.PACKAGE / 'rtx_payload', data['files'], baseline=False)
    assert len(runtime.module_hashes()) == 17
    policy = json.loads((runtime.PACKAGE / 'rtx_adaptive.json').read_text())
    assert policy['1']['candidate_steps'] == [3, 7]


def test_shared_read_phase_contract_is_packaged():
    # Exercise the CPU-only method without importing the GPU runtime.
    root = runtime.PACKAGE / 'rtx_payload/sglang/srt/speculative'
    for filename in ['spec_info.py', 'spec_registry.py']:
        tree = ast.parse((root / filename).read_text())
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == 'is_last_shared_read_phase')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), filename, 'exec'), namespace)
        class Phase:
            def is_target_verify(self):
                return False
            def is_draft_extend_v2(self):
                return True
        class Algorithm:
            def is_dflash_family(self):
                return False
            def is_dspark(self):
                return False
        assert namespace['is_last_shared_read_phase'](Algorithm(), Phase()) is True


def test_runtime_is_private_and_reusable(installation, tmp_path):
    _, installed, manifest = installation
    output = runtime.prepare_runtime(tmp_path / 'data')
    assert runtime.prepare_runtime(tmp_path / 'data') == output
    assert (output / 'sglang/unchanged.py').is_symlink()
    for entry in manifest['files']:
        copied = output / entry['path']
        assert not copied.is_symlink()
        assert sha(copied.read_bytes()) == entry['sha256']
        if entry['base_sha256'] is not None:
            assert sha((installed / entry['path']).read_bytes()) == entry['base_sha256']
    assert not list(output.parent.glob('.building-*'))


def test_spawned_python_imports_selected_runtime(installation, tmp_path):
    output = runtime.prepare_runtime(tmp_path / 'data')
    environment = {**os.environ, 'PYTHONPATH': str(output)}
    # A conflicting package in the launch directory must not outrank the view.
    conflict = tmp_path / 'sglang'
    conflict.mkdir()
    (conflict / '__init__.py').write_text('raise RuntimeError("wrong working-directory package")\n')
    result = subprocess.run([sys.executable, '-P', '-c',
                             'from sglang.srt.speculative import spec_utils; import qsa_pack; '
                             'from sglang import unchanged; '
                             'assert spec_utils.VALUE.startswith("after-"); '
                             'assert qsa_pack.VALUE.startswith("after-"); '
                             'assert unchanged.VALUE == "unchanged"'],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_inherited_profile_restores_prior_mode():
    environment = {runtime.ROOT_ENV: '/private/rtx', runtime.PROFILE_ENV: '1',
                   runtime.PREVIOUS_MODE_ENV: 'dynamic',
                   'PYTHONPATH': '/private/rtx:/user/modules',
                   'SGLANG_RAGGED_VERIFY_MODE': 'static'}
    runtime.clear_inherited_profile(environment)
    assert environment == {'PYTHONPATH': '/user/modules', 'SGLANG_RAGGED_VERIFY_MODE': 'dynamic'}


def test_plugin_failures_are_fatal(monkeypatch):
    from sglang_ssd_stream import plugin
    def fail():
        raise RuntimeError('incompatible module')
    monkeypatch.setattr(plugin, '_register', fail)
    with pytest.raises(SystemExit, match='incompatible module'):
        # Match the upstream loader's Exception boundary, not BaseException.
        try:
            plugin.register()
        except Exception:
            pytest.fail('The loader swallowed a compatibility failure')


def test_wrong_base_fails_before_runtime_creation(installation, tmp_path):
    _, installed, manifest = installation
    entry = next(e for e in manifest['files'] if e['base_sha256'] is not None)
    (installed / entry['path']).write_text('incompatible')
    with pytest.raises(RuntimeError, match='installed base'):
        runtime.prepare_runtime(tmp_path / 'data')
    assert not (tmp_path / 'data').exists()


def test_corrupt_payload_is_rejected(installation, tmp_path):
    package, _, manifest = installation
    (package / 'rtx_payload' / manifest['files'][0]['path']).write_text('corrupt')
    with pytest.raises(RuntimeError, match='RTX runtime'):
        runtime.prepare_runtime(tmp_path / 'data')


def test_corrupt_cached_runtime_is_not_reused(installation, tmp_path):
    _, _, manifest = installation
    output = runtime.prepare_runtime(tmp_path / 'data')
    (output / manifest['files'][0]['path']).write_text('corrupt')
    with pytest.raises(RuntimeError, match='RTX runtime'):
        runtime.prepare_runtime(tmp_path / 'data')


def test_different_installations_get_different_views(installation, tmp_path, monkeypatch):
    _, installed, _ = installation
    first = runtime.prepare_runtime(tmp_path / 'data')
    import shutil
    second_install = tmp_path / 'second-site-packages'
    shutil.copytree(installed, second_install)
    monkeypatch.setattr(runtime, '_installed_root', lambda: second_install)
    second = runtime.prepare_runtime(tmp_path / 'data')
    assert first != second


def test_bad_manifest_path_is_rejected(installation, tmp_path):
    package, _, manifest = installation
    manifest['files'][0]['path'] = '../outside.py'
    (package / 'rtx_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='path'):
        runtime.prepare_runtime(tmp_path / 'data')
