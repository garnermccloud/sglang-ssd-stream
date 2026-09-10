"""Verify the frozen source and recompute the published benchmark medians."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from statistics import median

if not __debug__:
    raise RuntimeError('Run this verifier without Python optimization (-O/PYTHONOPTIMIZE).')

PASSES = ('control-before', 'candidate-before-functional', 'candidate-after-functional', 'control-after')
WORKLOADS = ('list', 'prose', 'code', 'reasoning')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts or not parts.parts:
        raise ValueError('Invalid bundle path: ' + relative)
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes the selected root: ' + relative)
    return path


def verify(root, base_root=None, target_root=None):
    manifest = read(root / 'manifest.json')
    assert manifest['format'] == 1 and manifest['weights_changed'] is False
    entries = manifest['files']
    assert len(entries) == 17
    assert len({e['file'] for e in entries}) == len(entries)
    assert len({e['destination'] for e in entries}) == len(entries)
    for entry in entries:
        source = safe_path(root, entry['file'])
        assert source.is_file() and source.stat().st_size == entry['bytes'], entry['file']
        assert digest(source) == entry['sha256'], entry['file']
        assert entry['destination'].startswith('/')
        assert entry['file'] == 'payload' + entry['destination']
        selected_root = base_root if base_root is not None else target_root
        if selected_root is not None:
            destination = safe_path(selected_root, entry['destination'][1:])
            expected = entry['baseline_sha256'] if base_root is not None else entry['sha256']
            actual = digest(destination) if destination.is_file() else None
            assert actual == expected, f'Incompatible runtime file: {entry["destination"]}'

    summary = read(root / 'evidence/summary.json')
    runs = {name: read(root / 'evidence' / (name + '.json')) for name in PASSES}
    count = 0
    for records in runs.values():
        assert set(records) == set(WORKLOADS)
        for rows in records.values():
            assert len(rows) == 3
            for row in rows:
                count += 1
                assert row['tokens'] == 1024 and row['num_retractions'] == 0
                assert row['finish_reason']['type'] == 'length'
                assert math.isclose(row['wall_tps'], row['tokens'] / row['wall_s'], rel_tol=1e-12)
                assert sum(row['correct_drafts_histogram']) == row['verify_rounds']
                assert sum(i * n for i, n in enumerate(row['correct_drafts_histogram'])) == row['accepted_drafts']
    assert count == 48
    result = {}
    for workload in WORKLOADS:
        control = runs[PASSES[0]][workload] + runs[PASSES[3]][workload]
        candidate = runs[PASSES[1]][workload] + runs[PASSES[2]][workload]
        values = {}
        for arm, rows in [('control', control), ('candidate', candidate)]:
            values[arm] = median(row['wall_tps'] for row in rows)
            assert math.isclose(values[arm], summary['comparison'][workload][arm]['wall_tps'], rel_tol=1e-12)
            if arm == 'candidate':
                assert all(row['conditional_acceptance'] is None and row['proposed_drafts'] is None for row in rows)
        delta = 100 * (values['candidate'] / values['control'] - 1)
        assert math.isclose(delta, summary['comparison'][workload]['wall_tps_delta_percent'], rel_tol=1e-12)
        result[workload] = {**values, 'change_percent': delta}
    functional = read(root / 'evidence/functional.json')
    assert len(functional) == summary['functional_passed'] == 12
    assert all(row['passed'] is True for row in functional)
    return {'source_files_verified': len(entries), 'measurements_verified': count, 'comparison': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--base-root', type=Path, help='Read-only compatibility check of an unpacked baseline root filesystem')
    group.add_argument('--target-root', type=Path, help='Read-only check of an already overlaid root filesystem')
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent, args.base_root, args.target_root), indent=2))


if __name__ == '__main__':
    main()
