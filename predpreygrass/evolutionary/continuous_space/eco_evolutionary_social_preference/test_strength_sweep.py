"""Runner preflight checks must fail before starting expensive inference."""
import hashlib
import json
import sys

import pytest

from . import strength_sweep
from .summarize import METRICS


def baseline_file(tmp_path, mismatched=False):
    checkpoint = tmp_path / 'checkpoint'
    def row(seed):
        return {'seed': seed, 'movement': {'means': {key: 1 for key in METRICS}}}
    path = tmp_path / 'baseline.json'
    path.write_text(json.dumps({'checkpoint': str(checkpoint), 'strength': 1,
                               'results': {'0.0': [row(42)], '1.0': [row(42)],
                                           '-1.0': [row(43 if mismatched else 42)]}}))
    return checkpoint, path


def test_optional_checkpoint_inputs_are_fingerprinted_as_absent(tmp_path):
    checkpoint = tmp_path / 'run/checkpoint'
    checkpoint.mkdir(parents=True)
    metadata = checkpoint / 'class_and_ctor_args.pkl'
    metadata.write_bytes(b'fallback metadata')
    result = strength_sweep.fingerprint(checkpoint)
    assert result[str((checkpoint / 'gene_pools.json').resolve())] is None
    assert result[str(metadata.resolve())] == hashlib.sha256(metadata.read_bytes()).hexdigest()


def test_mismatched_baseline_seeds_fail_before_inference(tmp_path, monkeypatch):
    checkpoint, baseline = baseline_file(tmp_path, mismatched=True)
    monkeypatch.setattr(sys, 'argv', ['sweep', str(checkpoint), '--baseline', str(baseline)])
    monkeypatch.setattr(strength_sweep, 'load_modules', lambda _: pytest.fail('inference started'))
    with pytest.raises(ValueError, match='identical nonempty seed sets'):
        strength_sweep.main()


def test_formatted_strength_collisions_fail_before_creating_outputs(tmp_path, monkeypatch):
    checkpoint, baseline = baseline_file(tmp_path)
    output = tmp_path / 'outputs'
    monkeypatch.setattr(sys, 'argv', ['sweep', str(checkpoint), '--baseline', str(baseline),
                                    '--output-dir', str(output), '--strengths', '2.0000001', '2.0000002'])
    with pytest.raises(SystemExit) as exc:
        strength_sweep.main()
    assert exc.value.code == 2
    assert not output.exists()
