"""Tests for _build_mix_weights: expected-fraction realization and guards.

Run:  .venv/bin/python -m pytest src/openpi/training/mix_weights_test.py -q
"""

import numpy as np
import pytest
import torch

from openpi.training.data_loader import _build_mix_weights


class _FakeMeta:
    def __init__(self, tasks):
        self.tasks = tasks


class _FakeLeRobot:
    def __init__(self, task_index, tasks):
        self.hf_dataset = {"task_index": np.asarray(task_index)}
        self.meta = _FakeMeta(tasks)


def _fake_dataset(n_new=21_000, n_mix=6_300):
    # Mimics a combined repo: task 0 = new task (dumbbell), 1 = mix (box).
    tasks = {0: "drop the dumbbell in the trash can", 1: "put the box on the shelf"}
    idx = np.concatenate([np.zeros(n_new, dtype=int), np.ones(n_mix, dtype=int)])
    return _FakeLeRobot(idx, tasks)


@pytest.mark.parametrize("fraction", [0.05, 0.20, 0.50])
def test_realized_fraction_within_2pct(fraction):
    ds = _fake_dataset()
    w = _build_mix_weights(ds, keywords=("box",), fraction=fraction, base_weights=None)
    gen = torch.Generator()
    gen.manual_seed(0)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=w, num_samples=len(w), replacement=True, generator=gen
    )
    picks = np.fromiter(iter(sampler), dtype=int)
    is_mix = picks >= 21_000
    realized = is_mix.mean()
    assert abs(realized - fraction) < 0.02, f"realized {realized:.4f} vs target {fraction}"


def test_expected_fraction_exact_in_weights():
    ds = _fake_dataset()
    for fraction in (0.05, 0.20, 0.50):
        w = np.asarray(_build_mix_weights(ds, keywords=("box",), fraction=fraction))
        mix_share = w[21_000:].sum() / w.sum()
        assert abs(mix_share - fraction) < 1e-9


def test_composes_with_base_weights_preserving_fraction():
    ds = _fake_dataset()
    rng = np.random.default_rng(0)
    base = torch.as_tensor(rng.uniform(0.2, 1.2, size=27_300), dtype=torch.double)
    w = np.asarray(_build_mix_weights(ds, keywords=("box",), fraction=0.2, base_weights=base))
    assert abs(w[21_000:].sum() / w.sum() - 0.2) < 1e-9
    # Within-group relative ordering follows base weights.
    ratios = w[:21_000] / np.asarray(base)[:21_000]
    assert np.allclose(ratios, ratios[0])


def test_guards():
    ds = _fake_dataset()
    with pytest.raises(ValueError):
        _build_mix_weights(ds, keywords=("box",), fraction=1.5)
    with pytest.raises(ValueError):
        _build_mix_weights(ds, keywords=(), fraction=0.2)
    with pytest.raises(RuntimeError):
        _build_mix_weights(ds, keywords=("banana",), fraction=0.2)
