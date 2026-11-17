# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Threefry noise path in SpyreTopKTopPSampler.

Covers:
- Threefry path is taken when not use_fp64_gumbel and not generators.
- Philox fallback is taken when generators are present.
- Philox fallback is taken when use_fp64_gumbel=True.
- Distribution check: generate_exp_spyre on CPU produces Exp(1) samples.
- Sampled token ids are within [0, vocab) range.
"""

import unittest.mock

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.rng.exp_spyre import ExpSpyreGenerator, generate_exp_spyre
from spyre_inference.v1.sample import topk_topp_sampler
from spyre_inference.v1.sample.topk_topp_sampler import SpyreTopKTopPSampler


@pytest.fixture(autouse=True)
def _no_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable C++ sampling kernels so the Python fallback path runs."""
    monkeypatch.setattr(topk_topp_sampler, "has_sampling_kernels", lambda: False)


def _sampler(fp64: bool = False) -> SpyreTopKTopPSampler:
    return SpyreTopKTopPSampler("raw_logprobs", fp64)


# ---------------------------------------------------------------------------
# Path-selection tests: check which RNG function is actually called.
# ---------------------------------------------------------------------------


def _spy_generator_class():
    """Patch the ExpSpyreGenerator the sampler uses with a constructor spy."""
    return unittest.mock.patch.object(
        topk_topp_sampler,
        "ExpSpyreGenerator",
        wraps=ExpSpyreGenerator,
    )


def test_threefry_path_taken_without_generators() -> None:
    """When no generators and not fp64, forward_native builds an ExpSpyreGenerator."""
    logits = torch.randn(4, 128, dtype=torch.float32)
    with _spy_generator_class() as spy:
        _sampler().forward_native(logits.clone(), {}, None, None)
    spy.assert_called_once()
    # The generator pads internally, so it is sized for exactly B * V.
    call_n = spy.call_args[0][0]
    assert call_n == logits.numel(), f"expected n={logits.numel()}, got {call_n}"


def test_philox_path_taken_with_generators() -> None:
    """When generators is non-empty, no ExpSpyreGenerator may be built."""
    logits = torch.randn(4, 128, dtype=torch.float32)
    gens = {0: torch.Generator().manual_seed(42)}
    with _spy_generator_class() as spy:
        _sampler().forward_native(logits.clone(), gens, None, None)
    spy.assert_not_called()


def test_philox_path_taken_with_fp64_gumbel() -> None:
    """When use_fp64_gumbel=True, no ExpSpyreGenerator may be built."""
    logits = torch.randn(4, 128, dtype=torch.float32)
    with _spy_generator_class() as spy:
        _sampler(fp64=True).forward_native(logits.clone(), {}, None, None)
    spy.assert_not_called()


# ---------------------------------------------------------------------------
# Persistent generator: counter advance, reuse for smaller steps, growth.
# ---------------------------------------------------------------------------


def test_noise_differs_between_steps() -> None:
    """Successive draws of one sampler use different invocation indices."""
    logits = torch.randn(4, 128, dtype=torch.float32)
    sampler = _sampler()
    first = sampler._make_noise(logits)
    second = sampler._make_noise(logits)
    assert first.shape == logits.shape and first.dtype == torch.float32
    assert not torch.equal(first, second), "noise repeated between steps"
    # A fresh sampler starts at invocation 0 again (CPU stream is deterministic).
    assert torch.equal(_sampler()._make_noise(logits), first)


def test_noise_matches_padded_logits_shape() -> None:
    """Noise has the logits' shape; on device ``SpyreSampler`` pads them to 64s first."""
    assert topk_topp_sampler.padded_vocab(49159) == 49216
    logits = torch.randn(3, topk_topp_sampler.padded_vocab(49159), dtype=torch.float32)
    assert _sampler()._make_noise(logits).shape == logits.shape


def test_unaligned_vocab_tokens_in_range() -> None:
    """A vocab that is not a multiple of 64 only ever yields real token ids."""
    logits = torch.randn(3, 1000 + 7, dtype=torch.float32)
    tokens, _ = _sampler().forward_native(logits, {}, None, None)
    assert (tokens >= 0).all() and (tokens < logits.shape[1]).all()


def test_smaller_step_reuses_generator() -> None:
    """A step with fewer elements slices the existing generator, no rebuild."""
    big = torch.randn(8, 128, dtype=torch.float32)
    small = torch.randn(2, 128, dtype=torch.float32)
    sampler = _sampler()
    with _spy_generator_class() as spy:
        sampler._make_noise(big)
        noise = sampler._make_noise(small)
    spy.assert_called_once()
    assert noise.shape == small.shape


def test_larger_step_grows_generator_and_continues_counter() -> None:
    """A larger step rebuilds the generator at the current draw count."""
    small = torch.randn(2, 128, dtype=torch.float32)
    big = torch.randn(8, 128, dtype=torch.float32)
    sampler = _sampler()
    with _spy_generator_class() as spy:
        sampler._make_noise(small)  # invocation 0
        grown = sampler._make_noise(big)  # new generator, starts at invocation 1
    assert spy.call_count == 2
    assert spy.call_args_list[1].kwargs["start_invocation"] == 1
    expected = ExpSpyreGenerator(
        big.numel(),
        rounds=topk_topp_sampler._THREEFRY_ROUNDS,
        device="cpu",
        start_invocation=1,
    )
    assert torch.equal(grown, expected.generate()[: big.numel()].reshape(big.shape))
    # Not a repeat of the stream at invocation 0.
    first = ExpSpyreGenerator(big.numel(), device="cpu").generate()
    assert not torch.equal(grown, first[: big.numel()].reshape(big.shape))


# ---------------------------------------------------------------------------
# Correctness tests: output values are valid.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows,vocab", [(1, 32), (8, 1000)])
def test_threefry_tokens_in_range(rows: int, vocab: int) -> None:
    """Sampled tokens must be within [0, vocab)."""
    logits = torch.randn(rows, vocab, dtype=torch.float32)
    tokens, _ = _sampler().forward_native(logits.clone(), {}, None, None)
    assert tokens.shape == (rows,)
    assert (tokens >= 0).all() and (tokens < vocab).all()


def test_threefry_draw_is_deterministic_for_same_key() -> None:
    """Two calls to generate_exp_spyre with the same args produce the same tensor."""
    a = generate_exp_spyre(256, device="cpu")
    b = generate_exp_spyre(256, device="cpu")
    assert torch.equal(a, b), "generate_exp_spyre should be deterministic"


def test_threefry_draw_finite_and_positive() -> None:
    """generate_exp_spyre on CPU must produce finite positive fp32 values."""
    out = generate_exp_spyre(10000, device="cpu")
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all(), "output contains NaN or Inf"
    assert (out > 0).all(), "output contains non-positive values"


# ---------------------------------------------------------------------------
# Distribution test: Exp(1) moments without hardware.
# ---------------------------------------------------------------------------


def test_generate_exp_spyre_cpu_distribution() -> None:
    """generate_exp_spyre on CPU should produce Exp(1) samples.

    Checks: mean ≈ 1.0 and std ≈ 1.0 within a loose tolerance.
    With n=100_000 and 5-sigma bounds the failure probability is < 3e-7.
    """
    n = 100_000
    out = generate_exp_spyre(n, device="cpu").double()
    mean = out.mean().item()
    std = out.std().item()
    # 5-sigma bounds: std(mean) = 1/sqrt(n) ≈ 0.00316, std(std) ≈ 1/sqrt(2n) ≈ 0.00224
    assert abs(mean - 1.0) < 0.02, f"mean={mean:.4f}, expected ≈ 1.0"
    assert abs(std - 1.0) < 0.02, f"std={std:.4f}, expected ≈ 1.0"


# ---------------------------------------------------------------------------
# Seeded-generator path is unaffected by Threefry change.
# ---------------------------------------------------------------------------


def test_seeded_generators_still_reproducible() -> None:
    """The per-request seeded path must remain reproducible after the Threefry change."""
    logits = torch.randn(4, 32000, dtype=torch.float32)

    def draw() -> torch.Tensor:
        gens = {0: torch.Generator().manual_seed(99), 2: torch.Generator().manual_seed(7)}
        return _sampler().forward_native(logits.clone(), gens, None, None)[0]

    a, b = draw(), draw()
    # Seeded rows (0 and 2) must be identical across calls.
    assert torch.equal(a[[0, 2]], b[[0, 2]]), "seeded rows differ between calls"


# ---------------------------------------------------------------------------
# The argmax iota is pre-permuted to match the staggered logits (the permutation
# itself is tested in torch-spyre).
# ---------------------------------------------------------------------------


def test_iota_constants_read_back_as_arange_on_device() -> None:
    if not spyre_available():
        pytest.skip("Spyre device not available")
    width = 4096
    iota = topk_topp_sampler.iota_constants(width, torch.device("spyre"))
    assert torch.equal(iota.cpu(), torch.arange(width, dtype=torch.float32))
