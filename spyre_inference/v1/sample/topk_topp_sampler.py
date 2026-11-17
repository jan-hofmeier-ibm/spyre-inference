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

import torch
from vllm.config import CompilationMode, get_cached_compilation_config
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p_pytorch,
)

from spyre_inference.custom_ops.lazy_compile import CompileOutermost, maybe_compile
from spyre_inference.rng.exp_spyre import ExpSpyreGenerator
from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels


def apply_top_k_top_p_sort_free(
    logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor
) -> torch.Tensor:
    """``apply_top_k_top_p_pytorch`` without the full-vocab sort.

    Top-p only ever keeps top-k survivors, so it runs on a ``topk`` window.
    vllm keeps every token tied with a row's k-th value, and fp16 logits tie
    there often, so the window carries slack and widens to the exact tie count
    when ties run past it.

    The logits tensor is updated in-place."""
    max_k = int(k.max())
    vocab = logits.shape[1]
    if max_k >= vocab:
        # k covers the full vocab — no filtering needed.
        return apply_top_k_top_p_pytorch(logits, k, p)
    if min(2 * max_k, vocab) > 128 and logits.device.type == "spyre":
        # k is too large for Spyre's topk kernel (max k=128) but small enough
        # that skipping the filter still leaves a meaningful top-k window.
        # Return logits unfiltered; the Gumbel draw selects from all tokens.
        return logits
    vals, idx = logits.topk(min(2 * max_k, vocab), dim=-1)
    kth = vals.gather(1, k.long().unsqueeze(1) - 1)
    if (vals[:, -1:] == kth).any():
        vals, idx = logits.topk(int((logits >= kth).sum(dim=-1).max()), dim=-1)
    vals.masked_fill_(vals < kth, -float("inf"))
    vals, idx = vals.flip(-1), idx.flip(-1)
    probs_sum = vals.softmax(dim=-1).cumsum_(dim=-1)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    vals.masked_fill_(top_p_mask, -float("inf"))
    return logits.fill_(-float("inf")).scatter_(1, idx, vals)


# Device lanes per stick pair: the noise is generated with rows of this alignment so the
# flat generator output reshapes to [B, padded_vocab(V)] without a row starting
# mid-stick. (The logits' own device layout pads each row the same way.)
_STICK = 64

# Threefry-4x16 rounds for the sampling noise. 8 passes the KS / chi-square /
# autocorrelation gates for LLM sampling at n=800k and is ~1.27x faster than 13
# (spyre-threefry/docs/results/bench_rounds_results.md).
_THREEFRY_ROUNDS = 8


def padded_vocab(vocab: int) -> int:
    """``vocab`` rounded up to a whole number of 64-element sticks."""
    return -(-vocab // _STICK) * _STICK


_IOTA: dict[tuple[int, torch.device], torch.Tensor] = {}


def iota_constants(width: int, device: torch.device) -> torch.Tensor:
    """fp32 ``[width]`` column indices ``0..width-1``, uploaded once.

    On Spyre the score is fp32 with the staggered ``DL16_TO_FP32`` element arrangement
    (EA), where a pointwise op pairs elements by *physical slot*. A plain ``arange`` is
    STANDARD-EA and lands each index in the wrong slot (right stick, wrong lane), so
    the host data is pre-permuted: slot ``s`` of stick ``g`` holds ``g*64 + perm[s]``,
    with ``perm = torch_spyre.element_arrangement.stick_permutation(DL16_TO_FP32)``.
    The tensor is then retagged ``DL16_TO_FP32``, which makes the pointwise op accept
    it as an operand of the score and reads it back as ``arange(width)``.
    ``width`` must be a whole number of sticks. Exact in fp32 up to 2**24.
    """
    key = (width, device)
    if key not in _IOTA:
        ids = torch.arange(width, dtype=torch.int32)
        if device.type != "spyre":
            _IOTA[key] = ids.to(torch.float32).to(device)
        else:
            from torch_spyre._C import ElementArrangement
            from torch_spyre.element_arrangement import stick_permutation

            assert width % _STICK == 0, f"width {width} is not a multiple of {_STICK}"
            host = (ids // _STICK * _STICK).to(torch.float32) + torch.tensor(
                stick_permutation(ElementArrangement.DL16_TO_FP32), dtype=torch.float32
            )[ids % _STICK]
            iota = host.to(device)
            _retag_element_arrangement(iota, ElementArrangement.DL16_TO_FP32)
            _IOTA[key] = iota
    return _IOTA[key]


def first_max_index(score: torch.Tensor, iota: torch.Tensor) -> torch.Tensor:
    """First column attaining each row's maximum, as int64 ``[B]``; like ``argmax``.

    ``aten.argmax`` is a CPU fallback on Spyre, so the index is found with ``amax``,
    an equality mask and an ``amin`` over the iota. The result is a stick-dimension
    reduction: a *sparse* ``[1, B, 32]`` layout (one valid element per stick) that
    other sparse reductions share, but that torch-spyre cannot convert to a dense
    ``[B]`` layout. Everything vLLM combines it with (see ``SpyreSampler``) is
    therefore made with the same kind of reduction.

    The fill for non-maximal columns is ``iota + width`` rather than the scalar
    ``width``: a scalar lowers to a STANDARD-EA buffer, and torch-spyre rejects a
    STANDARD and a staggered full-size operand in one pointwise op. The fill is still
    ``>= width``, so the ``amin`` is unaffected.
    """
    top = score.amax(dim=-1, keepdim=True)
    fill = iota + float(score.shape[1])
    return torch.where(score == top, iota, fill).amin(dim=-1).to(torch.int64)


def _retag_element_arrangement(q: torch.Tensor, ea) -> None:
    """Tag ``q`` with element arrangement ``ea``; no data is touched.

    The logits are ``DL16_TO_FP32`` (staggered) and the noise is STANDARD.
    torch-spyre rejects a pointwise op on a staggered and a STANDARD full tensor
    (``propagate_layouts`` case 3.3) and nothing converts the EA of an fp32 tensor.
    Gumbel-max only needs the noise to be iid, so reading ``q``'s physical slots
    under the logits' arrangement is as good as permuting it: the permutation of
    iid draws is again iid draws. This is a metadata change on the tensor, with no
    device work and no graph break.
    """
    from torch_spyre._C import get_spyre_tensor_layout, set_spyre_tensor_layout

    set_spyre_tensor_layout(q, get_spyre_tensor_layout(q).with_element_arrangement(ea))


class SpyreGumbelArgmax(CompileOutermost):
    """Compiled Gumbel-max draw: ``argmax(logits - log(q))``, entirely on device.

    ``logits_fp32`` is staggered (``DL16_TO_FP32``, from ``to_dtype_d2d``) and ``q``
    is STANDARD-EA fp32, both ``[B, Vp]`` with ``Vp`` a multiple of 64 (``SpyreSampler``
    pads the logits with ``-inf``; slicing a staggered fp32 tensor to a partial last
    stick is silently wrong on the device, so nothing is sliced here).
    ``sample`` retags ``q`` with the logits' arrangement before
    the graph runs (see ``_retag_element_arrangement``), so the subtraction is
    between operands of one arrangement. The index comes from ``first_max_index``.

    ``q`` is generated **outside** this compiled graph (``ExpSpyreGenerator``
    advances on device and runs its own compiled graph outside
    ``SpyreGumbelArgmax.forward``), so there is no graph break here.

    One instance is created per ``(rows, vocab)`` shape — ``@maybe_compile``
    caches exactly one kernel per instance, and ``dynamic=False`` requires a
    fixed shape.  ``SpyreTopKTopPSampler`` holds a dict keyed by shape.

    On CPU (unit tests, fallback) ``spyre_compile_enabled`` is False so
    ``@maybe_compile`` runs the body as plain Python — no compilation needed.

    ``allow_inference_recompiles = True`` suppresses the compile-guard warning
    for shapes that first appear after warmup (e.g. an unexpected batch size).
    Each new ``(B, V)`` shape creates its own instance, so this class-level
    flag just opts every instance out of the guard rather than allowlisting each
    one from the outside.
    """

    allow_inference_recompiles: bool = True

    def __init__(self, compile_enabled: bool) -> None:
        # Instances are created lazily, on the first batch of a new shape (warmup
        # included), where the vLLM config context is no longer live and
        # ``CompileOutermost.arm_outer_compile`` would raise. The caller samples the
        # mode at sampler construction and passes it in. ``arm_outer_compile`` is
        # idempotent and returns early once ``spyre_compile_enabled`` is set.
        self.spyre_compile_enabled = compile_enabled
        self.spyre_compiled_kernel = None
        super().__init__()

    def sample(self, logits_fp32: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Run the draw. Retags ``q`` and fetches the iota constant outside the graph."""
        if logits_fp32.device.type != "spyre":
            return self.forward(logits_fp32, q, None)
        from torch_spyre._C import get_spyre_tensor_layout

        _retag_element_arrangement(q, get_spyre_tensor_layout(logits_fp32).element_arrangement)
        return self.forward(
            logits_fp32, q, iota_constants(logits_fp32.shape[1], logits_fp32.device)
        )

    @maybe_compile
    def forward(
        self,
        logits_fp32: torch.Tensor,
        q: torch.Tensor,
        iota: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return ``argmax(logits_fp32 - log(q))`` as a 1-D int64 tensor.

        Args:
            logits_fp32: ``[B, Vp]`` fp32 (staggered EA on Spyre); pad columns are
                ``-inf`` so they never win.
            q: ``[B, Vp]`` fp32 Exp(1) variates, retagged to the logits' EA on Spyre.
            iota: ``[Vp]`` fp32 column indices, pre-permuted and tagged staggered
                (see ``iota_constants``). ``None`` off Spyre.

        Returns:
            ``[B]`` sampled token indices, int64 (what upstream's greedy path
            produces, so the two combine in vLLM's final ``torch.where``).
        """
        score = logits_fp32 - q.log()
        if logits_fp32.device.type != "spyre":
            return score.argmax(dim=-1).view(-1)
        assert iota is not None
        return first_max_index(score, iota)


class SpyreTopKTopPSampler(TopKTopPSampler):
    """Sort-free top-k plus a fused Gumbel-max draw for random sampling.

    Upstream only takes the sort-free top-k path under ``allow_cpu_sync`` (CPU
    platform only); Spyre D2Hs logits before sampling, so that host-device sync
    is free and the full-vocab sort it otherwise runs is pure waste. Top-p
    combined with top-k skips the sort too (``apply_top_k_top_p_sort_free``);
    top-p alone sorts.

    The draw runs in a fused Gumbel-max kernel that hashes its noise per element
    from a per-row seed rather than materializing a ``[B, V]`` noise tensor.
    That draw is why ``forward_native``
    reimplements upstream's tail rather than delegating to ``super()`` (which
    would softmax + ``random_sample``). ``use_fp64_gumbel`` keeps fresh fp64
    noise, drawn in log space: ``argmax(softmax(x)/q) == argmax(x - log q)`` for
    ``q ~ Exp(1)``, which skips the softmax."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # One SpyreGumbelArgmax instance per (rows, vocab) shape.  Each instance
        # compiles and caches exactly one kernel (dynamic=False), so a new shape
        # gets its own kernel.  Populated at warmup (one call per row-width bucket)
        # and reused at inference.
        self._gumbel_kernels: dict[tuple[int, int], SpyreGumbelArgmax] = {}
        # Sampled here because construction runs inside the vLLM config context,
        # while the SpyreGumbelArgmax instances below are created lazily outside it.
        self._compile_enabled: bool = (
            get_cached_compilation_config().mode is not CompilationMode.NONE
        )
        # Persistent ExpSpyreGenerator, created lazily on the first noise request
        # and sized for that request's numel.
        #
        # Possible alternative (not implemented): one ExpSpyreGenerator per numel
        # (the batch size is small, so there would be few), all drawing invocation
        # indices from one shared host-side counter, so their streams never overlap.
        # Before each draw a generator brings its device-side word 2 (X2) up to date
        # with one compiled int16 add of delta = (inv - last_inv) & 0xFFFF, using a
        # cached 0-d delta tensor. Switching batch size then needs no host round trip
        # and no oversized draw. The cost is one compiled pipeline per distinct numel
        # plus a first-use compile of the advance for each new delta value. The
        # slicing approach below was chosen to start with.
        self._noise_generator: ExpSpyreGenerator | None = None
        self._noise_draws: int = 0

    def _gumbel_argmax_kernel(self, shape: tuple[int, int]) -> SpyreGumbelArgmax:
        """Return the cached ``SpyreGumbelArgmax`` for ``shape``, creating it if needed."""
        if shape not in self._gumbel_kernels:
            self._gumbel_kernels[shape] = SpyreGumbelArgmax(self._compile_enabled)
        return self._gumbel_kernels[shape]

    def _make_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """Generate ``[B, Vp]`` Exp(1) noise on ``logits.device``.

        ``logits`` is already padded to a multiple of 64 columns by ``SpyreSampler``
        (so ``Vp == logits.shape[1]``). Rows must start on a stick boundary or the
        flat generator output cannot be viewed as ``[B, V]`` on the device
        (``49159*c0/32`` coordinates).

        Uses a persistent ``ExpSpyreGenerator`` sized for the largest ``numel``
        seen so far.  If a later request requires a larger ``numel``, the
        generator is rebuilt starting from the current draw count.
        """
        rows, vocab = logits.shape
        numel = rows * vocab
        if self._noise_generator is None or numel > self._noise_generator._n:
            self._noise_generator = ExpSpyreGenerator(
                numel,
                rounds=_THREEFRY_ROUNDS,
                device=logits.device,
                start_invocation=self._noise_draws,
            )
        # Wrap-around (known limitation, not fixed): the invocation counter is
        # counter word 2, an int16, so the noise stream repeats after 65,536 draws
        # (see the ExpSpyreGenerator docstring). Possible fixes, both without a
        # device-to-host transfer or a graph break:
        #   (a) when self._noise_draws passes a multiple of 65,536, change one key
        #       word and rebuild the key schedule (small host-to-device copy, in one
        #       place); or
        #   (b) use the currently zero word 3 as the high half of the counter. That
        #       needs a per-generator X3 (not the shared _zero_row tensor) and a
        #       second compiled add when the low half wraps.
        noise = self._noise_generator.generate()[:numel].reshape(rows, vocab)
        self._noise_draws += 1
        return noise

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Mirrors upstream TopKTopPSampler.forward_native (vLLM 0.32.0) with two
        # Spyre changes: sort-free top-k (and top-k + top-p) and a log-space
        # Gumbel draw. Re-sync with upstream on a vLLM bump.
        if k is not None and p is not None:
            logits = apply_top_k_top_p_sort_free(logits, k, p)
        else:
            logits = apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        if not self.use_fp64_gumbel and has_sampling_kernels() and logits.device.type == "cpu":
            # Per-row seeds key the kernel's noise, so the draw is one pass with
            # no noise tensor; seeded requests stay reproducible.
            seeds = torch.randint(0, 2**62, (logits.shape[0],), dtype=torch.long)
            for i, generator in generators.items():
                seeds[i] = torch.randint(0, 2**62, (1,), generator=generator)
            return (
                torch.ops._spyre_C.fused_gumbel_argmax(
                    logits.float(),  # ty: ignore[invalid-argument-type]
                    seeds,  # ty: ignore[invalid-argument-type]
                ),
                logits_to_return,
            )
        # Exp(1) noise, generated like upstream random_sample but pinned to fp32
        # (fp64 under use_fp64_gumbel) independent of the logits dtype, so the
        # log never runs in fp16 (where small q underflows to 0 -> log = -inf).
        noise_dtype = torch.float64 if self.use_fp64_gumbel else torch.float32
        if not self.use_fp64_gumbel and not generators:
            # Threefry path: generate Exp(1) noise on the same device as logits,
            # then dispatch through the shape-keyed compiled kernel.
            #
            # Noise generation runs OUTSIDE the compiled graph: ExpSpyreGenerator
            # advances on device and runs its own compiled graph outside
            # SpyreGumbelArgmax.forward, so there are zero graph breaks.
            #
            # Logits (staggered EA fp32, from to_dtype_d2d in Sampler.forward) and q
            # (standard EA fp32, from i16_pair_to_fp32 in the cipher) differ in
            # element arrangement; SpyreGumbelArgmax moves q into the logits'
            # arrangement itself (see its docstring).
            q = self._make_noise(logits)
            kernel = self._gumbel_argmax_kernel((logits.shape[0], logits.shape[1]))
            return kernel.sample(logits, q), logits_to_return
        q = torch.empty(logits.shape, dtype=noise_dtype, device=logits.device)
        if not generators:
            q.exponential_()
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
        return (logits - q.log_()).argmax(dim=-1).view(-1), logits_to_return
