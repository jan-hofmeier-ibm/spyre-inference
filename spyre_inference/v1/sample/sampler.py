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

import dataclasses

import torch
import torch.nn.functional as F
from vllm.config import CompilationMode, get_cached_compilation_config
from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.sampler import _SAMPLING_EPS, Sampler

from spyre_inference.custom_ops.lazy_compile import CompileOutermost, maybe_compile
from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels
from spyre_inference.v1.sample.topk_topp_sampler import (
    first_max_index,
    iota_constants,
    padded_vocab,
)

logger = init_logger(__name__)


def _t(t: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    """Move *t* to *device* if it is not already there; None passes through."""
    if t is None or t.device == device:
        return t
    return t.to(device)


def _temp_to_logits_shape(
    temp: torch.Tensor,
    vocab: int,
    target_device: torch.device,
    clamp_greedy: bool = True,
) -> torch.Tensor:
    """Broadcast a CPU ``[B]`` temperature vector to ``[B, vocab]`` on *target_device*.

    Direct ``[B,1]`` → ``[B,V]`` pointwise broadcast fails in the Spyre layout
    planner (mixed stick element-arrangement).  The workaround: materialise a
    contiguous ``[B, vocab]`` tensor on CPU (cheap — it is just ``B`` scalars
    repeated ``vocab`` times), then H2D in one shot.  The resulting tensor has
    the same layout as the logits, so the subsequent ``div_`` sees matching
    sticks and compiles cleanly.

    On non-Spyre devices (CPU tests) the function returns a plain ``[B, 1]``
    tensor that standard broadcasting handles without the extra allocation.

    Args:
        clamp_greedy: when True (default), clamp near-zero temperatures to 1.0
            to avoid division by zero for greedy rows in a mixed batch.
    """
    assert temp.device.type == "cpu"
    B = temp.shape[0]
    if clamp_greedy:
        temp = torch.where(temp < _SAMPLING_EPS, torch.ones_like(temp), temp)
    if target_device.type != "spyre":
        # CPU / non-Spyre: plain unsqueeze, standard broadcast works fine.
        return temp.unsqueeze(1)
    # Spyre path: build the full [B, vocab] tensor on CPU, H2D as fp16, then cast
    # to fp32 on-device.  The on-device fp16→fp32 cast (to_dtype_d2d) produces the
    # same "staggered EA" layout as logits.to(float32), making the subsequent div_
    # a same-layout op that the layout planner can handle.
    tiled_fp16_cpu = temp.unsqueeze(1).expand(B, vocab).contiguous().to(torch.float16)
    temp_fp16 = tiled_fp16_cpu.to(target_device)  # H2D as fp16
    return temp_fp16.to(torch.float32)  # staggered EA fp32


# Cached fp16 ``[B, Vp]`` tiles: 0 for real columns, -inf for pad columns.
_PAD_MASKS: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _pad_vocab(logits: torch.Tensor) -> torch.Tensor:
    """Pad ``[B, V]`` logits on Spyre to ``[B, padded_vocab(V)]`` with ``-inf``.

    vLLM slices the lm_head output to the real vocab (``org_vocab_size``), so the
    row width is generally not a whole number of sticks. A fp32 tensor with a partial
    last stick cannot be padded or sliced reliably on the device (a slice of a
    staggered fp32 tensor is silently wrong), so the pad is added here in fp16,
    before ``Sampler.forward`` casts to fp32. ``-inf`` columns never win a draw.

    ``F.pad``/``cat`` give the result its stick on the batch dim, which no later
    op can reconcile with the vocab-stick temperature tile. Instead copy into a zero
    buffer (stick stays on the vocab dim) and add a cached 0/-inf tile.
    """
    if logits.device.type != "spyre":
        return logits
    rows, vocab = logits.shape
    padded = padded_vocab(vocab)
    if padded == vocab:
        return logits
    key = (rows, vocab, logits.device)
    mask = _PAD_MASKS.get(key)
    if mask is None:
        host = torch.zeros(rows, padded, dtype=torch.float16)
        host[:, vocab:] = float("-inf")
        mask = _PAD_MASKS[key] = host.to(logits.device)
    out = torch.zeros(rows, padded, dtype=logits.dtype, device=logits.device)
    out[:, :vocab] = logits
    return out + mask


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    if not has_sampling_kernels() or logits.device.type != "cpu":
        return Sampler.greedy_sample(logits)
    return torch.ops._spyre_C.greedy_argmax(logits)  # ty: ignore[invalid-argument-type]


# vLLM's final ``torch.where(temperature < eps, greedy, random, out=greedy)`` and the
# casts after it run eagerly. A stick-dimension reduction (the argmax over the vocab)
# yields a *sparse* layout, one valid element per stick, that torch-spyre cannot
# convert to the dense layout of an ordinary ``[B]`` tensor, and an eager op mixing
# the two fails ("no mechanism to resolve stick incompatibility"). So the greedy
# index, the random (Gumbel) index and the temperature compared against ``eps`` are
# all produced by compiled stick-dimension reductions and share the sparse layout.


class SpyreGreedyArgmax(CompileOutermost):
    """Compiled ``argmax(logits)`` as a sparse int64 ``[B]`` (``aten.argmax`` is a
    CPU fallback on Spyre, which also copies the full logits to the host).

    One instance per ``(rows, vocab)`` shape, like ``SpyreGumbelArgmax``.
    """

    allow_inference_recompiles: bool = True

    def __init__(self, compile_enabled: bool) -> None:
        self.spyre_compile_enabled = compile_enabled
        self.spyre_compiled_kernel = None
        super().__init__()

    def sample(self, logits_fp32: torch.Tensor) -> torch.Tensor:
        return self.forward(logits_fp32, iota_constants(logits_fp32.shape[1], logits_fp32.device))

    @maybe_compile
    def forward(self, logits_fp32: torch.Tensor, iota: torch.Tensor) -> torch.Tensor:
        return first_max_index(logits_fp32, iota)


class SpyreSparseTemperature(CompileOutermost):
    """Compiled row-wise ``amax`` of a ``[B, 64]`` temperature tile.

    Every column of a row holds that row's temperature, so the result is the
    temperature as a sparse ``[B]`` tensor (the layout of the sampled indices).
    """

    allow_inference_recompiles: bool = True

    def __init__(self, compile_enabled: bool) -> None:
        self.spyre_compile_enabled = compile_enabled
        self.spyre_compiled_kernel = None
        super().__init__()

    @maybe_compile
    def forward(self, tile: torch.Tensor) -> torch.Tensor:
        return tile.amax(dim=-1)


class SpyreSampler(Sampler):
    """Sampler subclass that fixes cross-device issues when logits are on Spyre.

    The upstream ``Sampler`` assumes all metadata tensors (temperature, top-k,
    penalties, …) are on the same device as logits — on GPU that is implicit
    because everything is on CUDA.  On Spyre, logits are on-device after
    ``_apply_head`` but metadata tensors are always on CPU (built by the
    scheduler).

    Each override below moves only the tensors it touches to ``logits.device``
    before operating, then delegates remaining logic to ``super()``.  That way
    each op in the pipeline runs on Spyre fp32 and we add overrides one at a
    time as new failures are exposed.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Set by the runner while it warms up the sampler: the dummy request sets
        # top-k/top-p so the device kernels get compiled, which a real one would
        # send to the host (see ``_host_reason``).
        self.warming_up: bool = False
        # Sampled here because construction runs inside the vLLM config context,
        # while the kernels below are created lazily outside it.
        self._compile_enabled: bool = (
            get_cached_compilation_config().mode is not CompilationMode.NONE
        )
        self._greedy_kernels: dict[tuple[int, int], SpyreGreedyArgmax] = {}
        self._temperature_kernels: dict[int, SpyreSparseTemperature] = {}
        # The CPU temperature of the batch being sampled, while ``sample`` has
        # swapped a sparse device copy into the metadata (see ``sample``).
        self._host_temperature: torch.Tensor | None = None

    def _host_reason(self, md: SamplingMetadata) -> str | None:
        """Why this step cannot run on the device, or None if it can.

        Each of these features currently has no working device path (observed with
        ``scripts/diag/probe_sampler_features.py``: missing kernels for ``sort``,
        ``masked_fill_``, ``_log_softmax``, a ``topk`` layout error, ...), and the
        failure raises inside the EngineCore and takes the server down. Such a step
        is sampled on the host instead. Remove a check when its feature works on
        the device.
        """
        if md.generators:
            # ``aten::exponential_`` does not exist on Spyre; per-request
            # ``torch.Generator`` draws need the host.
            return "seeded request"
        if self.warming_up:
            return None
        if md.top_k is not None or md.top_p is not None:
            return "top_k/top_p"
        if not md.no_penalties:
            return "penalties"
        if md.allowed_token_ids_mask is not None:
            return "allowed_token_ids"
        if md.bad_words_token_ids:
            return "bad_words"
        if md.max_num_logprobs is not None or md.logprob_token_ids:
            return "logprobs"
        holder = md.thinking_budget_state_holder
        if holder is not None and holder.has_tracked_requests():
            return "thinking budget"
        for proc in md.logitsprocs.all:
            active = {
                "MinPLogitsProcessor": lambda p: p.min_p_count,
                "LogitBiasLogitsProcessor": lambda p: p.biases,
                "MinTokensLogitsProcessor": lambda p: p.min_toks,
            }.get(type(proc).__name__)
            # An unknown (plugin) processor is assumed active.
            if active is None or active(proc):
                return f"logits processor {type(proc).__name__}"
        return None

    def forward(self, logits, sampling_metadata, *args, **kwargs):
        if logits.device.type == "spyre":
            reason = self._host_reason(sampling_metadata)
            if reason is not None:
                # One logits D2H and one small ids H2D; everything else (penalties,
                # logprobs, top-k/top-p, the draw) runs unchanged on the CPU.
                logger.warning_once(
                    "Sampling on the host (logits copied off the Spyre device) for "
                    "requests using: %s. These features have no device path yet.",
                    reason,
                )
                device = logits.device
                out = super().forward(logits.cpu(), sampling_metadata, *args, **kwargs)
                return dataclasses.replace(out, sampled_token_ids=out.sampled_token_ids.to(device))
        # Everything downstream (temperature, penalties, top-k/top-p, the Gumbel
        # draw, logprobs) then sees stick-aligned ``[B, padded_vocab(V)]`` logits.
        return super().forward(_pad_vocab(logits), sampling_metadata, *args, **kwargs)

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        # The runner calls this directly for prompt logprobs, with device logits,
        # outside ``forward``. ``aten::_log_softmax.out`` has no Spyre kernel.
        return Sampler.compute_logprobs(logits.cpu() if logits.device.type == "spyre" else logits)

    def gather_logprobs(self, logprobs, num_logprobs, token_ids):
        # Same reason as ``compute_logprobs``: ``aten::gather.out`` and ``topk`` are
        # not usable on Spyre. The runner copies the result into CPU tensors anyway.
        return super().gather_logprobs(logprobs.cpu(), num_logprobs, token_ids.cpu())

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.device.type != "spyre":
            return greedy_sample(logits)
        shape = (logits.shape[0], logits.shape[1])
        if shape not in self._greedy_kernels:
            self._greedy_kernels[shape] = SpyreGreedyArgmax(self._compile_enabled)
        return self._greedy_kernels[shape].sample(logits)

    def _sparse_temperature(self, temp: torch.Tensor, device: torch.device) -> torch.Tensor:
        """``[B]`` temperature on ``device`` in the sparse layout of the sampled ids.

        The ``[B, 64]`` tile goes up as fp16 and is cast to fp32 on device, the same
        route the logits take, then a compiled row ``amax`` reduces it.
        """
        rows = temp.shape[0]
        if rows not in self._temperature_kernels:
            self._temperature_kernels[rows] = SpyreSparseTemperature(self._compile_enabled)
        tile = temp.unsqueeze(1).expand(rows, 64).contiguous().to(torch.float16)
        return self._temperature_kernels[rows].forward(tile.to(device).to(torch.float32))

    def apply_temperature(  # type: ignore[override]
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        # Expand CPU [B] temp to [B, vocab] on logits.device via the stick-tiling
        # workaround, then div_ with a same-layout tensor (no broadcast needed).
        # _temp_to_logits_shape applies the greedy guard (temp<eps → 1.0) on CPU
        # before the H2D, mirroring what upstream does when not all_random.
        if temp.device.type != "cpu":
            # ``sample`` swapped in the sparse device copy; the CPU one is stashed.
            temp = self._host_temperature if self._host_temperature is not None else temp.cpu()
        temp_2d = _temp_to_logits_shape(
            temp, logits.shape[1], logits.device, clamp_greedy=not all_random
        )
        return logits.div_(temp_2d)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Only a mixed greedy/random batch reaches the upstream ``torch.where(
        # temperature < eps, greedy, random, out=greedy)``. It needs the temperature
        # on the device, in the same sparse layout as the two sampled-index tensors.
        # A copy of the CPU temperature is kept for ``apply_temperature``, which
        # builds its tile on the host.
        temp = sampling_metadata.temperature
        mixed = not (sampling_metadata.all_greedy or sampling_metadata.all_random)
        if temp is not None and mixed and logits.device.type == "spyre":
            self._host_temperature = temp
            sampling_metadata = dataclasses.replace(
                sampling_metadata, temperature=self._sparse_temperature(temp, logits.device)
            )
        try:
            return super().sample(logits, sampling_metadata, logprobs_mode_override)
        finally:
            self._host_temperature = None

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list,
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits
        dev = logits.device
        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            _t(sampling_metadata.prompt_token_ids, dev),
            _t(sampling_metadata.presence_penalties, dev),
            _t(sampling_metadata.frequency_penalties, dev),
            _t(sampling_metadata.repetition_penalties, dev),
            output_token_ids,
        )

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        # ``masked_fill_`` with ``allowed_token_ids_mask`` on CPU fails on Spyre.
        if sampling_metadata.allowed_token_ids_mask is not None:
            mask = sampling_metadata.allowed_token_ids_mask
            if mask.shape[1] < logits.shape[1]:
                # Pad columns (see ``_pad_vocab``) are already -inf: mask them too.
                mask = F.pad(mask, (0, logits.shape[1] - mask.shape[1]), value=True)
            sampling_metadata = dataclasses.replace(
                sampling_metadata,
                allowed_token_ids_mask=_t(mask, logits.device),
            )
        return super().apply_logits_processors(logits, sampling_metadata, predict_bonus_token)
