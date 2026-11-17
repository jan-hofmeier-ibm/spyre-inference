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
import torch.distributed as dist
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from spyre_inference import envs

from .lazy_compile import CompileOutermost, maybe_compile
from .utils import convert
from .vocab_parallel_embedding import promote_tied_lm_head


@LogitsProcessor.register_oot(name="LogitsProcessor")
class SpyreLogitsProcessor(CompileOutermost, LogitsProcessor):
    def _apply_head(self, lm_head, hidden_states, embedding_bias=None):
        """Project through the lm_head; logits stay on Spyre for TP=1.

        The head's ``SpyreUnquantizedLMHeadMethod.apply`` returns fp16 logits on Spyre.
        For TP>1 ``_gather_logits`` handles the all_gather and D2H. For TP=1 logits
        stay on Spyre: ``logits.to(torch.float32)`` in ``Sampler.forward`` now works
        via ``to_dtype_d2d`` (torch-spyre threefry-enablement branch), so the old
        workaround D2H is no longer needed.
        """
        # A model may hand us its embedding table as the head; give it a padded `Wᵀ`
        # on the first call, where the head's identity is finally unambiguous.
        promote_tied_lm_head(lm_head)

        return super()._apply_head(lm_head, hidden_states, embedding_bias)

    @maybe_compile
    def _all_gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_gather(logits)

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # `Platform.use_all_gather()` is constant-True on Spyre, so the gather is always the
        # all-gather form; compiled it lowers to `spyre::all_gather_async`, then D2H for sampling.
        if envs.SPYRE_LOGITS_CPU_GATHER:
            return self._cpu_all_gather_logits(logits)
        return convert(self._all_gather_logits(logits), device="cpu")

    @staticmethod
    def _cpu_all_gather_logits(logits: torch.Tensor) -> torch.Tensor:
        # The device-to-host copy converts every fp16 element on the host, so copying one
        # vocab shard per rank is cheaper than a full-vocab copy on every rank.
        tp = get_tp_group()
        shard = convert(logits, device="cpu").contiguous()
        shards = [torch.empty_like(shard) for _ in range(tp.world_size)]
        dist.all_gather(shards, shard, group=tp.cpu_group)  # ty: ignore[possibly-missing-attribute]
        return torch.cat(shards, dim=-1)
