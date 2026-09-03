# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared base for Python model-executor causal LMs.

A concrete model builds its own graph, lm_head, and config; the wiring that is
identical across models -- the weight-loading entry point and logits
computation -- lives here.

Model forward is driven by the runners owned by Python ModelExecutor;
PyCausalLM no longer calls model.forward() directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn

from scripts.logger import logger

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


class PyModelBase(nn.Module):
    """Base class for causal LMs the C++ ``PyCausalLM`` bridge drives.

    Subclass contract:
      * build ``self.model`` and ``self.lm_head``;
      * implement ``load_weights()``.
    """

    model: nn.Module
    lm_head: nn.Module

    @staticmethod
    def resolve_dtype(dtype: object) -> torch.dtype:
        """Resolve a torch dtype from a ``torch.dtype`` or its string name."""
        if isinstance(dtype, torch.dtype):
            return dtype
        name = dtype if dtype else "bfloat16"
        resolved = getattr(torch, name, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unknown dtype: {dtype!r}")
        return resolved

    def compute_logits(
        self, hidden: torch.Tensor, selected_idxes: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if selected_idxes is not None and selected_idxes.numel() > 0:
            hidden = hidden.index_select(0, selected_idxes)
        logits = self.lm_head(hidden)
        try:
            import os as _os

            if _os.environ.get("DSA_LOGIT_DEBUG") == "1":
                from scripts.logger import logger as _argmax_logger

                top = torch.topk(logits.detach().float(), k=5, dim=-1)
                _argmax_logger.info(
                    "LOGIT_ARGMAX rows=%s hid_std=%.6f sel=%s "
                    "first_row_top5_ids=%s vals=%s",
                    tuple(logits.shape),
                    float(hidden.detach().float().std()),
                    selected_idxes.tolist() if selected_idxes is not None else None,
                    top.indices[0].tolist(),
                    [round(v, 3) for v in top.values[0].tolist()],
                )
        except Exception:
            pass
        try:
            from xllm.python import dsa_dump

            dsa_dump.snap("logits", {"hidden": hidden, "logits": logits})
        except Exception:
            pass
        return logits

    # -- weight loading -------------------------------------------------------
    def load_weights(
        self,
        state_dicts: List["StateDict"],
        tp_rank: int,
        tp_size: int,
    ) -> None:
        """Load checkpoint weights into this model's parameters.

        Called by the C++ bridge (``PyCausalLM::load_model``).
        The model owns ALL weight transform logic: TP slicing, fused-weight
        concatenation, format conversion, etc.
        """
        raise NotImplementedError
