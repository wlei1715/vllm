# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 MoE, NPU correctness version (Ascend 910B).

Reference math is `tests/models/inkling2/test_moe.py` (authoritative):

* Routing (``_torch_select_experts``):
    logits = x @ gate.weight.T          (bf16 GEMM, fp32 accumulation)
    ids    = topk(sigmoid(logits) + gate.bias, topk, sorted=True)
    w      = softmax(logsigmoid(logits)[ids]) * route_scale
  No softmax over the full expert set — renormalized *sigmoids* of the
  selected experts only.

* Expert FFN (``_moe_reference``):  w13 rows are INTERLEAVED —
    gate_up = x @ w13.T               # [.., 2*inter]
    gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
  i.e. w13 row 2i is gate_i and row 2i+1 is up_i (NOT a contiguous
  first-half/second-half split).  Activation with softcaps:
    act = gate_cap*tanh(gate/gate_cap)*sigmoid(gate)
        * up_cap*tanh(up/up_cap)
  The routing weight is folded in before the down projection:
    out += (act*w) @ w2.T

This module runs the BF16 checkpoint (offline-dequantized); quantized
checkpoints are rejected with a clear error.
"""

from __future__ import annotations

import torch
from torch import nn


class Inkling2NpuGate(nn.Module):
    """Top-k router: renormalized sigmoid probabilities scaled by
    `route_scale`."""

    def __init__(self, config) -> None:
        super().__init__()
        num_experts = config.n_routed_experts
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.use_bias = getattr(config, "use_gate_bias", True)
        self.weight = nn.Parameter(
            torch.empty(num_experts, config.hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                torch.empty(num_experts, dtype=torch.float32), requires_grad=False
            )
            if self.use_bias
            else None
        )

    @staticmethod
    def _load_weight(param: nn.Parameter, loaded: torch.Tensor) -> None:
        assert loaded.shape == param.shape and loaded.dtype == param.dtype
        param.data.copy_(loaded)

    @staticmethod
    def _load_bias(param: nn.Parameter, loaded: torch.Tensor) -> None:
        param.data.copy_(loaded.to(torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """bf16 `(T, H)` -> fp32 routing weights `(T, topk)`, ids `(T, topk)`."""
        logits = torch.nn.functional.linear(x, self.weight).float()
        keys = logits.sigmoid()
        if self.bias is not None:
            keys = keys + self.bias
        # torch.topk returns lower indices first on ties, matching the
        # "breaks ties low" contract of the Triton selection.
        ids = torch.topk(keys, self.topk, dim=-1, sorted=True).indices
        selected = torch.nn.functional.logsigmoid(logits.gather(1, ids))
        weights = selected.softmax(-1) * self.route_scale
        return weights, ids


class Inkling2NpuExperts(nn.Module):
    """BF16 experts: per-expert GEMM loop (correctness-first; the
    vllm-ascend fused_moe path is a later performance step)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.w13_weight = nn.Parameter(
            torch.empty(
                self.num_experts, 2 * config.intermediate_size, config.hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.num_experts, config.hidden_size, config.intermediate_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )

    @staticmethod
    def _load_weight(param: nn.Parameter, loaded: torch.Tensor) -> None:
        assert loaded.shape == param.shape and loaded.dtype == param.dtype, (
            f"{loaded.shape} {loaded.dtype} != {param.shape} {param.dtype}"
        )
        param.data.copy_(loaded)

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor
    ) -> torch.Tensor:
        """bf16 `(T, H)` x, fp32 `(T, topk)` weights, `(T, topk)` ids ->
        bf16 `(T, H)`."""
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        # Loop over experts; per expert a dense GEMM over its token rows.
        for expert in range(self.num_experts):
            token, slot = (ids == expert).nonzero(as_tuple=True)
            if token.numel() == 0:
                continue
            gate_up = torch.nn.functional.linear(
                x[token], self.w13_weight[expert]
            )
            # w13 rows are interleaved: even rows gate, odd rows up.
            gate, up = gate_up[:, 0::2], gate_up[:, 1::2]
            act = (
                self.gate_cap * torch.tanh(gate.float() / self.gate_cap)
                * torch.sigmoid(gate.float())
                * self.up_cap * torch.tanh(up.float() / self.up_cap)
            )
            act = (act * weights[token, slot, None]).to(torch.bfloat16)
            down = torch.nn.functional.linear(act, self.w2_weight[expert])
            out.index_add_(0, token, down.float())
        return out.to(x.dtype)


class Inkling2NpuMoE(nn.Module):
    """`Inkling2MoE` counterpart for the NPU branch: gate + BF16 experts."""

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = Inkling2NpuGate(config)
        self.experts = Inkling2NpuExperts(config)

    def routed_expert_ids(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """bf16 `(T, H)` -> bf16 `(T, H)`."""
        weights, ids = self.gate(x)
        return self.experts(x.contiguous(), weights, ids)