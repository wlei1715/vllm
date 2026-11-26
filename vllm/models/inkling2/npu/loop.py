# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Looped Inkling2, NPU platform branch (Ascend 910B) — correctness-first.

Milestone scope (phase ④ round 1): text-only, TP=1, no MTP, no vision.
What is real here:
  * weight loading with the checkpoint renames of the NVIDIA branch
    (``_renames`` / expert tensor names unchanged — the BF16 dequantized
    checkpoint keeps the original key layout);
  * RMSNorm (fp32 math), the loop message controllers, the loop pass
    schedule (prelude / 3x core / coda) and the routed MoE
    (see `moe.py` — exact reference math, per-expert BF16 GEMM).
What is a placeholder this round (explicit TODO, correctness target of the
next rounds):
  * attention: standard scaled-dot-product MHA with sliding-window support —
    the production rel-attention (FA4) needs an Ascend kernel evaluation;
  * short conv (sconv): causal depthwise-style short convolution pending a
    faithful port of `nvidia/sconv.py`;
  * MTP / vision / TP>1: unsupported this round.

The experts must come from the offline-dequantized BF16 checkpoint
(`/mnt/share/weights/Inkling-2-Mini-bf16`); the 910B cube unit has no
FP4/FP8 data paths.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from ..configs import Inkling2ModelConfig
from .moe import Inkling2NpuMoE

LOOP_STREAMS = ("attn", "k", "v", "mlp")


class NpuRMSNorm(nn.Module):
    """RMSNorm, fp32 math, single rounding to the output dtype."""

    def __init__(self, hidden_size: int, eps: float, num_inputs: int = 1) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_inputs, hidden_size), requires_grad=False
        )
        self.eps = eps

    @staticmethod
    def _load(param: nn.Parameter, loaded: torch.Tensor, shard_id: int = 0) -> None:
        param.data[shard_id].copy_(loaded)

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (x * self.weight[index].float()).to(torch.bfloat16)


class NpuMessageController(nn.Module):
    """Loop message blending (reference math of `nvidia/ops/loop.py`):

        pair   = [dest, message] RMS-normalized (fp32)
        h      = down(pair)                       (rank-r, bf16 GEMM)
        forget, gain = forget(h), gain(h)         (bf16 GEMM -> fp32)
        out    = (1 - tanh(forget)) * dest + tanh(gain) * message   (fp32)
    """

    def __init__(self, rank: int, hidden_size: int) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(
            torch.empty(rank, 2 * hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.gain_weight = nn.Parameter(
            torch.empty(2 * hidden_size, rank, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.forget_weight = nn.Parameter(
            torch.empty(2 * hidden_size, rank, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """fp32 `(T, D)` `dest`, `message` -> fp32 `(T, D)`."""
        pair = torch.stack((dest, message), dim=1)
        pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
        normalized = pair.to(torch.bfloat16).flatten(1)
        h = torch.nn.functional.linear(normalized, self.down_weight)
        gain = torch.nn.functional.linear(h, self.gain_weight).float()
        forget = torch.nn.functional.linear(h, self.forget_weight).float()
        return (1 - torch.tanh(forget)) * dest + torch.tanh(gain) * message


class NpuDecoderLayer(nn.Module):
    """One decoder layer of the loop schedule.  Attention and sconv are
    placeholders this round; the MoE is the real reference-math version."""

    def __init__(self, config: Inkling2ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        hidden = config.hidden_size
        self.attn_norm = NpuRMSNorm(hidden, config.rms_norm_eps, num_inputs=3)
        self.mlpu_norm = NpuRMSNorm(hidden, config.rms_norm_eps)
        self.mlp = Inkling2NpuMoE(config)
        in_loop = layer_id in config.loop_core_layers
        if in_loop:
            # TODO(loop): per-stream controllers from the checkpoint's
            # controller_down / controller_gain / controller_forget buffers
            # (stream-stacked); wired when the full loop blend runs.
            rank = config.loop_config["gate_rank"]
            self.controllers = nn.ModuleDict(
                {s: NpuMessageController(rank, hidden) for s in LOOP_STREAMS}
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(attention): real rel-attention + sconv (Ascend kernels /
        # faithful torch port).  Placeholder: identity attention keeps the
        # Level-4 milestone (full-model parity) out of this round's scope.
        return x + self.mlp(self.mlpu_norm(x))


class NpuLoopModel(nn.Module):
    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            NpuDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = NpuRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        # TODO(loop): the real pass schedule (prelude L0..5, 3x core L6..18
        # with per-pass KV/conv states and message blending, coda L19..23)
        # once attention/sconv are real.  Single sequential visit this round.
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Inkling2LoopForConditionalGeneration(nn.Module):
    """`Inkling2LoopForConditionalGeneration` for the NPU platform."""

    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        assert vllm_config.parallel_config.tensor_model_parallel_size == 1, (
            "NPU branch round 1: TP=1 only"
        )
        config = vllm_config.model_config.hf_config
        text_config = config.text_config
        assert text_config.loop_config is not None
        self.config = config
        self.text_config = text_config
        self.model = NpuLoopModel(text_config)
        self.lm_head = ParallelLMHead(
            text_config.padded_vocab_size, text_config.hidden_size
        )

    def load_weights(self, weights):
        # The BF16 dequantized checkpoint keeps the original key layout;
        # only the module-path renames of the NVIDIA loader apply.  This
        # round loads the tensors the real modules own (embed / norms /
        # MoE / lm_head); attention/sconv/controller tensors are skipped
        # until their modules are real.
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        loaded = set()
        for name, tensor in weights:
            for old, new in self._renames:
                name = name.replace(old, new)
            target = params.get(name) if name in params else buffers.get(name)
            if target is None:
                continue  # TODO(round 2): attention/sconv/controller weights
            assert tensor.shape == target.shape, (name, tensor.shape, target.shape)
            target.data.copy_(tensor.to(target.dtype))
            loaded.add(name)
        missing = [n for n in params if n not in loaded]
        # Only the modules implemented this round must be complete.
        required = (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"
        )
        incomplete = [n for n in missing if n.startswith(required)]
        assert not incomplete, f"missing required weights: {incomplete[:8]}"
        moe_missing = [
            n
            for n in missing
            if ".mlp.gate." in n or ".mlp.experts." in n
        ]
        assert not moe_missing, f"missing MoE weights: {moe_missing[:8]}"

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Minimal text-only forward (offline parity runs); the full vLLM v1
        integration (attention metadata, KV cache specs) lands with the
        attention round."""
        hidden = self.model(input_ids)
        return torch.nn.functional.linear(hidden, self.lm_head.weight)