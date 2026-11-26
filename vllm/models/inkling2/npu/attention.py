# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 rel-attention + short convs, NPU branch (correctness first).

The production NVIDIA path runs a fused Triton prologue (q/r clipping, K/V
conv + norm, KV-cache insert, sheared relative-logit precompute) and FA4.
Here the same math runs as torch ops on Ascend (see `ops.py` for the
formulas and their source references).

Correctness-round scope: a single sequence per forward, positions
``0..T-1``. The KV and conv-state caches are plain per-pass torch tensors
(``{pass_idx: tensor}``); hooking them into the vLLM-ascend block manager is
the next round's work, as is a fused Ascend attention kernel (this one
materializes the ``(heads, T, S)`` score matrix).
"""

from __future__ import annotations

import torch
from torch import nn

from .ops import (
    feature_rms,
    gate_sink,
    l2_clip,
    rel_bias,
    short_conv_dynamic,
    short_conv_static,
)


class Inkling2NpuShortConv(nn.Module):
    """`Inkling2ShortConv` (nvidia/sconv.py) as plain torch.

    Static: `weight` `(dim, 1, W)` (checkpoint layout), column `j` pairs with
    lag `W-1-j`. Dynamic (full layers' K/V): per-token tap deltas
    `static_w (W, dim) + weight_proj(x) -> (T, heads, W)`, row `k` pairs with
    lag `k`. Conv state is kept per stream **and per loop pass**.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        *,
        dynamic_input_size: int | None = None,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dynamic = dynamic_input_size is not None
        if self.dynamic:
            self.static_w = nn.Parameter(
                torch.empty(kernel_size, dim, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.weight_proj = nn.Parameter(
                torch.empty(
                    num_heads * kernel_size,
                    dynamic_input_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(dim, 1, kernel_size, dtype=torch.bfloat16),
                requires_grad=False,
            )
        # pass_idx -> (W - 1, dim) of the inputs before the current batch.
        self._past: dict[int, torch.Tensor] = {}

    def reset_state(self) -> None:
        self._past.clear()

    def forward(
        self,
        x: torch.Tensor,
        *,
        x_dynamic: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """`x` bf16 `(T, dim)` -> fp32 conv output `(T, dim)`."""
        past = self._past.get(pass_idx)
        if self.dynamic:
            assert x_dynamic is not None
            dyn = torch.nn.functional.linear(x_dynamic, self.weight_proj)
            dyn = dyn.view(x.shape[0], self.num_heads, self.kernel_size)
            y, past = short_conv_dynamic(
                x, self.static_w, dyn, self.dim // self.num_heads, past
            )
        else:
            y, past = short_conv_static(x, self.weight, past)
        self._past[pass_idx] = past
        return y


class Inkling2NpuAttention(nn.Module):
    """`Inkling2Attention` (nvidia/attention.py) as plain torch."""

    def __init__(self, config, is_local: bool) -> None:
        super().__init__()
        self.is_local = is_local
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.d_rel = config.d_rel
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5
        self.window = config.sliding_window_size if is_local else None
        self.rel_extent = (
            config.sliding_window_size if is_local else config.rel_extent
        )
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps
        self.kernel_size = config.sconv_kernel_size

        hidden = config.hidden_size
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        rel_size = self.num_heads * self.d_rel
        bf16 = torch.bfloat16
        # Checkpoint parameter names are kept (no fused qkvr / TP sharding):
        # wq_du, wr_du, gate_attn, attn_sink_proj, wk_dv, wv_dv, wo_ud.
        self.wq_du = nn.Linear(hidden, q_size, bias=False, dtype=bf16)
        self.wr_du = nn.Linear(hidden, rel_size, bias=False, dtype=bf16)
        self.gate_attn = nn.Linear(hidden, q_size, bias=False, dtype=bf16)
        self.attn_sink_proj = nn.Linear(hidden, self.num_heads, bias=False, dtype=bf16)
        self.wk_dv = nn.Linear(hidden, kv_size, bias=False, dtype=bf16)
        self.wv_dv = nn.Linear(hidden, kv_size, bias=False, dtype=bf16)
        self.wo_ud = nn.Linear(q_size, hidden, bias=False, dtype=bf16)
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent, dtype=bf16),
            requires_grad=False,
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, dtype=bf16),
            requires_grad=False,
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel, dtype=bf16),
            requires_grad=False,
        )
        dynamic_input = None if is_local else hidden
        self.k_sconv = Inkling2NpuShortConv(
            kv_size,
            self.kernel_size,
            dynamic_input_size=dynamic_input,
            num_heads=self.num_kv_heads,
        )
        self.v_sconv = Inkling2NpuShortConv(
            kv_size,
            self.kernel_size,
            dynamic_input_size=dynamic_input,
            num_heads=self.num_kv_heads,
        )
        # pass_idx -> (k, v) of the tokens seen so far.
        self._kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def reset_state(self) -> None:
        self._kv.clear()
        self.k_sconv.reset_state()
        self.v_sconv.reset_state()

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        T = x_q.shape[0]
        H, D, Hkv = self.num_heads, self.head_dim, self.num_kv_heads

        q = self.wq_du(x_q).view(T, H, D)
        r = self.wr_du(x_q).view(T, H, self.d_rel)
        gate = self.gate_attn(x_q).view(T, H, D)
        sinks = self.attn_sink_proj(x_q).view(T, H)
        k = self.wk_dv(x_k).view(T, Hkv, D)
        v = self.wv_dv(x_v).view(T, Hkv, D)

        # Prologue: clip q/r (round then scale by tau), K/V convs + norms.
        q_clip = l2_clip(q, self.q_scale, self.qk_clip, tau=tau).to(torch.bfloat16)
        r_clip = l2_clip(r, self.r_scale, self.rel_clip, tau=tau).to(torch.bfloat16)
        # The kernels round the conv output to the projection dtype before
        # the K clip / V norm (nvidia/ops/sconv.py:795).
        k = self.k_sconv(
            k.reshape(T, Hkv * D),
            x_dynamic=x_k,
            pass_idx=pass_idx,
        ).view(T, Hkv, D).to(torch.bfloat16)
        v = self.v_sconv(
            v.reshape(T, Hkv * D),
            x_dynamic=x_v,
            pass_idx=pass_idx,
        ).view(T, Hkv, D).to(torch.bfloat16)
        k = l2_clip(k, torch.ones_like(k), self.qk_clip).to(torch.bfloat16)
        v = feature_rms(v, self.eps).to(torch.bfloat16)

        # Present K/V of this visit into the pass's cache, attend over all.
        if pass_idx in self._kv:
            k_all = torch.cat((self._kv[pass_idx][0], k), 0)
            v_all = torch.cat((self._kv[pass_idx][1], v), 0)
        else:
            k_all, v_all = k, v
        self._kv[pass_idx] = (k_all, v_all)

        S = k_all.shape[0]
        assert S == T, "correctness round: one full-sequence forward per pass"
        group = H // Hkv
        kk = k_all.repeat_interleave(group, dim=1)  # (S, H, D)
        vv = v_all.repeat_interleave(group, dim=1)

        scores = torch.einsum("thd,shd->hts", q_clip.float(), kk.float())
        scores = scores * self.scaling
        scores = scores + self._rel_scores(r_clip, positions, H, T, S)
        mask = self._mask(positions, T, S)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)  # (H, T)
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hts,shd->thd", probs, vv.float())
        out = gate_sink(out, gate, lse, sinks)
        return self.wo_ud(out.reshape(T, H * D).to(torch.bfloat16))

    def _rel_scores(
        self,
        r_clip: torch.Tensor,
        positions: torch.Tensor,
        H: int,
        T: int,
        S: int,
    ) -> torch.Tensor:
        """Relative logits `(H, T, S)` indexed by key distance."""
        rel = rel_bias(r_clip, self.rel_logits_proj, self.rel_extent)  # (T,H,E)
        dist = positions[:, None] - positions[None, :]  # (T, S)
        idx = dist.clamp(0, self.rel_extent - 1)
        bias = rel.permute(1, 0, 2).gather(
            2, idx.unsqueeze(0).expand(H, -1, -1)
        )
        # Out of range: -inf negative distances; window layers also mask the
        # distances past the extent (nvidia/ops/sconv.py:486 ``shift_rel_proj``),
        # full layers add 0.
        bad = dist < 0
        if self.is_local:
            bad = bad | (dist >= self.rel_extent)
        return bias.masked_fill(bad.unsqueeze(0), float("-inf"))

    def _mask(self, positions: torch.Tensor, T: int, S: int) -> torch.Tensor | None:
        dist = positions[:, None] - positions[None, :]
        bad = dist < 0
        if self.window is not None:
            bad = bad | (dist >= self.window)
        return bad


__all__ = ["Inkling2NpuAttention", "Inkling2NpuShortConv"]
