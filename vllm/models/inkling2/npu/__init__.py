# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 NPU (Ascend 910B) platform branch.

Correctness-first torch composition: text-only, TP=1, no MTP, no vision.
Expert weights are the offline-dequantized BF16 checkpoint (the Ascend cube
unit has no FP4/FP8 data paths, see A1 of the adaptation notes).  MoE runs
as a per-expert BF16 GEMM loop with the exact routing / SwiGLU math of the
NVIDIA reference (`tests/models/inkling2/test_moe.py::_moe_reference`).
"""

from .loop import Inkling2LoopForConditionalGeneration
from .moe import Inkling2NpuMoE

__all__ = ["Inkling2LoopForConditionalGeneration", "Inkling2NpuMoE"]