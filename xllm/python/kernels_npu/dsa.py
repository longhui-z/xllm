# Copyright 2026 The xLLM Authors.
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

"""NPU DeepSeek-V4 DSA kernels.

These wrap the AscendC operators registered as ``torch.ops.xllm_ops.*`` by
``core/kernels/npu/npu_ops_library.cpp``. They drive the two-stage sparse
attention (original + compressed KV), the KV compressor, the quantized
lightning indexer, and the HyperConnection pre/post used by DeepSeek-V4's DSA
attention path.
"""

from __future__ import annotations

import torch

from xllm.python import dsa_dump


def dequant_swiglu_quant(
    x: torch.Tensor,
    weight_scale: torch.Tensor | None,
    activation_scale: torch.Tensor | None,
    bias: torch.Tensor | None = None,
    quant_scale: torch.Tensor | None = None,
    quant_offset: torch.Tensor | None = None,
    group_index: torch.Tensor | None = None,
    activate_left: bool = True,
    quant_mode: int = 1,
    swiglu_mode: int = 1,
    clamp_limit: float = 0.0,
    glu_alpha: float = 1.0,
    glu_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused dequant + SwiGLU + dynamic quant (replaces manual loop)."""
    return torch.ops.xllm_ops.dequant_swiglu_quant(
        x, weight_scale, activation_scale, bias, quant_scale, quant_offset,
        group_index, activate_left, quant_mode, swiglu_mode,
        clamp_limit, glu_alpha, glu_bias,
    )


def moe_gating_top_k_hash(
    x: torch.Tensor,
    k: int,
    bias: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    tid2eid: torch.Tensor | None,
    k_group: int,
    group_count: int,
    routed_scaling_factor: float,
    eps: float,
    group_select_mode: int,
    renorm: int,
    norm_type: int,
    out_flag: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek-V4 MoE hash routing gate."""
    return torch.ops.xllm_ops.moe_gating_top_k_hash(
        x, k, bias, input_ids, tid2eid, k_group, group_count,
        routed_scaling_factor, eps, group_select_mode, renorm, norm_type, out_flag,
    )


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HyperConnection pre: mix hc_mult streams into one sub-block input.

    Returns ``(attn_input, post, comb)`` where post/comb feed ``hc_post``.
    """
    return torch.ops.xllm_ops.hc_pre(
        x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps
    )


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """HyperConnection post: combine sub-block output with the residual streams."""
    return torch.ops.xllm_ops.hc_post(x, residual, post, comb)


def compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    kv_block_table: torch.Tensor | None,
    score_block_table: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    seqused: torch.Tensor | None,
    start_pos: torch.Tensor | None,
    rope_head_dim: int,
    cmp_ratio: int,
    coff: int,
    norm_eps: float,
    rotary_mode: int,
    enable_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool KV along the token axis by ``cmp_ratio`` (NSA-style compressor).

    ``kv_state`` and ``score_state`` are updated in place.

    Returns ``(cmp_kv, wkv_proj, softmax_res, norm_x, norm_rstd)``; only
    ``cmp_kv`` is consumed by the DSA path.
    """
    # C++ moves DSA metadata to the active device before dispatch. Keep this
    # adapter deterministic; experimental clone/noalias paths do not belong in
    # the public binding.
    kv_block_table = kv_block_table.to(x.device) if kv_block_table is not None else None
    score_block_table = (
        score_block_table.to(x.device) if score_block_table is not None else None
    )
    cu_seqlens = cu_seqlens.to(x.device) if cu_seqlens is not None else None
    seqused = seqused.to(x.device) if seqused is not None else None
    start_pos = start_pos.to(x.device) if start_pos is not None else None
    dsa_dump.snap(
        "compressor",
        {
            "x": x,
            "kv_state": kv_state,
            "score_state": score_state,
            "kv_block_table": kv_block_table,
            "score_block_table": score_block_table,
            "cu_seqlens": cu_seqlens,
            "seqused": seqused,
            "start_pos": start_pos,
        },
        extra={"cmp_ratio": cmp_ratio, "coff": coff, "rope_head_dim": rope_head_dim},
    )
    _cmp_out = torch.ops.xllm_ops.compressor(
        x,
        wkv,
        wgate,
        kv_state,
        score_state,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        kv_block_table,
        score_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        rope_head_dim,
        cmp_ratio,
        coff,
        norm_eps,
        rotary_mode,
        enable_grad,
    )
    dsa_dump.snap(
        "compressor_out",
        {"cmp_kv": _cmp_out[0], "kv_state": kv_state, "score_state": score_state},
    )
    return _cmp_out


def sparse_attn_sharedkv(
    q: torch.Tensor,
    ori_kv: torch.Tensor | None,
    cmp_kv: torch.Tensor | None,
    ori_sparse_indices: torch.Tensor | None,
    cmp_sparse_indices: torch.Tensor | None,
    ori_block_table: torch.Tensor | None,
    cmp_block_table: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_ori_kv: torch.Tensor | None,
    cu_seqlens_cmp_kv: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    seqused_kv: torch.Tensor | None,
    sinks: torch.Tensor | None,
    metadata: torch.Tensor | None,
    softmax_scale: float,
    cmp_ratio: int,
    ori_mask_mode: int,
    cmp_mask_mode: int,
    ori_win_left: int,
    ori_win_right: int,
    layout_q: str,
    layout_kv: str,
    return_softmax_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-stage sparse attention over original and compressed KV."""
    return torch.ops.xllm_ops.sparse_attn_sharedkv(
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_kv,
        sinks,
        metadata,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        return_softmax_lse,
    )


def sparse_attn_sharedkv_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_ori_kv: torch.Tensor | None,
    cu_seqlens_cmp_kv: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    seqused_kv: torch.Tensor | None,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    ori_topk: int,
    cmp_topk: int,
    cmp_ratio: int,
    ori_mask_mode: int,
    cmp_mask_mode: int,
    ori_win_left: int,
    ori_win_right: int,
    layout_q: str,
    layout_kv: str,
    has_ori_kv: bool,
    has_cmp_kv: bool,
) -> torch.Tensor:
    """Build the AICPU tiling metadata for :func:`sparse_attn_sharedkv`."""
    dsa_dump.snap(
        "sparse_attn_sharedkv_metadata",
        {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
            "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
            "seqused_q": seqused_q,
            "seqused_kv": seqused_kv,
        },
        extra={
            "num_heads_q": num_heads_q,
            "num_heads_kv": num_heads_kv,
            "head_dim": head_dim,
            "batch_size": batch_size,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_kv": max_seqlen_kv,
            "ori_topk": ori_topk,
            "cmp_topk": cmp_topk,
            "cmp_ratio": cmp_ratio,
            "ori_mask_mode": ori_mask_mode,
            "cmp_mask_mode": cmp_mask_mode,
            "has_ori_kv": has_ori_kv,
            "has_cmp_kv": has_cmp_kv,
        },
    )
    return torch.ops.xllm_ops.sparse_attn_sharedkv_metadata(
        num_heads_q,
        num_heads_kv,
        head_dim,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_kv,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        ori_topk,
        cmp_topk,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        has_ori_kv,
        has_cmp_kv,
    )


def _cann_sparse_flash_mla():
    import cann_ops_transformer

    return cann_ops_transformer


def sparse_flash_mla(
    q: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CANN built-in two-stage sparse MLA attention (sparse_flash_mla).

    Replacement for ``sparse_attn_sharedkv``. Thin indirection over
    ``cann_ops_transformer.sparse_flash_mla``; the caller passes the same
    logical arguments (q/ori_kv/cmp_kv/block tables/seqlens/sinks/metadata/...)
    with ``layout_kv="PA_BBND"`` and the compressed-side ``seqused_cmp_kv`` /
    ``cmp_residual_kv`` computed by the caller.
    """
    return _cann_sparse_flash_mla().sparse_flash_mla(q, **kwargs)


def sparse_flash_mla_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    **kwargs,
) -> torch.Tensor:
    """CANN built-in metadata builder for :func:`sparse_flash_mla`."""
    return _cann_sparse_flash_mla().sparse_flash_mla_metadata(
        num_heads_q, num_heads_kv, head_dim, **kwargs
    )


def quant_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    query_quant_mode: int,
    key_quant_mode: int,
    actual_seq_lengths_query: torch.Tensor | None,
    actual_seq_lengths_key: torch.Tensor | None,
    block_table: torch.Tensor | None,
    metadata: torch.Tensor | None,
    layout_query: str,
    layout_key: str,
    sparse_count: int,
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    cmp_ratio: int,
    return_value: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the compressed key blocks each query attends to (int8 q/k)."""
    dsa_dump.snap(
        "quant_lightning_indexer",
        {
            "query": query,
            "key": key,
            "weights": weights,
            "query_dequant_scale": query_dequant_scale,
            "key_dequant_scale": key_dequant_scale,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_key": actual_seq_lengths_key,
            "block_table": block_table,
            "metadata": metadata,
        },
        extra={
            "query_quant_mode": query_quant_mode,
            "key_quant_mode": key_quant_mode,
            "sparse_count": sparse_count,
            "sparse_mode": sparse_mode,
            "cmp_ratio": cmp_ratio,
        },
    )
    _out = torch.ops.xllm_ops.quant_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        query_quant_mode,
        key_quant_mode,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        metadata,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        pre_tokens,
        next_tokens,
        cmp_ratio,
        return_value,
    )
    dsa_dump.snap("quant_lightning_indexer_out", {"topk": _out[0], "val": _out[1]})
    return _out


def quant_lightning_indexer_metadata(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    query_quant_mode: int,
    key_quant_mode: int,
    actual_seq_lengths_query: torch.Tensor | None,
    actual_seq_lengths_key: torch.Tensor | None,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    layout_query: str,
    layout_key: str,
    sparse_count: int,
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    cmp_ratio: int,
    device: str,
) -> torch.Tensor:
    """Build the AICPU tiling metadata for :func:`quant_lightning_indexer`."""
    return torch.ops.xllm_ops.quant_lightning_indexer_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        query_quant_mode,
        key_quant_mode,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        pre_tokens,
        next_tokens,
        cmp_ratio,
        device,
    )


def quant_lightning_indexer_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    cmp_residual_k: torch.Tensor | None,
    block_table: torch.Tensor | None,
    output_idx_offset: torch.Tensor | None,
    metadata: torch.Tensor | None,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    topk: int,
    quant_mode: int,
    max_seqlen_q: int,
    layout_q: str,
    layout_k: str,
    mask_mode: int,
    cmp_ratio: int,
    return_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select compressed key blocks (int8 q/k native, no fp8 conversion)."""
    dsa_dump.snap(
        "quant_lightning_indexer_v2",
        {
            "q": q,
            "k": k,
            "w": w,
            "q_descale": q_descale,
            "k_descale": k_descale,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "seqused_q": seqused_q,
            "seqused_k": seqused_k,
            "cmp_residual_k": cmp_residual_k,
            "block_table": block_table,
            "metadata": metadata,
        },
        extra={
            "num_heads_q": num_heads_q,
            "num_heads_k": num_heads_k,
            "head_dim": head_dim,
            "topk": topk,
            "quant_mode": quant_mode,
            "max_seqlen_q": max_seqlen_q,
            "mask_mode": mask_mode,
            "cmp_ratio": cmp_ratio,
        },
    )
    return torch.ops.xllm_ops.quant_lightning_indexer_v2(
        q, k, w, q_descale, k_descale,
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,
        block_table, output_idx_offset, metadata,
        num_heads_q, num_heads_k, head_dim,
        topk, quant_mode, max_seqlen_q,
        layout_q, layout_k,
        mask_mode, cmp_ratio, return_value,
    )


def quant_lightning_indexer_v2_metadata(
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    cmp_residual_k: torch.Tensor | None,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    topk: int,
    quant_mode: int,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    layout_q: str,
    layout_k: str,
    mask_mode: int,
    cmp_ratio: int,
    device: str,
) -> torch.Tensor:
    """Build AICPU tiling metadata for :func:`quant_lightning_indexer_v2`."""
    dsa_dump.snap(
        "quant_lightning_indexer_v2_metadata",
        {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "seqused_q": seqused_q,
            "seqused_k": seqused_k,
            "cmp_residual_k": cmp_residual_k,
        },
        extra={
            "num_heads_q": num_heads_q,
            "num_heads_k": num_heads_k,
            "head_dim": head_dim,
            "topk": topk,
            "quant_mode": quant_mode,
            "batch_size": batch_size,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "mask_mode": mask_mode,
            "cmp_ratio": cmp_ratio,
        },
    )
    return torch.ops.xllm_ops.quant_lightning_indexer_v2_metadata(
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,
        num_heads_q, num_heads_k, head_dim,
        topk, quant_mode, batch_size, max_seqlen_q, max_seqlen_k,
        layout_q, layout_k,
        mask_mode, cmp_ratio, device,
    )
