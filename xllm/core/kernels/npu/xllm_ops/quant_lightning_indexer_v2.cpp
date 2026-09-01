/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/library.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

std::tuple<at::Tensor, at::Tensor>
construct_quant_lightning_indexer_v2_output_tensor(const at::Tensor& q,
                                                   const at::Tensor& k,
                                                   int64_t topk,
                                                   std::string layout_q_str,
                                                   std::string layout_k_str,
                                                   bool return_value) {
  constexpr int64_t DIM_0 = 0;
  constexpr int64_t DIM_1 = 1;
  constexpr int64_t DIM_2 = 2;

  for (size_t i = 0; i < q.sizes().size(); i++) {
    TORCH_CHECK(q.size(i) > 0,
                "All values within q's shape should be greater than 0, "
                "but shape[",
                i,
                "] is ",
                q.size(i));
  }
  for (size_t i = 0; i < k.sizes().size(); i++) {
    TORCH_CHECK(k.size(i) > 0,
                "All values within k's shape should be greater than 0, "
                "but shape[",
                i,
                "] is ",
                k.size(i));
  }
  TORCH_CHECK(topk > 0, "topk should be greater than 0, but now is ", topk);

  int64_t k_head_num = (layout_k_str == "TND") ? k.size(DIM_1) : k.size(DIM_2);
  at::SmallVector<int64_t, 8> output_size;
  if (layout_q_str == "BSND") {
    output_size = {q.size(DIM_0), q.size(DIM_1), k_head_num, topk};
  } else {
    output_size = {q.size(DIM_0), k_head_num, topk};
  }
  at::Tensor sparse_indices_out =
      at::zeros(output_size, q.options().dtype(at::kInt));
  at::Tensor sparse_values_out;
  if (return_value) {
    sparse_values_out =
        at::zeros(output_size, q.options().dtype(at::kBFloat16));
  } else {
    sparse_values_out = at::zeros({0}, q.options().dtype(at::kBFloat16));
  }

  return {sparse_indices_out, sparse_values_out};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> quant_lightning_indexer_v2(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& w,
    const at::Tensor& q_descale,
    const at::Tensor& k_descale,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_k,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_k,
    const c10::optional<at::Tensor>& cmp_residual_k,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& output_idx_offset,
    const c10::optional<at::Tensor>& metadata,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    int64_t topk,
    int64_t quant_mode,
    int64_t max_seqlen_q,
    c10::string_view layout_q,
    c10::string_view layout_k,
    int64_t mask_mode,
    int64_t cmp_ratio,
    int64_t return_value) {
  std::string layout_q_str = std::string(layout_q);
  std::string layout_k_str = std::string(layout_k);

  auto qli_output = construct_quant_lightning_indexer_v2_output_tensor(
      q, k, topk, layout_q_str, layout_k_str, static_cast<bool>(return_value));
  at::Tensor sparse_indices_out = std::get<0>(qli_output);
  at::Tensor sparse_values_out = std::get<1>(qli_output);

  // v2 kernel supports int8 natively (quant_mode=2), no fp8 conversion needed.

  char* layout_q_ptr = const_cast<char*>(layout_q_str.c_str());
  char* layout_k_ptr = const_cast<char*>(layout_k_str.c_str());

  EXEC_NPU_CMD(aclnnQuantLightningIndexerV2,
               q,
               k,
               w,
               q_descale,
               k_descale,
               cu_seqlens_q,
               cu_seqlens_k,
               seqused_q,
               seqused_k,
               cmp_residual_k,
               block_table,
               output_idx_offset,
               metadata,
               topk,
               quant_mode,
               max_seqlen_q,
               layout_q_ptr,
               layout_k_ptr,
               mask_mode,
               cmp_ratio,
               return_value,
               sparse_indices_out,
               sparse_values_out);

  return {sparse_indices_out, sparse_values_out};
}

}  // namespace xllm::kernel::npu