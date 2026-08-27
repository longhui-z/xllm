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

auto get_valid_tensor = [](const c10::optional<at::Tensor>& tensor_opt,
                           at::Device device) {
  return tensor_opt.has_value()
             ? tensor_opt
             : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

}  // namespace

at::Tensor quant_lightning_indexer_v2_metadata(
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_k,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_k,
    const c10::optional<at::Tensor>& cmp_residual_k,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    int64_t topk,
    int64_t quant_mode,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    c10::string_view layout_q,
    c10::string_view layout_k,
    int64_t mask_mode,
    int64_t cmp_ratio,
    const c10::string_view device) {
  at::Device output_device = at::Device(std::string(device));
  if (cu_seqlens_q.has_value()) {
    output_device = cu_seqlens_q.value().device();
  } else if (cu_seqlens_k.has_value()) {
    output_device = cu_seqlens_k.value().device();
  } else if (seqused_q.has_value()) {
    output_device = seqused_q.value().device();
  } else if (seqused_k.has_value()) {
    output_device = seqused_k.value().device();
  }

  at::Tensor metadata =
      torch::zeros({kDsaMetadataBufferElements},
                   torch::dtype(torch::kInt32).device(output_device));
  auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
  auto cu_seqlens_k_val = get_valid_tensor(cu_seqlens_k, output_device);
  auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
  auto seqused_k_val = get_valid_tensor(seqused_k, output_device);
  auto cmp_residual_k_val = get_valid_tensor(cmp_residual_k, output_device);

  std::string layout_q_str = std::string(layout_q);
  char* layout_q_ptr = const_cast<char*>(layout_q_str.c_str());
  std::string layout_k_str = std::string(layout_k);
  char* layout_k_ptr = const_cast<char*>(layout_k_str.c_str());

  EXEC_NPU_CMD(aclnnQuantLightningIndexerV2Metadata,
               cu_seqlens_q_val,
               cu_seqlens_k_val,
               seqused_q_val,
               seqused_k_val,
               cmp_residual_k_val,
               num_heads_q,
               num_heads_k,
               head_dim,
               topk,
               quant_mode,
               batch_size,
               max_seqlen_q,
               max_seqlen_k,
               layout_q_ptr,
               layout_k_ptr,
               mask_mode,
               cmp_ratio,
               metadata);

  return metadata;
}

}  // namespace xllm::kernel::npu