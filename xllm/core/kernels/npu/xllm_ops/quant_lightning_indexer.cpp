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
construct_quant_lightning_indexer_output_tensor(const at::Tensor& query,
                                                const at::Tensor& key,
                                                int64_t sparse_count,
                                                std::string query_layout_str,
                                                std::string key_layout_str,
                                                bool return_value) {
  constexpr int64_t SIZE = 8;
  constexpr int64_t DIM_0 = 0;
  constexpr int64_t DIM_1 = 1;
  constexpr int64_t DIM_2 = 2;
  constexpr int64_t DIM_3 = 3;
  at::SmallVector<int64_t, SIZE> output_size;
  for (size_t i = 0; i < query.sizes().size(); i++) {
    TORCH_CHECK(query.size(i) > 0,
                "All values within query's shape should be greater "
                "than 0, but shape[",
                i,
                "] is ",
                query.size(i));
  }
  for (size_t i = 0; i < key.sizes().size(); i++) {
    TORCH_CHECK(key.size(i) > 0,
                "All values within key's shape should be greater "
                "than 0, but shape[",
                i,
                "] is ",
                key.size(i));
  }
  TORCH_CHECK(sparse_count > 0,
              "sparse count should be greater than 0, but now is ",
              sparse_count);
  int64_t key_head_num =
      (key_layout_str == "TND") ? key.size(DIM_1) : key.size(DIM_2);
  if (query_layout_str == "BSND") {
    output_size = {
        query.size(DIM_0), query.size(DIM_1), key_head_num, sparse_count};
  } else {
    output_size = {query.size(DIM_0), key_head_num, sparse_count};
  }
  at::Tensor sparse_indices_out =
      at::zeros(output_size, query.options().dtype(at::kInt));
  at::Tensor sparse_values_out;
  if (return_value) {
    sparse_values_out =
        at::zeros(output_size, query.options().dtype(at::kFloat));
  } else {
    sparse_values_out = at::zeros({0}, query.options().dtype(at::kFloat));
  }

  return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out,
                                            sparse_values_out);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> quant_lightning_indexer(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    int64_t query_quant_mode,
    int64_t key_quant_mode,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_key,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& metadata,
    c10::string_view layout_query,
    c10::string_view layout_key,
    int64_t sparse_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t cmp_ratio,
    bool return_value) {
  std::string query_layout_str = std::string(layout_query);
  std::string key_layout_str = std::string(layout_key);

  // construct the output tensor
  std::tuple<at::Tensor, at::Tensor> quant_lightning_indexer_output =
      construct_quant_lightning_indexer_output_tensor(query,
                                                      key,
                                                      sparse_count,
                                                      query_layout_str,
                                                      key_layout_str,
                                                      return_value);
  at::Tensor sparse_indices_out = std::get<0>(quant_lightning_indexer_output);
  at::Tensor sparse_values_out = std::get<1>(quant_lightning_indexer_output);

  // Convert int8 query/key to fp8 if needed (Ascend950PR kernel only supports
  // fp8). If already fp8, skip. Otherwise: int8 -> fp32 * dequant_scale -> fp8
  // via CPU.
  auto ensure_fp8 = [](const at::Tensor& t,
                       const at::Tensor& dequant_scale) -> at::Tensor {
    fprintf(stderr,
            "[QLI_DEBUG] ensure_fp8 ENTER scalar_type=%d\n",
            static_cast<int>(t.scalar_type()));
    fflush(stderr);
    if (t.scalar_type() == at::kFloat8_e4m3fn) {
      fprintf(stderr, "[QLI_DEBUG] already fp8, skip\n");
      fflush(stderr);
      return t;  // Already fp8, no conversion needed
    }
    // int8 -> fp32 -> dequant (multiply by scale) -> fp8
    auto t_cpu = t.cpu();
    auto t_fp32 = t_cpu.to(at::kFloat);
    // Broadcast dequant_scale to match t's shape for element-wise multiply
    auto scale_cpu = dequant_scale.cpu().to(at::kFloat);
    // scale shape is [B, 1] or [B], broadcast to [B, ..., HD]
    while (scale_cpu.dim() < t_fp32.dim()) {
      scale_cpu = scale_cpu.unsqueeze(-1);
    }
    auto t_dequant = t_fp32 * scale_cpu;
    // Debug: force output via fprintf
    fprintf(stderr, "[QLI_DEBUG] int8 shape=[");
    for (auto s : t.sizes()) fprintf(stderr, "%ld,", s);
    fprintf(stderr, "] scale shape=[");
    for (auto s : dequant_scale.sizes()) fprintf(stderr, "%ld,", s);
    fprintf(stderr,
            "] int8 rng=[%.4f,%.4f] scale rng=[%.6f,%.6f] dequant "
            "rng=[%.4f,%.4f]\n",
            t_cpu.min().item<float>(),
            t_cpu.max().item<float>(),
            dequant_scale.cpu().min().item<float>(),
            dequant_scale.cpu().max().item<float>(),
            t_dequant.min().item<float>(),
            t_dequant.max().item<float>());
    fflush(stderr);
    auto t_fp8 = t_dequant.to(at::kFloat8_e4m3fn);
    return t_fp8.to(t.device());
  };
  auto query_fp8 = ensure_fp8(query, query_dequant_scale);
  auto key_fp8 = ensure_fp8(key, key_dequant_scale);

  // convert str
  char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
  char* key_layout_ptr = const_cast<char*>(key_layout_str.c_str());
  const int64_t state_cache_stride_dim0 = key.stride(0);
  const int64_t scale_stride_dim0 = key_dequant_scale.stride(0);

  EXEC_NPU_CMD(aclnnQuantLightningIndexer,
               query_fp8,
               key_fp8,
               weights,
               query_dequant_scale,
               key_dequant_scale,
               actual_seq_lengths_query,
               actual_seq_lengths_key,
               block_table,
               metadata,
               query_quant_mode,
               key_quant_mode,
               query_layout_ptr,
               key_layout_ptr,
               sparse_count,
               sparse_mode,
               pre_tokens,
               next_tokens,
               cmp_ratio,
               return_value,
               state_cache_stride_dim0,
               scale_stride_dim0,
               sparse_indices_out,
               sparse_values_out);

  return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out,
                                            sparse_values_out);
}

}  // namespace xllm::kernel::npu
