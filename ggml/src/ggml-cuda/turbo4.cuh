#pragma once
#include "common.cuh"

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_set_rows_turbo4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
