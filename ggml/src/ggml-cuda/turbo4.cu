#include "turbo4.cuh"
#include "../ggml-turbo4.h"

static __device__ void turbo4_wht_shared(float * x, bool inverse) {
    const int j = threadIdx.x;
    x[j] *= inverse ? turbo4_signs2[j] : turbo4_signs1[j];
    __syncthreads();
    for (int h = 1; h < QK_TURBO4; h *= 2) {
        if (j%(2*h) < h) {
            const float a = x[j], b = x[j + h];
            x[j] = a + b;
            x[j + h] = a - b;
        }
        __syncthreads();
    }
    x[j] *= 0.08838834764831845f * (inverse ? turbo4_signs1[j] : turbo4_signs2[j]);
    __syncthreads();
}

static __global__ void k_turbo4_wht(const float * src, float * dst, bool inverse) {
    __shared__ float x[QK_TURBO4];
    const int64_t i = (int64_t) blockIdx.x*QK_TURBO4 + threadIdx.x;
    x[threadIdx.x] = src[i];
    __syncthreads();
    turbo4_wht_shared(x, inverse);
    dst[i] = x[threadIdx.x];
}

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]) && dst->ne[0]%QK_TURBO4 == 0);
    const bool inverse = ggml_get_op_params_i32(dst, 0) != 0;
    k_turbo4_wht<<<ggml_nelements(dst)/QK_TURBO4, QK_TURBO4, 0, ctx.stream()>>>(
        (const float *) dst->src[0]->data, (float *) dst->data, inverse);
}

template <typename idx_t>
static __global__ void k_set_rows_turbo4(const float * src, const char * indices, char * dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne11, int64_t ne12,
        size_t nb01, size_t nb02, size_t nb03, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3) {
    const int j = threadIdx.x;
    const int64_t n_blocks = ne00/QK_TURBO4;
    const int64_t ib = blockIdx.x%n_blocks;
    int64_t tmp = blockIdx.x/n_blocks;
    const int64_t i01 = tmp%ne01;
    tmp /= ne01;
    const int64_t i02 = tmp%ne02;
    const int64_t i03 = tmp/ne02;
    const idx_t row = *(const idx_t *) (indices + i01*nb10 + (i02%ne11)*nb11 + (i03%ne12)*nb12);
    const float * src_row = (const float *) ((const char *) src + i01*nb01 + i02*nb02 + i03*nb03);
    block_turbo4_0 * block = (block_turbo4_0 *) (dst + row*nb1 + i02*nb2 + i03*nb3) + ib;

    __shared__ float x[QK_TURBO4];
    __shared__ float sums[QK_TURBO4];
    __shared__ uint8_t indices_q[QK_TURBO4];
    __shared__ float norm;
    x[j] = src_row[ib*QK_TURBO4 + j];
    sums[j] = x[j]*x[j];
    __syncthreads();
    for (int h = QK_TURBO4/2; h > 0; h /= 2) {
        if (j < h) {
            sums[j] += sums[j + h];
        }
        __syncthreads();
    }
    if (j == 0) {
        norm = sqrtf(sums[0]);
    }
    __syncthreads();
    x[j] *= norm > 1e-10f ? 1.0f/norm : 0.0f;
    __syncthreads();
    turbo4_wht_shared(x, false);
    int idx = 0;
    while (idx < 15 && x[j] >= turbo4_midpoints[idx]) {
        ++idx;
    }
    indices_q[j] = idx;
    sums[j] = turbo4_centroids[idx]*turbo4_centroids[idx];
    __syncthreads();
    if (j%2 == 0) {
        block->qs[j/2] = indices_q[j] | (indices_q[j + 1] << 4);
    }
    for (int h = QK_TURBO4/2; h > 0; h /= 2) {
        if (j < h) {
            sums[j] += sums[j + h];
        }
        __syncthreads();
    }
    if (j == 0) {
        block->norm = __float2half(norm/sqrtf(sums[0]));
        block->rnorm = __float2half(0.0f);
    }
}

void ggml_cuda_set_rows_turbo4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * idx = dst->src[1];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && src->nb[0] == sizeof(float));
    GGML_ASSERT(src->ne[0]%QK_TURBO4 == 0);
    const int64_t blocks = ggml_nelements(src)/QK_TURBO4;
#define LAUNCH_TURBO4(idx_t) \
    k_set_rows_turbo4<idx_t><<<blocks, QK_TURBO4, 0, ctx.stream()>>>( \
        (const float *) src->data, (const char *) idx->data, (char *) dst->data, \
        src->ne[0], src->ne[1], src->ne[2], idx->ne[1], idx->ne[2], \
        src->nb[1], src->nb[2], src->nb[3], idx->nb[0], idx->nb[1], idx->nb[2], \
        dst->nb[1], dst->nb[2], dst->nb[3])
    if (idx->type == GGML_TYPE_I64) {
        LAUNCH_TURBO4(int64_t);
    } else {
        GGML_ASSERT(idx->type == GGML_TYPE_I32);
        LAUNCH_TURBO4(int32_t);
    }
#undef LAUNCH_TURBO4
}
