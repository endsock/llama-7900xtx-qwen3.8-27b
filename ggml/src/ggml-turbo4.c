#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-turbo4.h"

#include <math.h>
#include <string.h>

void ggml_turbo_wht_f32(const float * x, float * y, bool inverse) {
    float tmp[QK_TURBO4];
    for (int i = 0; i < QK_TURBO4; ++i) {
        tmp[i] = x[i] * (inverse ? turbo4_signs2[i] : turbo4_signs1[i]);
    }
    for (int h = 1; h < QK_TURBO4; h *= 2) {
        for (int i = 0; i < QK_TURBO4; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = tmp[j], b = tmp[j + h];
                tmp[j] = a + b;
                tmp[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < QK_TURBO4; ++i) {
        y[i] = tmp[i] * 0.08838834764831845f * (inverse ? turbo4_signs1[i] : turbo4_signs2[i]);
    }
}

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k % QK_TURBO4 == 0);
    for (int64_t b = 0; b < k / QK_TURBO4; ++b) {
        const float * src = x + b*QK_TURBO4;
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4; ++j) {
            norm_sq += src[j]*src[j];
        }
        const float norm = sqrtf(norm_sq);
        const float inv = norm > 1e-10f ? 1.0f/norm : 0.0f;
        float rotated[QK_TURBO4];
        for (int j = 0; j < QK_TURBO4; ++j) {
            rotated[j] = src[j]*inv;
        }
        ggml_turbo_wht_f32(rotated, rotated, false);
        memset(y[b].qs, 0, sizeof(y[b].qs));
        float recon_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4; ++j) {
            int idx = 0;
            while (idx < 15 && rotated[j] >= turbo4_midpoints[idx]) {
                ++idx;
            }
            y[b].qs[j/2] |= (uint8_t) (idx << (4*(j%2)));
            recon_sq += turbo4_centroids[idx]*turbo4_centroids[idx];
        }
        y[b].norm = GGML_FP32_TO_FP16(norm/sqrtf(recon_sq));
        y[b].rnorm = GGML_FP32_TO_FP16(0.0f);
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k % QK_TURBO4 == 0);
    for (int64_t i = 0; i < k; ++i) {
        const block_turbo4_0 * b = x + i/QK_TURBO4;
        const int j = i%QK_TURBO4;
        y[i] = GGML_FP16_TO_FP32(b->norm) * turbo4_centroids[(b->qs[j/2] >> (4*(j%2))) & 15];
    }
}
