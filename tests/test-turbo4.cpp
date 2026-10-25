#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

enum class innerq_mode { disabled, calibrate, scaled };

static bool test_backend(ggml_backend_t backend, int d, int nq, bool attention, bool mixed = false, innerq_mode mode = innerq_mode::disabled) {
    ggml_context * ctx = ggml_init({4*1024*1024, nullptr, true});
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 256);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, nq == 1 ? GGML_TYPE_I32 : GGML_TYPE_I64, 256);
    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, d, 256);
    ggml_tensor * scale = mode == innerq_mode::scaled ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, QK_TURBO4) : nullptr;
    ggml_tensor * undo = mode == innerq_mode::scaled ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, QK_TURBO4) : nullptr;
    ggml_tensor * stats = mode == innerq_mode::calibrate ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, QK_TURBO4 + 1) : nullptr;
    ggml_tensor * store = ggml_set_rows_turbo4(ctx, cache, src, ids, scale, stats);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nq, 1, 1);
    ggml_tensor * rotated = ggml_turbo_wht(ctx, q, false, scale);
    ggml_tensor * restored = ggml_turbo_wht(ctx, rotated, true, undo);
    ggml_tensor * output = nullptr;
    if (attention) {
        ggml_tensor * keys = store;
        if (mixed) {
            ggml_tensor * kcache = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, d, 256);
            keys = ggml_set_rows(ctx, kcache, src, ids);
        }
        output = ggml_flash_attn_ext(ctx, mixed ? q : rotated, keys, store, nullptr, 1.0f/std::sqrt(float(d)), 0.0f, 0.0f);
        GGML_ASSERT(ggml_backend_supports_op(backend, output));
        output = ggml_turbo_wht(ctx, output, true, scale);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, store);
    ggml_build_forward_expand(graph, restored);
    if (output) {
        ggml_build_forward_expand(graph, output);
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    std::vector<float> x(d*256), queries(d*nq);
    for (size_t i = d; i < x.size(); ++i) {
        x[i] = std::sin(float(i)*0.137f);
    }
    for (size_t i = 0; i < queries.size(); ++i) {
        queries[i] = std::cos(float(i)*0.091f);
    }
    std::vector<int32_t> ids32(256);
    std::vector<int64_t> ids64(256);
    for (int i = 0; i < 256; ++i) {
        ids32[i] = ids64[i] = 255 - i;
    }
    std::vector<float> inv(QK_TURBO4), scales(QK_TURBO4);
    for (int j = 0; j < QK_TURBO4; ++j) {
        inv[j] = 0.5f + 1.5f*j/(QK_TURBO4 - 1);
        scales[j] = 1.0f/inv[j];
    }
    if (scale) {
        ggml_backend_tensor_set(scale, inv.data(), 0, QK_TURBO4*sizeof(float));
        ggml_backend_tensor_set(undo, scales.data(), 0, QK_TURBO4*sizeof(float));
    }
    if (stats) {
        const float zeros[QK_TURBO4 + 1] = {};
        ggml_backend_tensor_set(stats, zeros, 0, sizeof(zeros));
    }
    ggml_backend_tensor_set(src, x.data(), 0, x.size()*sizeof(float));
    ggml_backend_tensor_set(q, queries.data(), 0, queries.size()*sizeof(float));
    ggml_backend_tensor_set(ids, nq == 1 ? (void *) ids32.data() : (void *) ids64.data(), 0, ggml_nbytes(ids));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> roundtrip(queries.size());
    ggml_backend_tensor_get(restored, roundtrip.data(), 0, roundtrip.size()*sizeof(float));
    float wht_error = 0.0f, quant_error = 0.0f, attn_error = 0.0f;
    for (size_t i = 0; i < queries.size(); ++i) {
        wht_error = std::max(wht_error, std::abs(roundtrip[i] - queries[i]));
    }
    float stats_error = 0;
    if (stats) {
        float actual[QK_TURBO4 + 1];
        float expected[QK_TURBO4 + 1] = {};
        ggml_backend_tensor_get(stats, actual, 0, sizeof(actual));
        for (size_t i = 0; i < x.size(); ++i) expected[i%QK_TURBO4] += x[i]*x[i];
        expected[QK_TURBO4] = x.size()/QK_TURBO4;
        for (int j = 0; j <= QK_TURBO4; ++j) {
            stats_error = std::max(stats_error, std::abs(actual[j] - expected[j])/std::max(1.0f, expected[j]));
        }
    }
    std::vector<block_turbo4_0> packed(d*256/QK_TURBO4), reference(d/QK_TURBO4);
    ggml_backend_tensor_get(cache, packed.data(), 0, packed.size()*sizeof(block_turbo4_0));
    std::vector<float> kv(d*256), cpu(d);
    dequantize_row_turbo4_0(packed.data(), kv.data(), kv.size());
    for (int r = 0; r < 256; ++r) {
        std::vector<float> values(x.begin() + r*d, x.begin() + (r + 1)*d);
        if (scale) {
            for (int j = 0; j < d; ++j) values[j] /= inv[j%QK_TURBO4];
        }
        quantize_row_turbo4_0_ref(values.data(), reference.data(), d);
        dequantize_row_turbo4_0(reference.data(), cpu.data(), d);
        for (int j = 0; j < d; ++j) {
            quant_error = std::max(quant_error, std::abs(cpu[j] - kv[(255 - r)*d + j]));
        }
    }
    if (output) {
        std::vector<float> actual(d*nq), qrot(d), expected(d), unrot(d), scores(256), keys = kv;
        if (mixed) {
            std::vector<block_q8_0> kpacked(d/QK8_0);
            for (int r = 0; r < 256; ++r) {
                quantize_row_q8_0_ref(x.data() + r*d, kpacked.data(), d);
                dequantize_row_q8_0(kpacked.data(), keys.data() + (255 - r)*d, d);
            }
        }
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(float));
        for (int n = 0; n < nq; ++n) {
            for (int j = 0; j < d; j += QK_TURBO4) {
                if (mixed) {
                    std::copy_n(queries.data() + n*d + j, QK_TURBO4, qrot.data() + j);
                } else {
                    float values[QK_TURBO4];
                    for (int l = 0; l < QK_TURBO4; ++l) {
                        values[l] = queries[n*d + j + l]*(scale ? inv[l] : 1.0f);
                    }
                    ggml_turbo_wht_f32(values, qrot.data() + j, false);
                }
            }
            for (int r = 0; r < 256; ++r) {
                float dot = 0.0f;
                for (int j = 0; j < d; ++j) {
                    dot += keys[r*d + j]*qrot[j];
                }
                scores[r] = dot/std::sqrt(float(d));
            }
            const float max_score = *std::max_element(scores.begin(), scores.end());
            float denom = 0.0f;
            for (float & score : scores) {
                score = std::exp(score - max_score);
                denom += score;
            }
            std::fill(expected.begin(), expected.end(), 0.0f);
            for (int r = 0; r < 256; ++r) {
                for (int j = 0; j < d; ++j) {
                    expected[j] += scores[r]/denom*kv[r*d + j];
                }
            }
            for (int j = 0; j < d; j += QK_TURBO4) {
                ggml_turbo_wht_f32(expected.data() + j, unrot.data() + j, true);
                if (scale) {
                    for (int l = 0; l < QK_TURBO4; ++l) unrot[j + l] *= inv[l];
                }
            }
            for (int j = 0; j < d; ++j) {
                attn_error = std::max(attn_error, std::abs(actual[n*d + j] - unrot[j]));
            }
        }
    }
    const bool ok = std::isfinite(wht_error + quant_error + attn_error) &&
                    wht_error < 1e-5f && quant_error < 0.002f && attn_error < 0.002f && stats_error < 2e-5f;
    std::printf("%s K=%s InnerQ=%d D=%d Q=%d WHT=%g quant=%g attention=%g stats=%g %s\n",
                ggml_backend_name(backend), mixed ? "q8_0" : "turbo4_0", (int) mode, d, nq, wht_error, quant_error, attn_error, stats_error, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const bool gpu = argc == 2 && std::strcmp(argv[1], "--gpu") == 0;
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(gpu ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(backend);
    bool ok = true;
    for (int d : {128, 256}) {
        for (int nq : {1, 5}) {
            ok = test_backend(backend, d, nq, gpu) && ok;
            ok = test_backend(backend, d, nq, gpu, false, innerq_mode::calibrate) && ok;
            ok = test_backend(backend, d, nq, gpu, false, innerq_mode::scaled) && ok;
            if (gpu) {
                ok = test_backend(backend, d, nq, true, true) && ok;
            }
        }
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
