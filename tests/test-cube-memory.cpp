#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Reference implementations — pure C++, must match the ggml compute kernels
// ---------------------------------------------------------------------------

static void ref_cleanup(const float * query, const float * codebook,
                        float * out, int d, int m) {
    const int d_codebook = d / 2;
    float   best_sim = -1e30f;
    int     best_idx = 0;
    for (int r = 0; r < m; r++) {
        double sum = 0.0;
        for (int i = 0; i < d; i++) {
            sum += (double)query[i] * (double)codebook[r * d + i];
        }
        float sim = (float)(sum / d_codebook);
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = r;
        }
    }
    memcpy(out, codebook + best_idx * d, d * sizeof(float));
}

static void ref_retrieve(const float * query, const float * slot_keys,
                          const float * slot_values, float * out,
                          int d_key, int n_slots, int d_value, int top_k) {
    // Compute all similarities
    std::vector<float> sims(n_slots);
    for (int s = 0; s < n_slots; s++) {
        double sum = 0.0;
        for (int i = 0; i < d_key; i++) {
            sum += (double)query[i] * (double)slot_keys[s * d_key + i];
        }
        sims[s] = (float)sum;
    }

    // Top-k via insertion sort (matches kernel's bubble-sort approach)
    std::vector<int>   topk_idx(top_k, 0);
    std::vector<float> topk_sim(top_k, -1e30f);

    for (int s = 0; s < n_slots; s++) {
        if (sims[s] > topk_sim[top_k - 1]) {
            topk_sim[top_k - 1] = sims[s];
            topk_idx[top_k - 1] = s;
            // Bubble up (descending)
            for (int j = top_k - 1; j > 0; j--) {
                if (topk_sim[j] > topk_sim[j - 1]) {
                    std::swap(topk_sim[j], topk_sim[j - 1]);
                    std::swap(topk_idx[j], topk_idx[j - 1]);
                } else {
                    break;
                }
            }
        }
    }

    // Softmax over top-k similarities
    float max_sim = topk_sim[0];
    double exp_sum = 0.0;
    for (int t = 0; t < top_k; t++) {
        topk_sim[t] = expf(topk_sim[t] - max_sim);
        exp_sum += topk_sim[t];
    }
    if (exp_sum < 1e-8) {
        exp_sum = 1e-8;
    }
    for (int t = 0; t < top_k; t++) {
        topk_sim[t] = (float)(topk_sim[t] / exp_sum);
    }

    // Weighted gather
    memset(out, 0, d_value * sizeof(float));
    for (int t = 0; t < top_k; t++) {
        const float w = topk_sim[t];
        const float * val = slot_values + topk_idx[t] * d_value;
        for (int v = 0; v < d_value; v++) {
            out[v] += w * val[v];
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic pseudo-random fill (LCG)
// ---------------------------------------------------------------------------

static void fill_deterministic(float * data, int n, uint32_t seed) {
    uint32_t state = seed;
    for (int i = 0; i < n; i++) {
        state = state * 1103515245u + 12345u;
        // Map to [-1, 1]
        data[i] = ((float)(state >> 16) / 32768.0f) - 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Helper: compare two float arrays with tolerance
// ---------------------------------------------------------------------------

static bool arrays_match(const float * a, const float * b, int n, float tol) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > tol) {
            fprintf(stderr, "  mismatch at [%d]: got %f, expected %f (diff %e)\n",
                    i, a[i], b[i], diff);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Helper: run a single-op ggml graph with a given thread count
// ---------------------------------------------------------------------------

static void compute_graph(struct ggml_context * ctx, struct ggml_tensor * result,
                          int n_threads) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool * tp = ggml_threadpool_new(&tpp);
    if (!tp) {
        fprintf(stderr, "FATAL: threadpool_new failed for n_threads=%d\n", n_threads);
        exit(1);
    }

    struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads, tp);
    std::vector<uint8_t> work(cplan.work_size);
    cplan.work_data = work.data();

    ggml_graph_compute(gf, &cplan);

    ggml_threadpool_free(tp);
}

// ---------------------------------------------------------------------------
// Test 1 — Cleanup: known-answer with deterministic data
// ---------------------------------------------------------------------------

static bool test_cleanup_known_answer(void) {
    const int d = 64, m = 32;

    // Reference data
    std::vector<float> q_data(d), cb_data(d * m), ref_out(d);
    fill_deterministic(q_data.data(),  d,       42);
    fill_deterministic(cb_data.data(), d * m,   99);
    ref_cleanup(q_data.data(), cb_data.data(), ref_out.data(), d, m);

    // ggml
    struct ggml_init_params params = {
        /* .mem_size   = */ 64 * 1024 * 1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * query    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
    struct ggml_tensor * codebook = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, m);
    memcpy(query->data,    q_data.data(),  d     * sizeof(float));
    memcpy(codebook->data, cb_data.data(), d * m * sizeof(float));

    struct ggml_tensor * result = ggml_cube_memory_cleanup(ctx, query, codebook);
    compute_graph(ctx, result, 1);

    const float * out = (const float *)result->data;
    bool ok = arrays_match(out, ref_out.data(), d, 1e-5f);

    ggml_free(ctx);
    return ok;
}

// ---------------------------------------------------------------------------
// Test 2 — Cleanup: multi-thread consistency
// ---------------------------------------------------------------------------

static bool test_cleanup_threads(void) {
    const int d = 128, m = 256;

    std::vector<float> q_data(d), cb_data(d * m);
    fill_deterministic(q_data.data(),  d,       7);
    fill_deterministic(cb_data.data(), d * m,   13);

    // Reference: single-thread result
    std::vector<float> ref_out(d);
    ref_cleanup(q_data.data(), cb_data.data(), ref_out.data(), d, m);

    int thread_counts[] = {1, 2, 4};
    for (int tc : thread_counts) {
        struct ggml_init_params params = {
            /* .mem_size   = */ 64 * 1024 * 1024,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ false,
        };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * query    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
        struct ggml_tensor * codebook = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, m);
        memcpy(query->data,    q_data.data(),  d     * sizeof(float));
        memcpy(codebook->data, cb_data.data(), d * m * sizeof(float));

        struct ggml_tensor * result = ggml_cube_memory_cleanup(ctx, query, codebook);
        compute_graph(ctx, result, tc);

        const float * out = (const float *)result->data;
        if (!arrays_match(out, ref_out.data(), d, 1e-5f)) {
            fprintf(stderr, "  FAIL at n_threads=%d\n", tc);
            ggml_free(ctx);
            return false;
        }

        ggml_free(ctx);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 3 — Retrieve: known-answer
// ---------------------------------------------------------------------------

static bool test_retrieve_known_answer(void) {
    const int d_key = 32, n_slots = 128, d_value = 16, top_k = 4;

    std::vector<float> q_data(d_key), sk_data(d_key * n_slots), sv_data(d_value * n_slots);
    fill_deterministic(q_data.data(),  d_key,              101);
    fill_deterministic(sk_data.data(), d_key * n_slots,    202);
    fill_deterministic(sv_data.data(), d_value * n_slots,  303);

    std::vector<float> ref_out(d_value);
    ref_retrieve(q_data.data(), sk_data.data(), sv_data.data(),
                 ref_out.data(), d_key, n_slots, d_value, top_k);

    struct ggml_init_params params = {
        /* .mem_size   = */ 64 * 1024 * 1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * query       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_key);
    struct ggml_tensor * slot_keys   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_key,   n_slots);
    struct ggml_tensor * slot_values = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_value, n_slots);
    memcpy(query->data,       q_data.data(),  d_key              * sizeof(float));
    memcpy(slot_keys->data,   sk_data.data(), d_key * n_slots    * sizeof(float));
    memcpy(slot_values->data, sv_data.data(), d_value * n_slots  * sizeof(float));

    struct ggml_tensor * result = ggml_cube_memory_retrieve(ctx, query, slot_keys,
                                                            slot_values, top_k);
    compute_graph(ctx, result, 1);

    const float * out = (const float *)result->data;
    bool ok = arrays_match(out, ref_out.data(), d_value, 1e-5f);

    ggml_free(ctx);
    return ok;
}

// ---------------------------------------------------------------------------
// Test 4 — Retrieve: top_k edge cases (1 and 8)
// ---------------------------------------------------------------------------

static bool test_retrieve_topk_edges(void) {
    const int d_key = 16, n_slots = 64, d_value = 8;

    std::vector<float> q_data(d_key), sk_data(d_key * n_slots), sv_data(d_value * n_slots);
    fill_deterministic(q_data.data(),  d_key,              500);
    fill_deterministic(sk_data.data(), d_key * n_slots,    600);
    fill_deterministic(sv_data.data(), d_value * n_slots,  700);

    int topk_values[] = {1, 8};
    for (int top_k : topk_values) {
        std::vector<float> ref_out(d_value);
        ref_retrieve(q_data.data(), sk_data.data(), sv_data.data(),
                     ref_out.data(), d_key, n_slots, d_value, top_k);

        struct ggml_init_params params = {
            /* .mem_size   = */ 64 * 1024 * 1024,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ false,
        };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * query       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_key);
        struct ggml_tensor * slot_keys   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_key,   n_slots);
        struct ggml_tensor * slot_values = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_value, n_slots);
        memcpy(query->data,       q_data.data(),  d_key              * sizeof(float));
        memcpy(slot_keys->data,   sk_data.data(), d_key * n_slots    * sizeof(float));
        memcpy(slot_values->data, sv_data.data(), d_value * n_slots  * sizeof(float));

        struct ggml_tensor * result = ggml_cube_memory_retrieve(ctx, query, slot_keys,
                                                                slot_values, top_k);
        compute_graph(ctx, result, 2);

        const float * out = (const float *)result->data;
        if (!arrays_match(out, ref_out.data(), d_value, 1e-5f)) {
            fprintf(stderr, "  FAIL at top_k=%d\n", top_k);
            ggml_free(ctx);
            return false;
        }

        ggml_free(ctx);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 5 — Retrieve: multi-thread consistency
// ---------------------------------------------------------------------------

static bool test_retrieve_threads(void) {
    const int d_key = 32, n_slots = 256, d_value = 16, top_k = 4;

    std::vector<float> q_data(d_key), sk_data(d_key * n_slots), sv_data(d_value * n_slots);
    fill_deterministic(q_data.data(),  d_key,              1001);
    fill_deterministic(sk_data.data(), d_key * n_slots,    2002);
    fill_deterministic(sv_data.data(), d_value * n_slots,  3003);

    std::vector<float> ref_out(d_value);
    ref_retrieve(q_data.data(), sk_data.data(), sv_data.data(),
                 ref_out.data(), d_key, n_slots, d_value, top_k);

    int thread_counts[] = {1, 2, 4};
    for (int tc : thread_counts) {
        struct ggml_init_params params = {
            /* .mem_size   = */ 64 * 1024 * 1024,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ false,
        };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * query       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_key);
        struct ggml_tensor * slot_keys   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_key,   n_slots);
        struct ggml_tensor * slot_values = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_value, n_slots);
        memcpy(query->data,       q_data.data(),  d_key              * sizeof(float));
        memcpy(slot_keys->data,   sk_data.data(), d_key * n_slots    * sizeof(float));
        memcpy(slot_values->data, sv_data.data(), d_value * n_slots  * sizeof(float));

        struct ggml_tensor * result = ggml_cube_memory_retrieve(ctx, query, slot_keys,
                                                                slot_values, top_k);
        compute_graph(ctx, result, tc);

        const float * out = (const float *)result->data;
        if (!arrays_match(out, ref_out.data(), d_value, 1e-5f)) {
            fprintf(stderr, "  FAIL at n_threads=%d\n", tc);
            ggml_free(ctx);
            return false;
        }

        ggml_free(ctx);
    }
    return true;
}

// ---------------------------------------------------------------------------

int main(void) {
    int pass = 0, fail = 0;

    struct { const char * name; bool (*fn)(void); } tests[] = {
        { "cleanup_known_answer",  test_cleanup_known_answer  },
        { "cleanup_threads",       test_cleanup_threads       },
        { "retrieve_known_answer", test_retrieve_known_answer },
        { "retrieve_topk_edges",   test_retrieve_topk_edges   },
        { "retrieve_threads",      test_retrieve_threads      },
    };

    for (auto & t : tests) {
        printf("%-30s ", t.name);
        if (t.fn()) {
            printf("PASS\n");
            pass++;
        } else {
            printf("FAIL\n");
            fail++;
        }
    }

    printf("\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
