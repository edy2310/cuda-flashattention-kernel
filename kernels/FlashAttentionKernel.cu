#include <cuda_runtime.h>
#include <math.h>

// This is a simple FlashAttention-style kernel with shared-memory tiling
// and an online softmax update. It uses warp-level primitives to reduce
// dot products efficiently. Each warp computes one output row (one query).
//
// Memory layout (contiguous, row-major):
// Q, K, V, O: [B, H, N, D]
// index = (((b * H + h) * N + n) * D + d)
//
// B: batch size
// H: number of heads
// N: sequence length
// D: head dimension

// Tile sizes for queries (M) and keys/values (N).
// Smaller tiles keep shared memory usage reasonable for larger D.
static constexpr int BLOCK_M = 16;
static constexpr int BLOCK_N = 32;

// Warp-level reduction for a sum across 32 lanes.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void flash_attention_naive(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B,
    int H,
    int N,
    int D
) {
    // Map blockIdx.x to a specific (batch, head) pair.
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;

    // Thread layout:
    // - threadIdx.y selects which query inside the block (one warp per query)
    // - threadIdx.x is the lane inside the warp (used for dot-product reduction)
    const int lane = threadIdx.x;
    const int q_base = blockIdx.y * BLOCK_M;
    const int q_local = threadIdx.y;
    const int q = q_base + q_local;
    const bool valid_q = (q < N);
    if (b >= B) {
        return;
    }

    // Base pointer for this (b, h).
    const int base_bh = ((b * H + h) * N) * D;

    // Shared memory layout:
    // sQ: [BLOCK_M, D] for the query tile
    // sK: [BLOCK_N, D] for the key tile
    // sV: [BLOCK_N, D] for the value tile
    // sO: [BLOCK_M, D] for the output accumulator
    //
    // We pad the stride by +1 to reduce shared-memory bank conflicts.
    extern __shared__ float smem[];
    const int stride = D + 1;
    float* sQ = smem;
    float* sK = sQ + BLOCK_M * stride;
    float* sV = sK + BLOCK_N * stride;
    float* sO = sV + BLOCK_N * stride;

    // Scale factor for dot products.
    const float scale = 1.0f / sqrtf((float)D);

    // Load the Q tile into shared memory (coalesced by linear index).
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;
    for (int idx = tid; idx < BLOCK_M * D; idx += threads) {
        int mq = idx / D;
        int d = idx - mq * D;
        int q_idx = q_base + mq;
        float val = 0.0f;
        if (q_idx < N) {
            val = Q[base_bh + q_idx * D + d];
        }
        sQ[mq * stride + d] = val;
    }

    // Initialize the output accumulator tile to zero.
    for (int idx = tid; idx < BLOCK_M * D; idx += threads) {
        int mq = idx / D;
        int d = idx - mq * D;
        sO[mq * stride + d] = 0.0f;
    }
    __syncthreads();

    // Online softmax state for this query row.
    float m = -INFINITY;
    float l = 0.0f;

    for (int k_base = 0; k_base < N; k_base += BLOCK_N) {
        // Load K and V tiles for this block of keys/values.
        for (int idx = tid; idx < BLOCK_N * D; idx += threads) {
            int mk = idx / D;
            int d = idx - mk * D;
            int k_idx = k_base + mk;
            float k_val = 0.0f;
            float v_val = 0.0f;
            if (k_idx < N) {
                k_val = K[base_bh + k_idx * D + d];
                v_val = V[base_bh + k_idx * D + d];
            }
            sK[mk * stride + d] = k_val;
            sV[mk * stride + d] = v_val;
        }
        __syncthreads();

        if (valid_q) {
            const float* q_row = &sQ[q_local * stride];
            float* o_row = &sO[q_local * stride];

            for (int mk = 0; mk < BLOCK_N; ++mk) {
                int k_idx = k_base + mk;
                if (k_idx >= N) {
                    break;
                }

                // Compute dot product using warp reduction.
                const float* k_row = &sK[mk * stride];
                float partial = 0.0f;
                for (int d = lane; d < D; d += warpSize) {
                    partial += q_row[d] * k_row[d];
                }
                float sum = warp_reduce_sum(partial);
                float score = __shfl_sync(0xffffffff, sum, 0) * scale;

                // Online softmax update:
                // m_new = max(m, score)
                // l_new = l * exp(m - m_new) + exp(score - m_new)
                // o = o * exp(m - m_new) + exp(score - m_new) * v
                float m_new = fmaxf(m, score);
                float alpha = expf(m - m_new);
                float beta = expf(score - m_new);
                l = l * alpha + beta;
                m = m_new;

                const float* v_row = &sV[mk * stride];
                for (int d = lane; d < D; d += warpSize) {
                    o_row[d] = o_row[d] * alpha + beta * v_row[d];
                }
            }
        }
        __syncthreads();
    }

    // Write the final output by normalizing the accumulator.
    if (valid_q) {
        const int o_offset = base_bh + q * D;
        const float inv_l = 1.0f / l;
        const float* o_row = &sO[q_local * stride];
        for (int d = lane; d < D; d += warpSize) {
            O[o_offset + d] = o_row[d] * inv_l;
        }
    }
}

// Simple launcher for the naive kernel.
// One thread computes one query token output.
extern "C" void launch_flash_attention_naive(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B,
    int H,
    int N,
    int D,
    cudaStream_t stream
) {
    // warpSize is a device-only symbol, so use the compile-time constant on host.
    dim3 block(32, BLOCK_M, 1);
    const int blocks_y = (N + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(B * H, blocks_y, 1);
    const size_t shared_bytes =
        (2 * BLOCK_M + 2 * BLOCK_N) * (size_t)(D + 1) * sizeof(float);
    flash_attention_naive<<<grid, block, shared_bytes, stream>>>(Q, K, V, O, B, H, N, D);
}