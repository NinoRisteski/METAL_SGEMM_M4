#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint m;
    uint n;
    uint k;
};

static inline float4 load4_or_zero(const device float *src,
                                   uint base,
                                   bool can_load4,
                                   bool valid0,
                                   bool valid1,
                                   bool valid2,
                                   bool valid3)
{
    if (can_load4) {
        return *((const device float4 *)(src + base));
    }

    return float4(
        valid0 ? src[base] : 0.0f,
        valid1 ? src[base + 1] : 0.0f,
        valid2 ? src[base + 2] : 0.0f,
        valid3 ? src[base + 3] : 0.0f
    );
}

// v5: 2D microtile with vectorized cooperative global loads.
// Same compute shape as v4: BM x BN = 64 x 64, BK=16, per-thread tile=4 x 4.
kernel void sgemm_microtile_2d_vec4(const device float *A [[buffer(0)]],
                                    const device float *B [[buffer(1)]],
                                    device float *C [[buffer(2)]],
                                    constant MatrixDims &dims [[buffer(3)]],
                                    uint2 tg_pos [[threadgroup_position_in_grid]],
                                    uint2 tid    [[thread_position_in_threadgroup]])
{
    constexpr uint BM = 64;
    constexpr uint BN = 64;
    constexpr uint BK = 16;
    constexpr uint TM = 4;
    constexpr uint TN = 4;
    constexpr uint THREADS_X = 16;

    threadgroup float As[BM][BK];
    threadgroup float Bs[BK][BN];

    float acc[TM][TN] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
    };

    const uint row_base = tg_pos.y * BM + tid.y * TM;
    const uint col_base = tg_pos.x * BN + tid.x * TN;
    const uint linear_tid = tid.y * THREADS_X + tid.x;

    for (uint k0 = 0; k0 < dims.k; k0 += BK) {
        // 1024 A tile elements, 256 threads -> one float4 per thread.
        {
            uint idx = linear_tid * 4;
            uint a_row = idx / BK;
            uint a_col = idx % BK;
            uint global_row = tg_pos.y * BM + a_row;
            uint global_col = k0 + a_col;
            uint base = global_row * dims.k + global_col;
            bool row_ok = global_row < dims.m;
            bool aligned_row = (dims.k & 3u) == 0u;
            bool can_load4 = row_ok && aligned_row && (global_col + 3u) < dims.k;

            float4 vals = load4_or_zero(
                A,
                base,
                can_load4,
                row_ok && global_col < dims.k,
                row_ok && (global_col + 1u) < dims.k,
                row_ok && (global_col + 2u) < dims.k,
                row_ok && (global_col + 3u) < dims.k
            );

            As[a_row][a_col] = vals.x;
            As[a_row][a_col + 1u] = vals.y;
            As[a_row][a_col + 2u] = vals.z;
            As[a_row][a_col + 3u] = vals.w;
        }

        // 1024 B tile elements, 256 threads -> one float4 per thread.
        {
            uint idx = linear_tid * 4;
            uint b_row = idx / BN;
            uint b_col = idx % BN;
            uint global_k = k0 + b_row;
            uint global_n = tg_pos.x * BN + b_col;
            uint base = global_k * dims.n + global_n;
            bool row_ok = global_k < dims.k;
            bool aligned_row = (dims.n & 3u) == 0u;
            bool can_load4 = row_ok && aligned_row && (global_n + 3u) < dims.n;

            float4 vals = load4_or_zero(
                B,
                base,
                can_load4,
                row_ok && global_n < dims.n,
                row_ok && (global_n + 1u) < dims.n,
                row_ok && (global_n + 2u) < dims.n,
                row_ok && (global_n + 3u) < dims.n
            );

            Bs[b_row][b_col] = vals.x;
            Bs[b_row][b_col + 1u] = vals.y;
            Bs[b_row][b_col + 2u] = vals.z;
            Bs[b_row][b_col + 3u] = vals.w;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < BK; ++kk) {
            float a_vals[TM];
            float b_vals[TN];

            #pragma unroll
            for (uint mi = 0; mi < TM; ++mi) {
                a_vals[mi] = As[tid.y * TM + mi][kk];
            }

            #pragma unroll
            for (uint ni = 0; ni < TN; ++ni) {
                b_vals[ni] = Bs[kk][tid.x * TN + ni];
            }

            #pragma unroll
            for (uint mi = 0; mi < TM; ++mi) {
                #pragma unroll
                for (uint ni = 0; ni < TN; ++ni) {
                    acc[mi][ni] = fma(a_vals[mi], b_vals[ni], acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint mi = 0; mi < TM; ++mi) {
        uint row = row_base + mi;
        if (row < dims.m) {
            #pragma unroll
            for (uint ni = 0; ni < TN; ++ni) {
                uint col = col_base + ni;
                if (col < dims.n) {
                    C[row * dims.n + col] = acc[mi][ni];
                }
            }
        }
    }
}
