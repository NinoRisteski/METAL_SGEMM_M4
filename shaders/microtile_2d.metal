#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint m;
    uint n;
    uint k;
};

// v4: 2D Microtiling - each thread computes a TM x TN output block.
// Tile: BM x BN (64 x 64), K tile: BK=16, thread tile: 4 x 4.
// 16 x 16 = 256 threads, each producing 16 output elements.
kernel void sgemm_microtile_2d(const device float *A [[buffer(0)]],
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
        // 1024 A tile elements, 256 threads -> 4 loads per thread.
        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            uint idx = linear_tid * 4 + i;
            uint a_row = idx / BK;
            uint a_col = idx % BK;
            uint global_row = tg_pos.y * BM + a_row;
            uint global_col = k0 + a_col;
            As[a_row][a_col] = (global_row < dims.m && global_col < dims.k)
                ? A[global_row * dims.k + global_col]
                : 0.0f;
        }

        // 1024 B tile elements, 256 threads -> 4 loads per thread.
        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            uint idx = linear_tid * 4 + i;
            uint b_row = idx / BN;
            uint b_col = idx % BN;
            uint global_k = k0 + b_row;
            uint global_n = tg_pos.x * BN + b_col;
            Bs[b_row][b_col] = (global_k < dims.k && global_n < dims.n)
                ? B[global_k * dims.n + global_n]
                : 0.0f;
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
