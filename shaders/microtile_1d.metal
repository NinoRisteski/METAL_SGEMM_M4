#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint m;
    uint n;
    uint k;
};

// v3: 1D Microtiling - each thread computes 1 row × TN columns
// tile: BM×BN (16×128), Thread tile: 1×TN (1×8)
// 16×16 = 256 threads
kernel void sgemm_microtile_1d(const device float *A [[buffer(0)]],
                               const device float *B [[buffer(1)]],
                               device float *C [[buffer(2)]],
                               constant MatrixDims &dims [[buffer(3)]],
                               uint2 tg_pos [[threadgroup_position_in_grid]],
                               uint2 tid    [[thread_position_in_threadgroup]])
{
    constexpr uint BM = 16;
    constexpr uint BN = 128;
    constexpr uint BK = 16;
    constexpr uint TN = 8;

    threadgroup float As[BM][BK];    // 16×16 = 256 floats = 1KB
    threadgroup float Bs[BK][BN];    // 16×128 = 2048 floats = 8KB

    float acc[TN] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    uint row = tg_pos.y * BM + tid.y;
    uint col_base = tg_pos.x * BN + tid.x * TN;

    uint linear_tid = tid.y * 16 + tid.x;  // 0-255

    for (uint k0 = 0; k0 < dims.k; k0 += BK) {
        // 256 threads, 256 elements → 1 element per thread
        {
            uint a_row = tid.y;
            uint a_col = tid.x;
            uint global_row = tg_pos.y * BM + a_row;
            uint global_col = k0 + a_col;
            As[a_row][a_col] = (global_row < dims.m && global_col < dims.k)
                ? A[global_row * dims.k + global_col]
                : 0.0f;
        }

        // 256 threads, 2048 elements → 8 elements per thread
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint idx = linear_tid * 8 + i;  // 0-2047
            uint b_row = idx / BN;          // which K row (0-15)
            uint b_col = idx % BN;          // which N col (0-127)
            uint global_k = k0 + b_row;
            uint global_n = tg_pos.x * BN + b_col;
            Bs[b_row][b_col] = (global_k < dims.k && global_n < dims.n)
                ? B[global_k * dims.n + global_n]
                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // each thread processes one row of A tile against TN columns of B tile
        #pragma unroll
        for (uint kk = 0; kk < BK; ++kk) {
            float a_val = As[tid.y][kk];  // Load A element once, reuse TN times
            #pragma unroll
            for (uint i = 0; i < TN; ++i) {
                acc[i] = fma(a_val, Bs[kk][tid.x * TN + i], acc[i]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint i = 0; i < TN; ++i) {
        uint col = col_base + i;
        if (row < dims.m && col < dims.n) {
            C[row * dims.n + col] = acc[i];
        }
    }
}
