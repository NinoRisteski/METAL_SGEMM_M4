#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint m;
    uint n;
    uint k;
};

// v2: 16x16 threadgroup tile, K tiled by 16
kernel void sgemm_tiled16(const device float *A [[buffer(0)]],
                          const device float *B [[buffer(1)]],
                          device float *C [[buffer(2)]],
                          constant MatrixDims &dims [[buffer(3)]],
                          uint2 tg_pos [[threadgroup_position_in_grid]],
                          uint2 tid    [[thread_position_in_threadgroup]],
                          uint2 gid    [[thread_position_in_grid]])
{
    constexpr uint TM = 16;
    constexpr uint TN = 16;
    constexpr uint TK = 16;

    threadgroup float As[TM][TK];
    threadgroup float Bs[TK][TN];

    uint row = gid.y;
    uint col = gid.x;

    float acc = 0.0f;

    // tile origin in global coords (in elements)
    uint row0 = tg_pos.y * TM;
    uint col0 = tg_pos.x * TN;

    uint local_i = tid.y; // 0..15
    uint local_j = tid.x; // 0..15

    uint out_i = row0 + local_i;
    uint out_j = col0 + local_j;

    for (uint k0 = 0; k0 < dims.k; k0 += TK) {
        uint a_i = out_i;
        uint a_k = k0 + local_j; // reuse local_j as k-lane for loading
        uint b_k = k0 + local_i; // reuse local_i as k-lane for loading
        uint b_j = out_j;

        // one A element per thread (covers 16x16 = 256 loads)
        As[local_i][local_j] =
            (a_i < dims.m && a_k < dims.k) ? A[a_i * dims.k + a_k] : 0.0f;

        // one B element per thread (also 256 loads)
        Bs[local_i][local_j] =
            (b_k < dims.k && b_j < dims.n) ? B[b_k * dims.n + b_j] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (out_i < dims.m && out_j < dims.n) {
            #pragma unroll
            for (uint kk = 0; kk < TK; ++kk) {
                acc += As[local_i][kk] * Bs[kk][local_j];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (out_i < dims.m && out_j < dims.n) {
        C[out_i * dims.n + out_j] = acc;
    }
}
