#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint m;
    uint n;
    uint k;
};

// v1: Contiguous global access
// Each thread computes one C[row, col]. Mapping keeps gid.x -> col for contiguous B/C.
kernel void sgemm_v1_contig_global(const device float *A [[buffer(0)]],
                                   const device float *B [[buffer(1)]],
                                   device float *C [[buffer(2)]],
                                   constant MatrixDims &dims [[buffer(3)]],
                                   uint2 gid [[thread_position_in_grid]])
{
    const uint row = gid.y;
    const uint col = gid.x;

    if (row >= dims.m || col >= dims.n) return;

    // Precompute row/col base offsets
    const uint a_row_base = row * dims.k;   // start of A row
    const uint c_idx      = row * dims.n + col;

    float acc = 0.0f;

    // For each kk: A is contiguous across kk; B is contiguous across col for fixed kk.
    // This preserves contiguous access patterns for both A (per thread) and B (across threads in x).
    for (uint kk = 0; kk < dims.k; ++kk) {
        const float a = A[a_row_base + kk];
        const float b = B[kk * dims.n + col];
        acc = fma(a, b, acc);
    }

    C[c_idx] = acc;
}
