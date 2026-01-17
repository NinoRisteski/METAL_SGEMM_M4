#include <metal_stdlib>
using namespace metal;

// Naive SGEMM: each thread computes one C[row, col],
struct MatrixDims {
    uint m;
    uint n;
    uint k;
};

kernel void naive(const device float *A [[buffer(0)]],
                                const device float *B [[buffer(1)]],
                                device float *C [[buffer(2)]],
                                constant MatrixDims &dims [[buffer(3)]],
                                uint2 gid [[thread_position_in_grid]])
{
    const uint row = gid.x;
    const uint col = gid.y;

    if (row >= dims.m || col >= dims.n) return;

    float acc = 0.0f;

    for (uint kk = 0; kk < dims.k; ++kk) {
        const float a = A[row * dims.k + kk];   // A[row, kk]
        const float b = B[kk * dims.n + col];   // B[kk, col]
        acc += a * b;
    }

    C[row * dims.n + col] = acc;                // C[row, col]
}
