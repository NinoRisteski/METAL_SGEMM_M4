#include <metal_stdlib>
using namespace metal;

// Naïve single-precision GEMM kernel: each thread computes one C[row, col]
struct MatrixDims {
    uint m;
    uint n;
    uint k;
};

// Buffers 0..2 are A, B, C; buffer 3 carries the matrix dimensions
kernel void sgemm(const device float *A [[buffer(0)]],
                  const device float *B [[buffer(1)]],
                  device float *C [[buffer(2)]],
                  constant MatrixDims &dims [[buffer(3)]],
                  uint2 gid [[thread_position_in_grid]]) {
    // gid gives the row and column this thread is responsible for
    uint row = gid.y;
    uint col = gid.x;
    if (row >= dims.m || col >= dims.n) {
        // Skip threads launched past the matrix bounds
        return;
    }

    // Sum across K dimension for this C[row, col]
    float acc = 0.0f;
    for (uint kk = 0; kk < dims.k; ++kk) {
        float a = A[row * dims.k + kk];      // element from A in current row/kk
        float b = B[kk * dims.n + col];      // element from B in kk/column
        acc += a * b;
    }
    C[row * dims.n + col] = acc;             // write result back to C
}
