#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint m;
    uint n;
    uint k;
};

kernel void sgemm(const device float *A [[buffer(0)]],
                  const device float *B [[buffer(1)]],
                  device float *C [[buffer(2)]],
                  constant MatrixDims &dims [[buffer(3)]],
                  uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= dims.m || col >= dims.n) {
        return;
    }

    float acc = 0.0f;
    for (uint kk = 0; kk < dims.k; ++kk) {
        float a = A[row * dims.k + kk];
        float b = B[kk * dims.n + col];
        acc += a * b;
    }
    C[row * dims.n + col] = acc;
}
