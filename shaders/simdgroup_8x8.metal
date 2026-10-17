#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint m;
    uint n;
    uint k;
};

// First simdgroup SGEMM experiment.
// One 32-lane simdgroup computes one 8x8 C tile. A/B edge tiles are staged
// through threadgroup memory so simdgroup_load never reads out of bounds.
kernel void sgemm_simdgroup_8x8(const device float *A [[buffer(0)]],
                                const device float *B [[buffer(1)]],
                                device float *C [[buffer(2)]],
                                constant MatrixDims &dims [[buffer(3)]],
                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]])
{
    constexpr uint TM = 8;
    constexpr uint TN = 8;
    constexpr uint TK = 8;

    threadgroup float As[TM][TK];
    threadgroup float Bs[TK][TN];
    threadgroup float Cs[TM][TN];

    simdgroup_matrix<float, TM, TN> c_frag =
        make_filled_simdgroup_matrix<float, TM, TN>(0.0f);

    const uint row0 = tg_pos.y * TM;
    const uint col0 = tg_pos.x * TN;

    for (uint k0 = 0; k0 < dims.k; k0 += TK) {
        simdgroup_matrix<float, TM, TK> a_frag;
        simdgroup_matrix<float, TK, TN> b_frag;
        bool full_tile = (row0 + TM <= dims.m) &&
            (col0 + TN <= dims.n) &&
            (k0 + TK <= dims.k);

        if (full_tile) {
            simdgroup_load(a_frag, A + row0 * dims.k + k0, dims.k);
            simdgroup_load(b_frag, B + k0 * dims.n + col0, dims.n);
        } else {
            #pragma unroll
            for (uint i = 0; i < 2; ++i) {
                uint idx = tid * 2 + i;
                uint row = idx / TK;
                uint col = idx % TK;

                uint global_row = row0 + row;
                uint global_k = k0 + col;
                As[row][col] = (global_row < dims.m && global_k < dims.k)
                    ? A[global_row * dims.k + global_k]
                    : 0.0f;

                uint global_n = col0 + col;
                Bs[row][col] = (global_k < dims.k && global_n < dims.n)
                    ? B[global_k * dims.n + global_n]
                    : 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_load(a_frag, (threadgroup const float *)As, TK, ulong2(0, 0), true);
            simdgroup_load(b_frag, (threadgroup const float *)Bs, TN);
        }

        simdgroup_multiply_accumulate(c_frag, a_frag, b_frag, c_frag);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 + TM <= dims.m && col0 + TN <= dims.n) {
        simdgroup_store(c_frag, C + row0 * dims.n + col0, dims.n);
    } else {
        simdgroup_store(c_frag, (threadgroup float *)Cs, TN);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint i = 0; i < 2; ++i) {
            uint idx = tid * 2 + i;
            uint row = idx / TN;
            uint col = idx % TN;
            uint global_row = row0 + row;
            uint global_col = col0 + col;
            if (global_row < dims.m && global_col < dims.n) {
                C[global_row * dims.n + global_col] = Cs[row][col];
            }
        }
    }
}
