#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint m;
    uint n;
    uint k;
};

// One 32-lane simdgroup computes one 8x8 C tile.
// Full tiles load directly from device memory.
// Edge tiles are staged through threadgroup memory with zero-padding.
kernel void sgemm_simdgroup_8x8(
    const device float *A [[buffer(0)]],
    const device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant MatrixDims &dims [[buffer(3)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
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

        const bool full_tile =
            (row0 + TM <= dims.m) &&
            (col0 + TN <= dims.n) &&
            (k0 + TK <= dims.k);

        if (full_tile) {
            // A tile: [M, K]
            simdgroup_load(
                a_frag,
                A + row0 * dims.k + k0,
                dims.k
            );

            // B tile: [K, N]
            simdgroup_load(
                b_frag,
                B + k0 * dims.n + col0,
                dims.n
            );
        } else {
            // 8x8 tile = 64 elements.
            // 32 lanes * 2 elements/lane = 64.
            #pragma unroll
            for (uint i = 0; i < 2; ++i) {
                const uint idx = tid * 2 + i;
                const uint row = idx / 8;
                const uint col = idx % 8;

                // -------------------------
                // Stage A tile: [M, K]
                // -------------------------
                const uint global_row = row0 + row;
                const uint global_k_a = k0 + col;

                As[row][col] =
                    (global_row < dims.m &&
                     global_k_a < dims.k)
                        ? A[global_row * dims.k + global_k_a]
                        : 0.0f;

                // -------------------------
                // Stage B tile: [K, N]
                // -------------------------
                const uint global_k_b = k0 + row;
                const uint global_n = col0 + col;

                Bs[row][col] =
                    (global_k_b < dims.k &&
                     global_n < dims.n)
                        ? B[global_k_b * dims.n + global_n]
                        : 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Load staged tiles using the same logical orientation
            // as the direct device-memory path.
            simdgroup_load(
                a_frag,
                (threadgroup const float *)As,
                TK
            );

            simdgroup_load(
                b_frag,
                (threadgroup const float *)Bs,
                TN
            );
        }

        simdgroup_multiply_accumulate(
            c_frag,
            a_frag,
            b_frag,
            c_frag
        );

        // Required before reusing As/Bs on the next edge-tile iteration.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const bool full_output_tile =
        (row0 + TM <= dims.m) &&
        (col0 + TN <= dims.n);

    if (full_output_tile) {
        simdgroup_store(
            c_frag,
            C + row0 * dims.n + col0,
            dims.n
        );
    } else {
        // Store the complete 8x8 fragment to temporary threadgroup memory,
        // then copy only valid edge elements to C.
        simdgroup_store(
            c_frag,
            (threadgroup float *)Cs,
            TN
        );

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint i = 0; i < 2; ++i) {
            const uint idx = tid * 2 + i;
            const uint row = idx / TN;
            const uint col = idx % TN;

            const uint global_row = row0 + row;
            const uint global_col = col0 + col;

            if (global_row < dims.m &&
                global_col < dims.n) {
                C[global_row * dims.n + global_col] =
                    Cs[row][col];
            }
        }
    }
}