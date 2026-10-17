#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint m;
    uint n;
    uint k;
};

// 4 SIMD groups cooperate on one 16x16 output tile.
//
// Layout:
//   SG0 -> C[ 0:8,  0:8]
//   SG1 -> C[ 0:8,  8:16]
//   SG2 -> C[ 8:16, 0:8]
//   SG3 -> C[ 8:16, 8:16]
//
// Per K-step:
//   A tile = 16x8
//   B tile = 8x16
//
// The 4 SIMD groups reuse the same staged A/B tiles.
kernel void sgemm_simdgroup_16x16(
    const device float *A [[buffer(0)]],
    const device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant MatrixDims &dims [[buffer(3)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint BM = 16;
    constexpr uint BN = 16;
    constexpr uint BK = 8;

    constexpr uint FM = 8;
    constexpr uint FN = 8;
    constexpr uint FK = 8;

    constexpr uint SIMD_WIDTH = 32;
    constexpr uint NUM_SIMDGROUPS = 4;
    constexpr uint THREADS = SIMD_WIDTH * NUM_SIMDGROUPS; // 128

    threadgroup float As[BM][BK];  // 16 x 8 = 128 floats
    threadgroup float Bs[BK][BN];  // 8 x 16 = 128 floats

    // Used only for safe partial output tiles.
    threadgroup float Cs[NUM_SIMDGROUPS][FM][FN];

    const uint simdgroup_id = tid / SIMD_WIDTH;

    // 2x2 arrangement of SIMD groups.
    const uint sg_row = simdgroup_id / 2;
    const uint sg_col = simdgroup_id % 2;

    const uint tile_row0 = tg_pos.y * BM;
    const uint tile_col0 = tg_pos.x * BN;

    const uint frag_row = sg_row * FM;
    const uint frag_col = sg_col * FN;

    simdgroup_matrix<float, FM, FN> c_frag =
        make_filled_simdgroup_matrix<float, FM, FN>(0.0f);

    for (uint k0 = 0; k0 < dims.k; k0 += BK) {

        // -------------------------------------------------
        // Cooperative load of A: 16 x 8 = 128 elements
        // One element per thread.
        // -------------------------------------------------

        {
            const uint idx = tid;

            const uint local_row = idx / BK;
            const uint local_k   = idx % BK;

            const uint global_row = tile_row0 + local_row;
            const uint global_k   = k0 + local_k;

            As[local_row][local_k] =
                (global_row < dims.m && global_k < dims.k)
                    ? A[global_row * dims.k + global_k]
                    : 0.0f;
        }

        // -------------------------------------------------
        // Cooperative load of B: 8 x 16 = 128 elements
        // One element per thread.
        // -------------------------------------------------

        {
            const uint idx = tid;

            const uint local_k   = idx / BN;
            const uint local_col = idx % BN;

            const uint global_k   = k0 + local_k;
            const uint global_col = tile_col0 + local_col;

            Bs[local_k][local_col] =
                (global_k < dims.k && global_col < dims.n)
                    ? B[global_k * dims.n + global_col]
                    : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -------------------------------------------------
        // Each SIMD group selects its 8x8 A and B fragments.
        //
        // SG0 / SG1 share upper A fragment.
        // SG2 / SG3 share lower A fragment.
        //
        // SG0 / SG2 share left B fragment.
        // SG1 / SG3 share right B fragment.
        // -------------------------------------------------

        simdgroup_matrix<float, FM, FK> a_frag;
        simdgroup_matrix<float, FK, FN> b_frag;

        const threadgroup float *a_ptr =
            &As[frag_row][0];

        const threadgroup float *b_ptr =
            &Bs[0][frag_col];

        simdgroup_load(
            a_frag,
            a_ptr,
            BK
        );

        simdgroup_load(
            b_frag,
            b_ptr,
            BN
        );

        simdgroup_multiply_accumulate(
            c_frag,
            a_frag,
            b_frag,
            c_frag
        );

        // Needed before As/Bs are overwritten by the next K tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -------------------------------------------------
    // Output
    // -------------------------------------------------

    const uint out_row0 = tile_row0 + frag_row;
    const uint out_col0 = tile_col0 + frag_col;

    // Threadgroup-uniform so the barrier below is reached by all simdgroups.
    const bool full_fragment =
        (tile_row0 + BM <= dims.m) &&
        (tile_col0 + BN <= dims.n);

    if (full_fragment) {

        simdgroup_store(
            c_frag,
            C + out_row0 * dims.n + out_col0,
            dims.n
        );

    } else {

        // Store fragment to threadgroup memory first,
        // then copy only valid elements.
        simdgroup_store(
            c_frag,
            (threadgroup float *)Cs[simdgroup_id],
            FN
        );

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Only lanes belonging to this SIMD group copy its 64 outputs.
        const uint lane = tid % SIMD_WIDTH;

        #pragma unroll
        for (uint i = 0; i < 2; ++i) {

            const uint idx = lane * 2 + i;

            const uint local_row = idx / FN;
            const uint local_col = idx % FN;

            const uint global_row = out_row0 + local_row;
            const uint global_col = out_col0 + local_col;

            if (global_row < dims.m &&
                global_col < dims.n) {

                C[global_row * dims.n + global_col] =
                    Cs[simdgroup_id][local_row][local_col];
            }
        }
    }
}