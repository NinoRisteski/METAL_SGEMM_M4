#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint m;
    uint n;
    uint k;
};

static inline float4 load4_or_zero(const device float *src,
                                   uint base,
                                   bool can_load4,
                                   bool valid0,
                                   bool valid1,
                                   bool valid2,
                                   bool valid3)
{
    if (can_load4) {
        return *((const device float4 *)(src + base));
    }

    return float4(
        valid0 ? src[base] : 0.0f,
        valid1 ? src[base + 1] : 0.0f,
        valid2 ? src[base + 2] : 0.0f,
        valid3 ? src[base + 3] : 0.0f
    );
}

// 4 SIMD groups cooperate on one 32x32 C tile.
//
// Each SIMD group owns a 16x16 region made from four 8x8 fragments:
//
// SG0: rows  0..15, cols  0..15
// SG1: rows  0..15, cols 16..31
// SG2: rows 16..31, cols  0..15
// SG3: rows 16..31, cols 16..31
//
// Each SIMD group holds a 2x2 grid of 8x8 accumulator fragments:
//
//        c00 | c01
//       -----+-----
//        c10 | c11
//
// Per K-step:
//   A tile = 32x8
//   B tile = 8x32
//
// Compared with the previous SIMD 16x16 kernel, each SIMD group now
// performs FOUR 8x8x8 matrix multiplies between threadgroup barriers.
//
// BK=16 variant with float4 global->threadgroup loads: two FK=8 steps (8 MMAs) per barrier pair.

kernel void sgemm_simdgroup_32x32_multifrag_bk16_vec4(
    const device float *A [[buffer(0)]],
    const device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant MatrixDims &dims [[buffer(3)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint BM = 32;
    constexpr uint BN = 32;
    constexpr uint BK = 16;

    constexpr uint FM = 8;
    constexpr uint FN = 8;
    constexpr uint FK = 8;

    constexpr uint SIMD_WIDTH = 32;
    constexpr uint NUM_SIMDGROUPS = 4;

    // Shared A/B tiles.
    //
    // A: 32x16 = 512 floats
    // B: 16x32 = 512 floats
    threadgroup float As[BM][BK];
    threadgroup float Bs[BK][BN];

    // Edge-output scratch.
    //
    // Four SIMD groups × four 8x8 fragments.
    threadgroup float Cs[NUM_SIMDGROUPS][4][FM][FN];

    const uint simdgroup_id = tid / SIMD_WIDTH;
    const uint lane = tid % SIMD_WIDTH;

    // 2x2 arrangement of SIMD groups over the 32x32 output tile.
    const uint sg_row = simdgroup_id / 2;
    const uint sg_col = simdgroup_id % 2;

    const uint tile_row0 = tg_pos.y * BM;
    const uint tile_col0 = tg_pos.x * BN;

    // Each SIMD group owns a 16x16 output region.
    const uint sg_row0 = sg_row * 16;
    const uint sg_col0 = sg_col * 16;

    // Four accumulator fragments per SIMD group.
    simdgroup_matrix<float, FM, FN> c00 =
        make_filled_simdgroup_matrix<float, FM, FN>(0.0f);

    simdgroup_matrix<float, FM, FN> c01 =
        make_filled_simdgroup_matrix<float, FM, FN>(0.0f);

    simdgroup_matrix<float, FM, FN> c10 =
        make_filled_simdgroup_matrix<float, FM, FN>(0.0f);

    simdgroup_matrix<float, FM, FN> c11 =
        make_filled_simdgroup_matrix<float, FM, FN>(0.0f);

    for (uint k0 = 0; k0 < dims.k; k0 += BK) {

        // ------------------------------------------------------------
        // Cooperative A load (float4)
        //
        // 32x16 = 512 floats = 128 float4 -> one float4 per thread.
        // ------------------------------------------------------------
        {
            const uint idx = tid * 4;
            const uint local_row = idx / BK;
            const uint local_k   = idx % BK;

            const uint global_row = tile_row0 + local_row;
            const uint global_k   = k0 + local_k;
            const uint base = global_row * dims.k + global_k;

            const bool row_ok = global_row < dims.m;
            const bool can_load4 =
                row_ok && ((dims.k & 3u) == 0u) && (global_k + 3u) < dims.k;

            const float4 vals = load4_or_zero(
                A, base, can_load4,
                row_ok && global_k < dims.k,
                row_ok && (global_k + 1u) < dims.k,
                row_ok && (global_k + 2u) < dims.k,
                row_ok && (global_k + 3u) < dims.k
            );

            As[local_row][local_k]      = vals.x;
            As[local_row][local_k + 1u] = vals.y;
            As[local_row][local_k + 2u] = vals.z;
            As[local_row][local_k + 3u] = vals.w;
        }

        // ------------------------------------------------------------
        // Cooperative B load (float4)
        //
        // 16x32 = 512 floats = 128 float4 -> one float4 per thread.
        // ------------------------------------------------------------
        {
            const uint idx = tid * 4;
            const uint local_k   = idx / BN;
            const uint local_col = idx % BN;

            const uint global_k   = k0 + local_k;
            const uint global_col = tile_col0 + local_col;
            const uint base = global_k * dims.n + global_col;

            const bool row_ok = global_k < dims.k;
            const bool can_load4 =
                row_ok && ((dims.n & 3u) == 0u) && (global_col + 3u) < dims.n;

            const float4 vals = load4_or_zero(
                B, base, can_load4,
                row_ok && global_col < dims.n,
                row_ok && (global_col + 1u) < dims.n,
                row_ok && (global_col + 2u) < dims.n,
                row_ok && (global_col + 3u) < dims.n
            );

            Bs[local_k][local_col]      = vals.x;
            Bs[local_k][local_col + 1u] = vals.y;
            Bs[local_k][local_col + 2u] = vals.z;
            Bs[local_k][local_col + 3u] = vals.w;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ------------------------------------------------------------
        // Load two A fragments and two B fragments for this SIMD group.
        //
        // These produce the four 8x8 output fragments:
        //
        // a0*b0 -> c00
        // a0*b1 -> c01
        // a1*b0 -> c10
        // a1*b1 -> c11
        // ------------------------------------------------------------

        #pragma unroll
        for (uint kk = 0; kk < BK; kk += FK) {
            simdgroup_matrix<float, FM, FK> a0;
            simdgroup_matrix<float, FM, FK> a1;

            simdgroup_matrix<float, FK, FN> b0;
            simdgroup_matrix<float, FK, FN> b1;

            const uint a_row0 = sg_row0;
            const uint a_row1 = sg_row0 + 8;

            const uint b_col0 = sg_col0;
            const uint b_col1 = sg_col0 + 8;

            simdgroup_load(
                a0,
                &As[a_row0][kk],
                BK
            );

            simdgroup_load(
                a1,
                &As[a_row1][kk],
                BK
            );

            simdgroup_load(
                b0,
                &Bs[kk][b_col0],
                BN
            );

            simdgroup_load(
                b1,
                &Bs[kk][b_col1],
                BN
            );

            // Four matrix operations per SIMD group per K-step.
            simdgroup_multiply_accumulate(
                c00,
                a0,
                b0,
                c00
            );

            simdgroup_multiply_accumulate(
                c01,
                a0,
                b1,
                c01
            );

            simdgroup_multiply_accumulate(
                c10,
                a1,
                b0,
                c10
            );

            simdgroup_multiply_accumulate(
                c11,
                a1,
                b1,
                c11
            );

        }

        // Protect As/Bs before loading the next K tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ------------------------------------------------------------
    // Output
    // ------------------------------------------------------------

    const bool full_output_tile =
        (tile_row0 + BM <= dims.m) &&
        (tile_col0 + BN <= dims.n);

    if (full_output_tile) {

        // SIMD group's 16x16 region:
        //
        // c00 | c01
        // ---- + ----
        // c10 | c11

        const uint global_row0 = tile_row0 + sg_row0;
        const uint global_col0 = tile_col0 + sg_col0;

        simdgroup_store(
            c00,
            C + global_row0 * dims.n + global_col0,
            dims.n
        );

        simdgroup_store(
            c01,
            C + global_row0 * dims.n + global_col0 + 8,
            dims.n
        );

        simdgroup_store(
            c10,
            C + (global_row0 + 8) * dims.n + global_col0,
            dims.n
        );

        simdgroup_store(
            c11,
            C + (global_row0 + 8) * dims.n + global_col0 + 8,
            dims.n
        );

    } else {

        // All threads take the same branch because full_output_tile
        // is threadgroup-wide.
        //
        // Store fragments to shared scratch, then copy only valid
        // elements to device memory.

        simdgroup_store(
            c00,
            (threadgroup float *)Cs[simdgroup_id][0],
            FN
        );

        simdgroup_store(
            c01,
            (threadgroup float *)Cs[simdgroup_id][1],
            FN
        );

        simdgroup_store(
            c10,
            (threadgroup float *)Cs[simdgroup_id][2],
            FN
        );

        simdgroup_store(
            c11,
            (threadgroup float *)Cs[simdgroup_id][3],
            FN
        );

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each lane copies two values from each 8x8 fragment.
        #pragma unroll
        for (uint frag = 0; frag < 4; ++frag) {

            const uint frag_row =
                (frag / 2) * 8;

            const uint frag_col =
                (frag % 2) * 8;

            #pragma unroll
            for (uint i = 0; i < 2; ++i) {

                const uint idx = lane * 2 + i;

                const uint local_row = idx / FN;
                const uint local_col = idx % FN;

                const uint global_row =
                    tile_row0 +
                    sg_row0 +
                    frag_row +
                    local_row;

                const uint global_col =
                    tile_col0 +
                    sg_col0 +
                    frag_col +
                    local_col;

                if (global_row < dims.m &&
                    global_col < dims.n) {

                    C[global_row * dims.n + global_col] =
                        Cs[simdgroup_id][frag][local_row][local_col];
                }
            }
        }
    }
}