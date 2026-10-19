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

// Double-buffered variant of the 32x32 MultiFrag BK16 Vec4 kernel.
//
// 4 SIMD groups, each owning a 2x2 grid of 8x8 accumulators (16x16 region).
// Two threadgroup buffers for A and B. Per K-tile:
//   1. issue global float4 loads for tile t+1 into registers
//   2. run the MMAs on tile t (from buffer t&1)
//   3. store the registers into buffer (t+1)&1
//   4. one barrier
// The global-load latency for the next tile overlaps with the MMAs, and there
// is one barrier per K-tile instead of two.
kernel void sgemm_simdgroup_32x32_multifrag_bk16_vec4_db(
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

    threadgroup float As[2][BM][BK];
    threadgroup float Bs[2][BK][BN];

    // Edge-output scratch (only used for partial output tiles).
    threadgroup float Cs[NUM_SIMDGROUPS][4][FM][FN];

    const uint simdgroup_id = tid / SIMD_WIDTH;
    const uint lane = tid % SIMD_WIDTH;

    const uint sg_row = simdgroup_id / 2;
    const uint sg_col = simdgroup_id % 2;

    const uint tile_row0 = tg_pos.y * BM;
    const uint tile_col0 = tg_pos.x * BN;

    const uint sg_row0 = sg_row * 16;
    const uint sg_col0 = sg_col * 16;

    simdgroup_matrix<float, FM, FN> c00 =
        make_filled_simdgroup_matrix<float, FM, FN>(0.0f);
    simdgroup_matrix<float, FM, FN> c01 =
        make_filled_simdgroup_matrix<float, FM, FN>(0.0f);
    simdgroup_matrix<float, FM, FN> c10 =
        make_filled_simdgroup_matrix<float, FM, FN>(0.0f);
    simdgroup_matrix<float, FM, FN> c11 =
        make_filled_simdgroup_matrix<float, FM, FN>(0.0f);

    // Per-thread load coordinates (one float4 of A and one of B per tile).
    const uint a_idx = tid * 4;
    const uint a_row = a_idx / BK;
    const uint a_col = a_idx % BK;

    const uint b_idx = tid * 4;
    const uint b_row = b_idx / BN;
    const uint b_col = b_idx % BN;

    const uint global_a_row = tile_row0 + a_row;
    const uint global_b_col = tile_col0 + b_col;

    const bool a_row_ok = global_a_row < dims.m;
    const bool a_aligned = (dims.k & 3u) == 0u;
    const bool b_aligned = (dims.n & 3u) == 0u;

    auto load_a = [&](uint k0) -> float4 {
        const uint gk = k0 + a_col;
        const uint base = global_a_row * dims.k + gk;
        const bool can_load4 = a_row_ok && a_aligned && (gk + 3u) < dims.k;
        return load4_or_zero(
            A, base, can_load4,
            a_row_ok && gk < dims.k,
            a_row_ok && (gk + 1u) < dims.k,
            a_row_ok && (gk + 2u) < dims.k,
            a_row_ok && (gk + 3u) < dims.k
        );
    };

    auto load_b = [&](uint k0) -> float4 {
        const uint gk = k0 + b_row;
        const bool row_ok = gk < dims.k;
        const uint base = gk * dims.n + global_b_col;
        const bool can_load4 = row_ok && b_aligned && (global_b_col + 3u) < dims.n;
        return load4_or_zero(
            B, base, can_load4,
            row_ok && global_b_col < dims.n,
            row_ok && (global_b_col + 1u) < dims.n,
            row_ok && (global_b_col + 2u) < dims.n,
            row_ok && (global_b_col + 3u) < dims.n
        );
    };

    auto store_tile = [&](uint buf, float4 av, float4 bv) {
        As[buf][a_row][a_col]      = av.x;
        As[buf][a_row][a_col + 1u] = av.y;
        As[buf][a_row][a_col + 2u] = av.z;
        As[buf][a_row][a_col + 3u] = av.w;

        Bs[buf][b_row][b_col]      = bv.x;
        Bs[buf][b_row][b_col + 1u] = bv.y;
        Bs[buf][b_row][b_col + 2u] = bv.z;
        Bs[buf][b_row][b_col + 3u] = bv.w;
    };

    // Prologue: stage K-tile 0 into buffer 0.
    store_tile(0, load_a(0), load_b(0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0;
    for (uint k0 = 0; k0 < dims.k; k0 += BK) {
        const uint next_k0 = k0 + BK;
        const bool has_next = next_k0 < dims.k;

        // Issue next tile's global loads; consumed after the MMAs below.
        float4 a_next = float4(0.0f);
        float4 b_next = float4(0.0f);
        if (has_next) {
            a_next = load_a(next_k0);
            b_next = load_b(next_k0);
        }

        #pragma unroll
        for (uint kk = 0; kk < BK; kk += FK) {
            simdgroup_matrix<float, FM, FK> a0;
            simdgroup_matrix<float, FM, FK> a1;
            simdgroup_matrix<float, FK, FN> b0;
            simdgroup_matrix<float, FK, FN> b1;

            simdgroup_load(a0, &As[buf][sg_row0][kk], BK);
            simdgroup_load(a1, &As[buf][sg_row0 + 8][kk], BK);
            simdgroup_load(b0, &Bs[buf][kk][sg_col0], BN);
            simdgroup_load(b1, &Bs[buf][kk][sg_col0 + 8], BN);

            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }

        // Write the prefetched tile into the other buffer. Nobody reads that
        // buffer in this iteration, and the previous iteration's barrier
        // guarantees all reads of it have finished.
        if (has_next) {
            store_tile(buf ^ 1u, a_next, b_next);
        }

        // Makes the freshly stored tile visible and protects the buffer just
        // read from being overwritten in the next iteration.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1u;
    }

    // Output. full_output_tile is threadgroup-uniform so the barrier below is
    // reached by every SIMD group.
    const bool full_output_tile =
        (tile_row0 + BM <= dims.m) &&
        (tile_col0 + BN <= dims.n);

    if (full_output_tile) {
        const uint row0 = tile_row0 + sg_row0;
        const uint col0 = tile_col0 + sg_col0;

        simdgroup_store(c00, C + row0 * dims.n + col0, dims.n);
        simdgroup_store(c01, C + row0 * dims.n + col0 + 8, dims.n);
        simdgroup_store(c10, C + (row0 + 8) * dims.n + col0, dims.n);
        simdgroup_store(c11, C + (row0 + 8) * dims.n + col0 + 8, dims.n);
    } else {
        simdgroup_store(c00, (threadgroup float *)Cs[simdgroup_id][0], FN);
        simdgroup_store(c01, (threadgroup float *)Cs[simdgroup_id][1], FN);
        simdgroup_store(c10, (threadgroup float *)Cs[simdgroup_id][2], FN);
        simdgroup_store(c11, (threadgroup float *)Cs[simdgroup_id][3], FN);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint frag = 0; frag < 4; ++frag) {
            const uint frag_row = (frag / 2) * 8;
            const uint frag_col = (frag % 2) * 8;

            #pragma unroll
            for (uint i = 0; i < 2; ++i) {
                const uint idx = lane * 2 + i;
                const uint local_row = idx / FN;
                const uint local_col = idx % FN;

                const uint global_row =
                    tile_row0 + sg_row0 + frag_row + local_row;
                const uint global_col =
                    tile_col0 + sg_col0 + frag_col + local_col;

                if (global_row < dims.m && global_col < dims.n) {
                    C[global_row * dims.n + global_col] =
                        Cs[simdgroup_id][frag][local_row][local_col];
                }
            }
        }
    }
}
