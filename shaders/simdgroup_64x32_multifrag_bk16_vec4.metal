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

// 4 SIMD groups cooperate on one 64x32 C tile (2x2 arrangement).
// Each SIMD group owns a 32x16 region = a 4x2 grid of 8x8 accumulator
// fragments (8 accumulators), so per K-step of 8 it loads 4 A + 2 B fragments
// and issues 8 MMAs.
//
//   SG0: rows  0..31, cols  0..15    SG1: rows  0..31, cols 16..31
//   SG2: rows 32..63, cols  0..15    SG3: rows 32..63, cols 16..31
//
// BK=16, float4 global->threadgroup loads:
//   A tile 64x16 = 1024 floats = 256 float4 -> 2 per thread
//   B tile 16x32 =  512 floats = 128 float4 -> 1 per thread
kernel void sgemm_simdgroup_64x32_multifrag_bk16_vec4(
    const device float *A [[buffer(0)]],
    const device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant MatrixDims &dims [[buffer(3)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint BM = 64;
    constexpr uint BN = 32;
    constexpr uint BK = 16;

    constexpr uint FM = 8;
    constexpr uint FN = 8;
    constexpr uint FK = 8;

    constexpr uint FRAGS_M = 4;  // per SIMD group
    constexpr uint FRAGS_N = 2;  // per SIMD group

    constexpr uint SIMD_WIDTH = 32;
    constexpr uint NUM_SIMDGROUPS = 4;
    constexpr uint THREADS = SIMD_WIDTH * NUM_SIMDGROUPS;  // 128

    threadgroup float As[BM][BK];
    threadgroup float Bs[BK][BN];

    // Edge-output scratch: one 8x8 buffer per fragment per SIMD group.
    threadgroup float Cs[NUM_SIMDGROUPS][FRAGS_M * FRAGS_N][FM][FN];

    const uint simdgroup_id = tid / SIMD_WIDTH;
    const uint lane = tid % SIMD_WIDTH;

    const uint sg_row = simdgroup_id / 2;
    const uint sg_col = simdgroup_id % 2;

    const uint tile_row0 = tg_pos.y * BM;
    const uint tile_col0 = tg_pos.x * BN;

    const uint sg_row0 = sg_row * (FRAGS_M * FM);  // 0 or 32
    const uint sg_col0 = sg_col * (FRAGS_N * FN);  // 0 or 16

    simdgroup_matrix<float, FM, FN> c[FRAGS_M][FRAGS_N];

    #pragma unroll
    for (uint i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (uint j = 0; j < FRAGS_N; ++j) {
            c[i][j] = make_filled_simdgroup_matrix<float, FM, FN>(0.0f);
        }
    }

    for (uint k0 = 0; k0 < dims.k; k0 += BK) {

        // A load: 2 float4 per thread.
        #pragma unroll
        for (uint i = 0; i < 2; ++i) {
            const uint idx = (tid + i * THREADS) * 4;
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

        // B load: 1 float4 per thread.
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

        #pragma unroll
        for (uint kk = 0; kk < BK; kk += FK) {
            simdgroup_matrix<float, FM, FK> a[FRAGS_M];
            simdgroup_matrix<float, FK, FN> b[FRAGS_N];

            #pragma unroll
            for (uint i = 0; i < FRAGS_M; ++i) {
                simdgroup_load(a[i], &As[sg_row0 + i * FM][kk], BK);
            }

            #pragma unroll
            for (uint j = 0; j < FRAGS_N; ++j) {
                simdgroup_load(b[j], &Bs[kk][sg_col0 + j * FN], BN);
            }

            #pragma unroll
            for (uint i = 0; i < FRAGS_M; ++i) {
                #pragma unroll
                for (uint j = 0; j < FRAGS_N; ++j) {
                    simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]);
                }
            }
        }

        // Protect As/Bs before loading the next K tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Threadgroup-uniform so the barrier in the edge path is reached by all
    // SIMD groups.
    const bool full_output_tile =
        (tile_row0 + BM <= dims.m) &&
        (tile_col0 + BN <= dims.n);

    if (full_output_tile) {
        const uint row0 = tile_row0 + sg_row0;
        const uint col0 = tile_col0 + sg_col0;

        #pragma unroll
        for (uint i = 0; i < FRAGS_M; ++i) {
            #pragma unroll
            for (uint j = 0; j < FRAGS_N; ++j) {
                simdgroup_store(
                    c[i][j],
                    C + (row0 + i * FM) * dims.n + (col0 + j * FN),
                    dims.n
                );
            }
        }
    } else {
        #pragma unroll
        for (uint i = 0; i < FRAGS_M; ++i) {
            #pragma unroll
            for (uint j = 0; j < FRAGS_N; ++j) {
                simdgroup_store(
                    c[i][j],
                    (threadgroup float *)Cs[simdgroup_id][i * FRAGS_N + j],
                    FN
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each lane copies two values from each 8x8 fragment.
        #pragma unroll
        for (uint i = 0; i < FRAGS_M; ++i) {
            #pragma unroll
            for (uint j = 0; j < FRAGS_N; ++j) {
                #pragma unroll
                for (uint e = 0; e < 2; ++e) {
                    const uint idx = lane * 2 + e;
                    const uint local_row = idx / FN;
                    const uint local_col = idx % FN;

                    const uint global_row =
                        tile_row0 + sg_row0 + i * FM + local_row;
                    const uint global_col =
                        tile_col0 + sg_col0 + j * FN + local_col;

                    if (global_row < dims.m && global_col < dims.n) {
                        C[global_row * dims.n + global_col] =
                            Cs[simdgroup_id][i * FRAGS_N + j][local_row][local_col];
                    }
                }
            }
        }
    }
}
