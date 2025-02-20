__device__ void sgemm_128x128(int M, int N, int K, float *A, float *B, float *C, int block_base_y, int block_base_x, int block_k, int kstride, float *smem)
{
    float *smemB = smem;
    float *smemA = smem + 16 * 256;

    int tx = threadIdx.x;

    // Warp tile
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    int B_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    int A_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    int x = block_base_y + A_lds_addr;
    int y = block_base_x + B_lds_addr;

    // sts addr
    int B_sts_addr = (tx % 8) * 132 +
                     (tx / 8) * 4;
    int A_sts_addr = (tx / 32) * 128 + (tx % 32);

    // ldg addr
    int B_ldg_addr = (block_base_x + tx / 8 * 4) * K + tx % 8 + kstride * block_k;
    int A_ldg_addr = (tx / 32 + kstride * block_k) * M + block_base_y + tx % 32;
    float A_ldg_reg[4];
    float B_ldg_reg[4];
// ldg
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if ((tx % 8 + kstride * block_k) < K && (block_base_x + tx / 8 * 4 + i) < N)
            B_ldg_reg[i] = B[B_ldg_addr + i * K];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if ((tx / 32 + kstride * block_k) < K && (block_base_y + tx % 32 + i * 32) < M)
            A_ldg_reg[i] = A[A_ldg_addr + i * 32];
    }

// sts
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        smemB[B_sts_addr + i] = B_ldg_reg[i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        smemA[A_sts_addr + i * 32] = A_ldg_reg[i];
    }
    __syncthreads();
    int write_stage_idx = 1;
    float A_frag[2][8];
    float B_frag[2][8];
    float output_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            output_frag[i][j] = 0;
        }
    }
    // lds
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        B_frag[0][i] = smemB[B_lds_addr + i];
        B_frag[0][i + 4] = smemB[B_lds_addr + i + 16];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        A_frag[0][i] = smemA[A_lds_addr + i];
        A_frag[0][i + 4] = smemA[A_lds_addr + i + 32];
    }

    for (int k = 0; k < kstride; k += 8)
    {
        // ldg
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if ((tx % 8 + kstride * block_k + k + 8) < K && (block_base_x + tx / 8 * 4 + i) < N)
                B_ldg_reg[i] = B[B_ldg_addr + i * K + (k + 8)];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if ((tx / 32 + kstride * block_k + k + 8) < K && (block_base_y + tx % 32 + i * 32) < M)
                A_ldg_reg[i] = A[A_ldg_addr + i * 32 + (k + 8) * M];
        }
        int load_stage_idx = write_stage_idx ^ 1;
#pragma unroll
        for (int subk = 0; subk < 8 - 1; ++subk)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                B_frag[(subk + 1) % 2][i] = smemB[load_stage_idx * 8 * 132 + B_lds_addr + (subk + 1) * 132 + i];
                B_frag[(subk + 1) % 2][i + 4] = smemB[load_stage_idx * 8 * 132 + B_lds_addr + (subk + 1) * 132 + i + 16];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                A_frag[(subk + 1) % 2][i] = smemA[load_stage_idx * 8 * 128 + A_lds_addr + (subk + 1) * 128 + i];
                A_frag[(subk + 1) % 2][i + 4] = smemA[load_stage_idx * 8 * 128 + A_lds_addr + (subk + 1) * 128 + i + 32];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += B_frag[subk % 2][i] * A_frag[subk % 2][j];
                }
            }
        }
        // sts
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            smemB[write_stage_idx * 8 * 132 + B_sts_addr + i] = B_ldg_reg[i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            smemA[write_stage_idx * 8 * 128 + A_sts_addr + i * 32] = A_ldg_reg[i];
        }
        __syncthreads();
        write_stage_idx ^= 1;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            B_frag[0][i] = smemB[(load_stage_idx ^ 1) * 8 * 132 + B_lds_addr + 0 * 132 + i];
            B_frag[0][i + 4] = smemB[(load_stage_idx ^ 1) * 8 * 132 + B_lds_addr + 0 * 132 + i + 16];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            A_frag[0][i] = smemA[(load_stage_idx ^ 1) * 8 * 128 + A_lds_addr + 0 * 128 + i];
            A_frag[0][i + 4] = smemA[(load_stage_idx ^ 1) * 8 * 128 + A_lds_addr + 0 * 128 + i + 32];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                output_frag[i][j] += B_frag[1][i] * A_frag[1][j];
            }
        }
    }

    int outOffset;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            outOffset = block_k * M * N + (y + i) * M + x + j;
            if (x + j < M && y + i < N)
            {
                C[outOffset] = output_frag[i][j];
            }
            outOffset = block_k * M * N + (y + i) * M + x + j + 32;
            if (x + j + 32 < M && y + i < N)
            {
                C[outOffset] = output_frag[i][j + 4];
            }
            outOffset = block_k * M * N + (y + i + 16) * M + x + j;
            if (x + j < M && y + i + 16 < N)
            {
                C[outOffset] = output_frag[i + 4][j];
            }
            outOffset = block_k * M * N + (y + i + 16) * M + x + j + 32;
            if (x + j + 32 < M && y + i + 16 < N)
            {
                C[outOffset] = output_frag[i + 4][j + 4];
            }
        }
    }
}

__device__ void sgemm_128x64(int M, int N, int K, float *A, float *B, float *C, int block_base_y, int block_base_x, int block_k, int kstride, float *smem)
{
    float *smemB = smem;
    float *smemA = smem + 16 * 128;

    int tx = threadIdx.x;

    // Warp tile
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    int B_lds_addr = (warp_id / 2) * 16 + mma_tid_y * 4;
    int A_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    int x = block_base_y + A_lds_addr;
    int y = block_base_x + B_lds_addr;

    // sts addr
    int B_sts_addr = (tx % 8) * 68 +
                     (tx / 8) * 2;
    int A_sts_addr = (tx / 32) * 128 + (tx % 32);

    // ldg addr
    int B_ldg_addr = (block_base_x + tx / 8 * 2) * K + tx % 8 + kstride * block_k;
    int A_ldg_addr = (tx / 32 + kstride * block_k) * M + block_base_y + tx % 32;
    float B_ldg_reg[2];
    float A_ldg_reg[4];
// ldg
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        if ((tx % 8 + kstride * block_k) < K && (block_base_x + tx / 8 * 2 + i) < N)
            B_ldg_reg[i] = B[B_ldg_addr + i * K];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if ((tx / 32 + kstride * block_k) < K && (block_base_y + tx % 32 + i * 32) < M)
            A_ldg_reg[i] = A[A_ldg_addr + i * 32];
    }

// sts
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        smemB[B_sts_addr + i] = B_ldg_reg[i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        smemA[A_sts_addr + i * 32] = A_ldg_reg[i];
    }
    __syncthreads();
    int write_stage_idx = 1;
    float B_frag[2][4];
    float A_frag[2][8];
    float output_frag[4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            output_frag[i][j] = 0;
        }
    }
    // lds
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        B_frag[0][i] = smemB[B_lds_addr + i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        A_frag[0][i] = smemA[A_lds_addr + i];
        A_frag[0][i + 4] = smemA[A_lds_addr + i + 32];
    }

    for (int k = 0; k < kstride; k += 8)
    {
        // ldg
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            if ((tx % 8 + kstride * block_k + k + 8) < K && (block_base_x + tx / 8 * 2 + i) < N)
                B_ldg_reg[i] = B[B_ldg_addr + i * K + (k + 8)];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if ((tx / 32 + kstride * block_k + k + 8) < K && (block_base_y + tx % 32 + i * 32) < M)
                A_ldg_reg[i] = A[A_ldg_addr + i * 32 + (k + 8) * M];
        }
        int load_stage_idx = write_stage_idx ^ 1;
#pragma unroll
        for (int subk = 0; subk < 8 - 1; ++subk)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                B_frag[(subk + 1) % 2][i] = smemB[load_stage_idx * 68 * 8 + B_lds_addr + (subk + 1) * 68 + i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                A_frag[(subk + 1) % 2][i] = smemA[load_stage_idx * 128 * 8 + A_lds_addr + (subk + 1) * 128 + i];
                A_frag[(subk + 1) % 2][i + 4] = smemA[load_stage_idx * 128 * 8 + A_lds_addr + (subk + 1) * 128 + i + 32];
            }

#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += B_frag[subk % 2][i] * A_frag[subk % 2][j];
                }
            }
        }
        // sts
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            smemB[write_stage_idx * 68 * 8 + B_sts_addr + i] = B_ldg_reg[i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            smemA[write_stage_idx * 128 * 8 + A_sts_addr + i * 32] = A_ldg_reg[i];
        }
        __syncthreads();
        write_stage_idx ^= 1;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            B_frag[0][i] = smemB[(load_stage_idx ^ 1) * 68 * 8 + B_lds_addr + 0 * 68 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            A_frag[0][i] = smemA[(load_stage_idx ^ 1) * 128 * 8 + A_lds_addr + 0 * 128 + i];
            A_frag[0][i + 4] = smemA[(load_stage_idx ^ 1) * 128 * 8 + A_lds_addr + 0 * 128 + i + 32];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                output_frag[i][j] += B_frag[1][i] * A_frag[1][j];
            }
        }
    }

    int outOffset;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            outOffset = block_k * M * N + (y + i) * M + x + j;
            if (x + j < M && y + i < N)
            {
                C[outOffset] = output_frag[i][j];
            }
            outOffset = block_k * M * N + (y + i) * M + x + j + 32;
            if (x + j + 32 < M && y + i < N)
            {
                C[outOffset] = output_frag[i][j + 4];
            }
        }
    }
}

__device__ void sgemm_128x32(int M, int N, int K, float *A, float *B, float *C, int block_base_y, int block_base_x, int block_k, int kstride, float *smem)
{
    float *smemB = smem;
    float *smemA = smem + 8 * 128;

    int tx = threadIdx.x;

    // Warp tile
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    int B_lds_addr = (warp_id / 4) * 16 + mma_tid_y * 4;
    int A_lds_addr = (warp_id % 4) * 32 + mma_tid_x * 4;

    int x = block_base_y + A_lds_addr;
    int y = block_base_x + B_lds_addr;

    // sts addr
    int B_sts_addr = (tx % 8) * 36 +
                     (tx / 8);
    int A_sts_addr = (tx / 32) * 128 + (tx % 32);

    // ldg addr
    int B_ldg_addr = (block_base_x + tx / 8) * K + tx % 8 + kstride * block_k;
    int A_ldg_addr = (tx / 32 + kstride * block_k) * M + block_base_y + tx % 32;
    float B_ldg_reg[1];
    float A_ldg_reg[4];
// ldg
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        if ((tx % 8 + kstride * block_k) < K && (block_base_x + tx / 8 + i) < N)
            B_ldg_reg[i] = B[B_ldg_addr + i * K];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if ((tx / 32 + kstride * block_k) < K && (block_base_y + tx % 32 + i * 32) < M)
            A_ldg_reg[i] = A[A_ldg_addr + i * 32];
    }

// sts
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        smemB[B_sts_addr + i] = B_ldg_reg[i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        smemA[A_sts_addr + i * 32] = A_ldg_reg[i];
    }
    __syncthreads();
    int write_stage_idx = 1;
    float A_frag[2][4];
    float B_frag[2][4];
    float output_frag[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            output_frag[i][j] = 0;
        }
    }
    // lds
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        B_frag[0][i] = smemB[B_lds_addr + i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        A_frag[0][i] = smemA[A_lds_addr + i];
    }

    for (int k = 0; k < kstride; k += 8)
    {
        // ldg
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            if ((tx % 8 + kstride * block_k + k + 8) < K && (block_base_x + tx / 8 + i) < N)
                B_ldg_reg[i] = B[B_ldg_addr + i * K + (k + 8)];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if ((tx / 32 + kstride * block_k + k + 8) < K && (block_base_y + tx % 32 + i * 32) < M)
                A_ldg_reg[i] = A[A_ldg_addr + i * 32 + (k + 8) * M];
        }
        int load_stage_idx = write_stage_idx ^ 1;
#pragma unroll
        for (int subk = 0; subk < 8 - 1; ++subk)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                B_frag[(subk + 1) % 2][i] = smemB[load_stage_idx * 36 * 8 + B_lds_addr + (subk + 1) * 36 + i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                A_frag[(subk + 1) % 2][i] = smemA[load_stage_idx * 128 * 8 + A_lds_addr + (subk + 1) * 128 + i];
            }

#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    output_frag[i][j] += B_frag[subk % 2][i] * A_frag[subk % 2][j];
                }
            }
        }
        // sts
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            smemB[write_stage_idx * 36 * 8 + B_sts_addr + i] = B_ldg_reg[i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            smemA[write_stage_idx * 128 * 8 + A_sts_addr + i * 32] = A_ldg_reg[i];
        }
        __syncthreads();
        write_stage_idx ^= 1;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            B_frag[0][i] = smemB[(load_stage_idx ^ 1) * 36 * 8 + B_lds_addr + 0 * 36 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            A_frag[0][i] = smemA[(load_stage_idx ^ 1) * 128 * 8 + A_lds_addr + 0 * 128 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                output_frag[i][j] += B_frag[1][i] * A_frag[1][j];
            }
        }
    }

    int outOffset;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            outOffset = block_k * M * N + (y + i) * M + x + j;
            if (x + j < M && y + i < N)
            {
                C[outOffset] = output_frag[i][j];
            }
        }
    }
}

__device__ void sgemm_64x128(int M, int N, int K, float *A, float *B, float *C, int block_base_y, int block_base_x, int block_k, int kstride, float *smem)
{
    float *smemB = smem;
    float *smemA = smem + 16 * 256;

    int tx = threadIdx.x;

    // Warp tile
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    int B_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    int A_lds_addr = (warp_id % 2) * 32 + mma_tid_x * 4;

    int x = block_base_y + A_lds_addr;
    int y = block_base_x + B_lds_addr;

    // sts addr
    int B_sts_addr = (tx % 8) * 132 +
                     (tx / 8) * 4;
    int A_sts_addr = (tx / 32) * 64 + (tx % 32);

    // ldg addr
    int B_ldg_addr = (block_base_x + tx / 8 * 4) * K + tx % 8 + kstride * block_k;
    int A_ldg_addr = (tx / 32 + kstride * block_k) * M + block_base_y + tx % 32;
    float B_ldg_reg[4];
    float A_ldg_reg[2];
// ldg
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if ((tx % 8 + kstride * block_k) < K && (block_base_x + tx / 8 * 4 + i) < N)
            B_ldg_reg[i] = B[B_ldg_addr + i * K];
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        if ((tx / 32 + kstride * block_k) < K && (block_base_y + tx % 32 + i * 32) < M)
            A_ldg_reg[i] = A[A_ldg_addr + i * 32];
    }

// sts
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        smemB[B_sts_addr + i] = B_ldg_reg[i];
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        smemA[A_sts_addr + i * 32] = A_ldg_reg[i];
    }
    __syncthreads();
    int write_stage_idx = 1;
    float B_frag[2][8];
    float A_frag[2][4];
    float output_frag[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            output_frag[i][j] = 0;
        }
    }
    // lds
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        B_frag[0][i] = smemB[B_lds_addr + i];
        B_frag[0][i + 4] = smemB[B_lds_addr + i + 16];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        A_frag[0][i] = smemA[A_lds_addr + i];
    }

    for (int k = 0; k < kstride; k += 8)
    {
        // ldg
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if ((tx % 8 + kstride * block_k + k + 8) < K && (block_base_x + tx / 8 * 4 + i) < N)
                B_ldg_reg[i] = B[B_ldg_addr + i * K + (k + 8)];
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            if ((tx / 32 + kstride * block_k + k + 8) < K && (block_base_y + tx % 32 + i * 32) < M)
                A_ldg_reg[i] = A[A_ldg_addr + i * 32 + (k + 8) * M];
        }
        int load_stage_idx = write_stage_idx ^ 1;
#pragma unroll
        for (int subk = 0; subk < 8 - 1; ++subk)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                B_frag[(subk + 1) % 2][i] = smemB[load_stage_idx * 132 * 8 + B_lds_addr + (subk + 1) * 132 + i];
                B_frag[(subk + 1) % 2][i + 4] = smemB[load_stage_idx * 132 * 8 + B_lds_addr + (subk + 1) * 132 + i + 16];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                A_frag[(subk + 1) % 2][i] = smemA[load_stage_idx * 64 * 8 + A_lds_addr + (subk + 1) * 64 + i];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    output_frag[i][j] += B_frag[subk % 2][i] * A_frag[subk % 2][j];
                }
            }
        }
        // sts
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            smemB[write_stage_idx * 132 * 8 + B_sts_addr + i] = B_ldg_reg[i];
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            smemA[write_stage_idx * 64 * 8 + A_sts_addr + i * 32] = A_ldg_reg[i];
        }
        __syncthreads();
        write_stage_idx ^= 1;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            B_frag[0][i] = smemB[(load_stage_idx ^ 1) * 132 * 8 + B_lds_addr + 0 * 132 + i];
            B_frag[0][i + 4] = smemB[(load_stage_idx ^ 1) * 132 * 8 + B_lds_addr + 0 * 132 + i + 16];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            A_frag[0][i] = smemA[(load_stage_idx ^ 1) * 64 * 8 + A_lds_addr + 0 * 64 + i];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                output_frag[i][j] += B_frag[1][i] * A_frag[1][j];
            }
        }
    }

    int outOffset;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            outOffset = block_k * M * N + (y + i) * M + x + j;
            if (x + j < M && y + i < N)
            {
                C[outOffset] = output_frag[i][j];
            }
            outOffset = block_k * M * N + (y + i + 16) * M + x + j;
            if (x + j < M && y + i + 16 < N)
            {
                C[outOffset] = output_frag[i + 4][j];
            }
        }
    }
}

__device__ void sgemm_32x128(int M, int N, int K, float *A, float *B, float *C, int block_base_y, int block_base_x, int block_k, int kstride, float *smem)
{
    float *smemB = smem;
    float *smemA = smem + 16 * 256;

    int tx = threadIdx.x;

    // Warp tile
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    int B_lds_addr = warp_id * 16 + mma_tid_y * 4;
    int A_lds_addr = mma_tid_x * 4;

    int x = block_base_y + A_lds_addr;
    int y = block_base_x + B_lds_addr;

    // sts addr
    int B_sts_addr = (tx % 8) * 132 +
                     (tx / 8) * 4;
    int A_sts_addr = (tx / 32) * 32 + (tx % 32);

    // ldg addr
    int B_ldg_addr = (block_base_x + tx / 8 * 4) * K + tx % 8 + kstride * block_k;
    int A_ldg_addr = (tx / 32 + kstride * block_k) * M + block_base_y + tx % 32;
    float B_ldg_reg[4];
    float A_ldg_reg[1];
// ldg
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if ((tx % 8 + kstride * block_k) < K && (block_base_x + tx / 8 * 4 + i) < N)
            B_ldg_reg[i] = B[B_ldg_addr + i * K];
    }
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        if ((tx / 32 + kstride * block_k) < K && (block_base_y + tx % 32 + i * 32) < M)
            A_ldg_reg[i] = A[A_ldg_addr + i * 32];
    }

// sts
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        smemB[B_sts_addr + i] = B_ldg_reg[i];
    }
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        smemA[A_sts_addr + i * 32] = A_ldg_reg[i];
    }
    __syncthreads();
    int write_stage_idx = 1;
    float A_frag[2][4];
    float B_frag[2][4];
    float output_frag[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            output_frag[i][j] = 0;
        }
    }
    // lds
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        A_frag[0][i] = smemA[A_lds_addr + i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        B_frag[0][i] = smemB[B_lds_addr + i];
    }

    for (int k = 0; k < kstride; k += 8)
    {
        // ldg
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if ((tx % 8 + kstride * block_k + k + 8) < K && (block_base_x + tx / 8 * 4 + i) < N)
                B_ldg_reg[i] = B[B_ldg_addr + i * K + (k + 8)];
        }
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            if ((tx / 32 + kstride * block_k + k + 8) < K && (block_base_y + tx % 32 + i * 32) < M)
                A_ldg_reg[i] = A[A_ldg_addr + i * 32 + (k + 8) * M];
        }
        int load_stage_idx = write_stage_idx ^ 1;
#pragma unroll
        for (int subk = 0; subk < 8 - 1; ++subk)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                B_frag[(subk + 1) % 2][i] = smemB[load_stage_idx * 132 * 8 + B_lds_addr + (subk + 1) * 132 + i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                A_frag[(subk + 1) % 2][i] = smemA[load_stage_idx * 32 * 8 + A_lds_addr + (subk + 1) * 32 + i];
            }

#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    output_frag[i][j] += B_frag[subk % 2][i] * A_frag[subk % 2][j];
                }
            }
        }
        // sts
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            smemB[write_stage_idx * 132 * 8 + B_sts_addr + i] = B_ldg_reg[i];
        }
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            smemA[write_stage_idx * 32 * 8 + A_sts_addr + i * 32] = A_ldg_reg[i];
        }
        __syncthreads();
        write_stage_idx ^= 1;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            B_frag[0][i] = smemB[(load_stage_idx ^ 1) * 132 * 8 + B_lds_addr + 0 * 132 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            A_frag[0][i] = smemA[(load_stage_idx ^ 1) * 32 * 8 + A_lds_addr + 0 * 32 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                output_frag[i][j] += B_frag[1][i] * A_frag[1][j];
            }
        }
    }

    int outOffset;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            outOffset = block_k * M * N + (y + i) * M + x + j;
            if (x + j < M && y + i < N)
            {
                C[outOffset] = output_frag[i][j];
            }
        }
    }
}

__device__ void sgemm_64x64(int M, int N, int K, float *A, float *B, float *C, int block_base_y, int block_base_x, int block_k, int kstride, float *smem)
{
    float *smemB = smem;
    float *smemA = smem + 16 * 128;

    int tx = threadIdx.x;

    // Warp tile
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    int B_lds_addr = (warp_id / 2) * 16 + mma_tid_y * 4;
    int A_lds_addr = (warp_id % 2) * 32 + mma_tid_x * 4;

    int x = block_base_y + A_lds_addr;
    int y = block_base_x + B_lds_addr;

    // sts addr
    int B_sts_addr = (tx % 8) * 68 +
                     (tx / 8) * 2;
    int A_sts_addr = (tx / 32) * 64 + (tx % 32);

    // ldg addr
    int B_ldg_addr = (block_base_x + tx / 8 * 2) * K + tx % 8 + kstride * block_k;
    int A_ldg_addr = (tx / 32 + kstride * block_k) * M + block_base_y + tx % 32;
    float A_ldg_reg[2];
    float B_ldg_reg[2];
// ldg
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        if ((tx % 8 + kstride * block_k) < K && (block_base_x + tx / 8 * 2 + i) < N)
            B_ldg_reg[i] = B[B_ldg_addr + i * K];
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        if ((tx / 32 + kstride * block_k) < K && (block_base_y + tx % 32 + i * 32) < M)
            A_ldg_reg[i] = A[A_ldg_addr + i * 32];
    }

// sts
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        smemB[B_sts_addr + i] = B_ldg_reg[i];
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        smemA[A_sts_addr + i * 32] = A_ldg_reg[i];
    }
    __syncthreads();
    int write_stage_idx = 1;
    float A_frag[2][4];
    float B_frag[2][4];
    float output_frag[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            output_frag[i][j] = 0;
        }
    }
    // lds
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        A_frag[0][i] = smemA[A_lds_addr + i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        B_frag[0][i] = smemB[B_lds_addr + i];
    }

    for (int k = 0; k < kstride; k += 8)
    {
        // ldg
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            if ((tx % 8 + kstride * block_k + k + 8) < K && (block_base_x + tx / 8 * 2 + i) < N)
                B_ldg_reg[i] = B[B_ldg_addr + i * K + (k + 8)];
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            if ((tx / 32 + kstride * block_k + k + 8) < K && (block_base_y + tx % 32 + i * 32) < M)
                A_ldg_reg[i] = A[A_ldg_addr + i * 32 + (k + 8) * M];
        }
        int load_stage_idx = write_stage_idx ^ 1;
#pragma unroll
        for (int subk = 0; subk < 8 - 1; ++subk)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                B_frag[(subk + 1) % 2][i] = smemB[load_stage_idx * 68 * 8 + B_lds_addr + (subk + 1) * 68 + i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                A_frag[(subk + 1) % 2][i] = smemA[load_stage_idx * 64 * 8 + A_lds_addr + (subk + 1) * 64 + i];
            }

#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    output_frag[i][j] += B_frag[subk % 2][i] * A_frag[subk % 2][j];
                }
            }
        }
        // sts
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            smemB[write_stage_idx * 68 * 8 + B_sts_addr + i] = B_ldg_reg[i];
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            smemA[write_stage_idx * 64 * 8 + A_sts_addr + i * 32] = A_ldg_reg[i];
        }
        __syncthreads();
        write_stage_idx ^= 1;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            B_frag[0][i] = smemB[(load_stage_idx ^ 1) * 68 * 8 + B_lds_addr + 0 * 68 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            A_frag[0][i] = smemA[(load_stage_idx ^ 1) * 64 * 8 + A_lds_addr + 0 * 64 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                output_frag[i][j] += B_frag[1][i] * A_frag[1][j];
            }
        }
    }

    int outOffset;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            outOffset = block_k * M * N + (y + i) * M + x + j;
            if (x + j < M && y + i < N)
            {
                C[outOffset] = output_frag[i][j];
            }
        }
    }
}

__device__ void sgemm_32x32(int M, int N, int K, float *A, float *B, float *C, int block_base_y, int block_base_x, int block_k, int kstride, float *smem)
{
    float *smemB = smem;
    float *smemA = smem + 8 * 128;

    int tx = threadIdx.x;

    // Warp tile
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    int B_lds_addr = (warp_id / 2) * 8 + mma_tid_y * 2;
    int A_lds_addr = (warp_id % 2) * 16 + mma_tid_x * 2;

    int x = block_base_y + A_lds_addr;
    int y = block_base_x + B_lds_addr;

    // sts addr
    int B_sts_addr = (tx % 8) * 36 +
                     (tx / 8);
    int A_sts_addr = (tx / 32) * 32 + (tx % 32);

    // ldg addr
    int B_ldg_addr = (block_base_x + tx / 8) * K + tx % 8 + kstride * block_k;
    int A_ldg_addr = (tx / 32 + kstride * block_k) * M + block_base_y + tx % 32;
    float A_ldg_reg[1];
    float B_ldg_reg[1];
// ldg
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        if ((tx % 8 + kstride * block_k) < K && (block_base_x + tx / 8 + i) < N)
            B_ldg_reg[i] = B[B_ldg_addr + i * K];
    }
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        if ((tx / 32 + kstride * block_k) < K && (block_base_y + tx % 32 + i * 32) < M)
            A_ldg_reg[i] = A[A_ldg_addr + i * 32];
    }

// sts
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        smemB[B_sts_addr + i] = B_ldg_reg[i];
    }
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        smemA[A_sts_addr + i * 32] = A_ldg_reg[i];
    }
    __syncthreads();
    int write_stage_idx = 1;
    float A_frag[2][2];
    float B_frag[2][2];
    float output_frag[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
#pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            output_frag[i][j] = 0;
        }
    }
    // lds
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        A_frag[0][i] = smemA[A_lds_addr + i];
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        B_frag[0][i] = smemB[B_lds_addr + i];
    }

    for (int k = 0; k < kstride; k += 8)
    {
        // ldg
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            if ((tx % 8 + kstride * block_k + k + 8) < K && (block_base_x + tx / 8 + i) < N)
                B_ldg_reg[i] = B[B_ldg_addr + i * K + (k + 8)];
        }
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            if ((tx / 32 + kstride * block_k + k + 8) < K && (block_base_y + tx % 32 + i * 32) < M)
                A_ldg_reg[i] = A[A_ldg_addr + i * 32 + (k + 8) * M];
        }
        int load_stage_idx = write_stage_idx ^ 1;
#pragma unroll
        for (int subk = 0; subk < 8 - 1; ++subk)
        {
#pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                B_frag[(subk + 1) % 2][i] = smemB[load_stage_idx * 36 * 8 + B_lds_addr + (subk + 1) * 36 + i];
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                A_frag[(subk + 1) % 2][i] = smemA[load_stage_idx * 32 * 8 + A_lds_addr + (subk + 1) * 32 + i];
            }

#pragma unroll
            for (int i = 0; i < 2; ++i)
            {
#pragma unroll
                for (int j = 0; j < 2; ++j)
                {
                    output_frag[i][j] += B_frag[subk % 2][i] * A_frag[subk % 2][j];
                }
            }
        }
        // sts
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            smemB[write_stage_idx * 36 * 8 + B_sts_addr + i] = B_ldg_reg[i];
        }
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            smemA[write_stage_idx * 32 * 8 + A_sts_addr + i * 32] = A_ldg_reg[i];
        }
        __syncthreads();
        write_stage_idx ^= 1;
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            B_frag[0][i] = smemB[(load_stage_idx ^ 1) * 36 * 8 + B_lds_addr + 0 * 36 + i];
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            A_frag[0][i] = smemA[(load_stage_idx ^ 1) * 32 * 8 + A_lds_addr + 0 * 32 + i];
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
#pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                output_frag[i][j] += B_frag[1][i] * A_frag[1][j];
            }
        }
    }

    int outOffset;
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
#pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            outOffset = block_k * M * N + (y + i) * M + x + j;
            if (x + j < M && y + i < N)
            {
                C[outOffset] = output_frag[i][j];
            }
        }
    }
}

__device__ void sgemm_16x16(int M, int N, int K, float *A, float *B, float *C, int block_base_y, int block_base_x, int block_k, int kstride, float *smem)
{
    float *smemB = smem;
    float *smemA = smem + 8 * 128;

    int tx = threadIdx.x;

    // Warp tile
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    int B_lds_addr = (warp_id / 2) * 4 + mma_tid_y;
    int A_lds_addr = (warp_id % 2) * 8 + mma_tid_x;

    int x = block_base_y + A_lds_addr;
    int y = block_base_x + B_lds_addr;

    // sts addr
    int B_sts_addr = (tx % 16) * 20 +
                     (tx / 16);
    int A_sts_addr = (tx / 16) * 16 + (tx % 16);

    // ldg addr
    int B_ldg_addr = (block_base_x + tx / 16) * K + tx % 16 + kstride * block_k;
    int A_ldg_addr = (tx / 16 + kstride * block_k) * M + block_base_y + tx % 16;
    float A_ldg_reg[1];
    float B_ldg_reg[1];
// ldg
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        if ((tx % 16 + kstride * block_k) < K && (block_base_x + tx / 16 + i) < N)
            B_ldg_reg[i] = B[B_ldg_addr + i * K];
    }
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        if ((tx / 16 + kstride * block_k) < K && (block_base_y + tx % 16 + i * 16) < M)
            A_ldg_reg[i] = A[A_ldg_addr + i * 16];
    }

// sts
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        smemB[B_sts_addr + i] = B_ldg_reg[i];
    }
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        smemA[A_sts_addr + i * 16] = A_ldg_reg[i];
    }
    __syncthreads();
    int write_stage_idx = 1;
    float A_frag[2][1];
    float B_frag[2][1];
    float output_frag[1][1];
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
#pragma unroll
        for (int j = 0; j < 1; ++j)
        {
            output_frag[i][j] = 0;
        }
    }
    // lds
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        A_frag[0][i] = smemA[A_lds_addr + i];
    }
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
        B_frag[0][i] = smemB[B_lds_addr + i];
    }

    for (int k = 0; k < kstride; k += 16)
    {
        // ldg
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            if ((tx % 16 + kstride * block_k + k + 16) < K && (block_base_x + tx / 16 + i) < N)
                B_ldg_reg[i] = B[B_ldg_addr + i * K + (k + 16)];
        }
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            if ((tx / 16 + kstride * block_k + k + 16) < K && (block_base_y + tx % 16 + i * 16) < M)
                A_ldg_reg[i] = A[A_ldg_addr + i * 16 + (k + 16) * M];
        }
        int load_stage_idx = write_stage_idx ^ 1;
#pragma unroll
        for (int subk = 0; subk < 16 - 1; ++subk)
        {
#pragma unroll
            for (int i = 0; i < 1; ++i)
            {
                B_frag[(subk + 1) % 2][i] = smemB[load_stage_idx * 20 * 16 + B_lds_addr + (subk + 1) * 20 + i];
            }
#pragma unroll
            for (int i = 0; i < 1; ++i)
            {
                A_frag[(subk + 1) % 2][i] = smemA[load_stage_idx * 16 * 16 + A_lds_addr + (subk + 1) * 16 + i];
            }

#pragma unroll
            for (int i = 0; i < 1; ++i)
            {
#pragma unroll
                for (int j = 0; j < 1; ++j)
                {
                    output_frag[i][j] += B_frag[subk % 2][i] * A_frag[subk % 2][j];
                }
            }
        }
        // sts
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            smemB[write_stage_idx * 20 * 16 + B_sts_addr + i] = B_ldg_reg[i];
        }
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            smemA[write_stage_idx * 16 * 16 + A_sts_addr + i * 16] = A_ldg_reg[i];
        }
        __syncthreads();
        write_stage_idx ^= 1;
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            B_frag[0][i] = smemB[(load_stage_idx ^ 1) * 20 * 16 + B_lds_addr + 0 * 20 + i];
        }
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
            A_frag[0][i] = smemA[(load_stage_idx ^ 1) * 16 * 16 + A_lds_addr + 0 * 16 + i];
        }
#pragma unroll
        for (int i = 0; i < 1; ++i)
        {
#pragma unroll
            for (int j = 0; j < 1; ++j)
            {
                output_frag[i][j] += B_frag[1][i] * A_frag[1][j];
            }
        }
    }

    int outOffset;
#pragma unroll
    for (int i = 0; i < 1; ++i)
    {
#pragma unroll
        for (int j = 0; j < 1; ++j)
        {
            outOffset = block_k * M * N + (y + i) * M + x + j;
            if (x + j < M && y + i < N)
            {
                C[outOffset] = output_frag[i][j];
            }
        }
    }
}

__global__ void reduce_256(int M[], int N[], int *s_strategy, float *input[])
{
    int i = blockIdx.z;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= M[i] * N[i])
    {
        return;
    }
    for (int iter = 1; iter < s_strategy[i]; iter++)
    {
        input[i][x] += input[i][x + iter * M[i] * N[i]] + 100;
    }

    return;
}

__global__ void gemm_vbats(int M[], int N[], int K[], float *A[], float *B[], float *C[], int T_strategy[], int S_strategy[], int unified_n_tile[])
{

    extern __shared__ float sh[];

    int i = blockIdx.z;
    int t = T_strategy[i];
    int s = S_strategy[i];

    int by;
    int bx;
    int bk;

    switch (t)
    {
    case 0:
        by = (blockIdx.x % unified_n_tile[i]) * 16;
        bx = (blockIdx.x / unified_n_tile[i]) * 16;
        bk = blockIdx.y;
        if ((blockIdx.y + 1) * K[i] / s <= K[i] && (blockIdx.x % unified_n_tile[i]) * 16 < M[i] && (blockIdx.x / unified_n_tile[i]) * 16 < N[i])
            sgemm_16x16(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, bk, K[i] / s, sh);
        break;
    case 1:
        by = (blockIdx.x % unified_n_tile[i]) * 32;
        bx = (blockIdx.x / unified_n_tile[i]) * 32;
        bk = blockIdx.y;
        if ((blockIdx.y + 1) * K[i] / s <= K[i] && (blockIdx.x % unified_n_tile[i]) * 32 < M[i] && (blockIdx.x / unified_n_tile[i]) * 32 < N[i])
            sgemm_32x32(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, bk, K[i] / s, sh);
        break;
    case 2:
        by = (blockIdx.x % unified_n_tile[i]) * 128;
        bx = (blockIdx.x / unified_n_tile[i]) * 32;
        bk = blockIdx.y;
        if ((blockIdx.y + 1) * K[i] / s <= K[i] && (blockIdx.x % unified_n_tile[i]) * 128 < M[i] && (blockIdx.x / unified_n_tile[i]) * 32 < N[i])
            sgemm_128x32(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, bk, K[i] / s, sh);
        break;
    case 3:
        by = (blockIdx.x % unified_n_tile[i]) * 32;
        bx = (blockIdx.x / unified_n_tile[i]) * 128;
        bk = blockIdx.y;
        if ((blockIdx.y + 1) * K[i] / s <= K[i] && (blockIdx.x % unified_n_tile[i]) * 32 < M[i] && (blockIdx.x / unified_n_tile[i]) * 128 < N[i])
            sgemm_32x128(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, bk, K[i] / s, sh);
        break;
    case 4:
        by = (blockIdx.x % unified_n_tile[i]) * 64;
        bx = (blockIdx.x / unified_n_tile[i]) * 64;
        bk = blockIdx.y;
        if ((blockIdx.y + 1) * K[i] / s <= K[i] && (blockIdx.x % unified_n_tile[i]) * 64 < M[i] && (blockIdx.x / unified_n_tile[i]) * 64 < N[i])
            sgemm_64x64(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, bk, K[i] / s, sh);
        break;
    case 5:
        by = (blockIdx.x % unified_n_tile[i]) * 128;
        bx = (blockIdx.x / unified_n_tile[i]) * 64;
        bk = blockIdx.y;
        if ((blockIdx.y + 1) * K[i] / s <= K[i] && (blockIdx.x % unified_n_tile[i]) * 128 < M[i] && (blockIdx.x / unified_n_tile[i]) * 64 < N[i])
            sgemm_128x64(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, bk, K[i] / s, sh);
        break;
    case 6:
        by = (blockIdx.x % unified_n_tile[i]) * 64;
        bx = (blockIdx.x / unified_n_tile[i]) * 128;
        bk = blockIdx.y;
        if ((blockIdx.y + 1) * K[i] / s <= K[i] && (blockIdx.x % unified_n_tile[i]) * 64 < M[i] && (blockIdx.x / unified_n_tile[i]) * 128 < N[i])
            sgemm_64x128(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, bk, K[i] / s, sh);
        break;
    case 7:
        by = (blockIdx.x % unified_n_tile[i]) * 128;
        bx = (blockIdx.x / unified_n_tile[i]) * 128;
        bk = blockIdx.y;
        if ((blockIdx.y + 1) * K[i] / s <= K[i] && (blockIdx.x % unified_n_tile[i]) * 128 < M[i] && (blockIdx.x / unified_n_tile[i]) * 128 < N[i])
            sgemm_128x128(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, bk, K[i] / s, sh);
        break;
    }

    return;
}