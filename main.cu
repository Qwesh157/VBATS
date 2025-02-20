#include <cstdlib>
#include <cstdio>
#include <vector>
#include <fstream>
#include <cublas_v2.h>
#include "./include/util.h"
#include "./include/kernel.h"

// Switch to control
#define VERIFICATION 0        // Verification between VBATS and cuBLAS
#define DETAIL_PRINT 0        // VBATS framework detail print

#define MAX_K 1024
#define MAX_K_TILE 128
#define N_RUNS 10
#define SMEM_SIZE 24 * 1024

#define DIV_CEIL(x, y) (x + y - 1) / y

int main(int argc, char **argv)
{

	ErrChk(cudaSetDevice(0));

	if (argc < 2)
	{
		printf("Usage: input the group size\n");
		exit(EXIT_FAILURE);
	}
	int dev = 0, driverVersion = 0, runtimeVersion = 0;
    ErrChk(cudaSetDevice(dev));
    cudaDeviceProp deviceProp;
    ErrChk(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
	const int SM_NUM = deviceProp.multiProcessorCount;
    printf("Maximum miltiProcessor Count: %d\n", SM_NUM);

	int group_count = atoi(argv[1]);

	int *m_array;
	int *n_array;
	int *k_array;
	int *lda_array;
	int *ldb_array;
	int *ldc_array;
	int *group_size;

	m_array = (int *)malloc(group_count * sizeof(int));
	n_array = (int *)malloc(group_count * sizeof(int));
	k_array = (int *)malloc(group_count * sizeof(int));
	lda_array = (int *)malloc(group_count * sizeof(int));
	ldb_array = (int *)malloc(group_count * sizeof(int));
	ldc_array = (int *)malloc(group_count * sizeof(int));
	group_size = (int *)malloc(group_count * sizeof(int));

	std::fstream fs;
	fs.open("./data/input");
	if (!fs.is_open())
	{
		printf("Error opening input\n");
		exit(EXIT_FAILURE);
	}

	// read matrix config
	for (int i = 0; i < group_count; ++i)
	{
		fs >> m_array[i] >> n_array[i] >> k_array[i];
	}

	ErrChk(cudaDeviceSynchronize());

	float *alpha_array = (float *)malloc(group_count * sizeof(float));
	float *beta_array = (float *)malloc(group_count * sizeof(float));

	float **d_A_array = nullptr;
	float **d_B_array = nullptr;
	float **d_C_array = nullptr;
	float **h_A = (float **)malloc(group_count * sizeof(float *));
	float **h_B = (float **)malloc(group_count * sizeof(float *));
	float **h_C_cuBLAS = (float **)malloc(group_count * sizeof(float *));

	std::vector<float *> d_A(group_count, nullptr);
	std::vector<float *> d_B(group_count, nullptr);
	std::vector<float *> d_C(group_count, nullptr);

	// cublasSgemmGroupedBatched
	cublasOperation_t *transa_array = (cublasOperation_t *)malloc(group_count * sizeof(cublasOperation_t));
	cublasOperation_t *transb_array = (cublasOperation_t *)malloc(group_count * sizeof(cublasOperation_t));

	/* step 1: create cublas handle */
	cublasHandle_t cublasH = NULL;
	ErrChk(cublasCreate(&cublasH));

	/* step 2: copy data to device */
	for (int i = 0; i < group_count; ++i)
	{
		ErrChk(cudaMalloc(reinterpret_cast<void **>(&d_A[i]), m_array[i] * k_array[i] * sizeof(float)));
		ErrChk(cudaMalloc(reinterpret_cast<void **>(&d_B[i]), k_array[i] * n_array[i] * sizeof(float)));
		ErrChk(cudaMalloc(reinterpret_cast<void **>(&d_C[i]), m_array[i] * n_array[i] * MAX_K / MAX_K_TILE * sizeof(float)));
		h_A[i] = (float *)malloc(m_array[i] * k_array[i] * sizeof(float));
		h_B[i] = (float *)malloc(k_array[i] * n_array[i] * sizeof(float));
		gen_data(h_A[i], m_array[i] * k_array[i]);
		gen_data(h_B[i], k_array[i] * n_array[i]);
		ErrChk(cudaMemcpy(d_A[i], h_A[i], m_array[i] * k_array[i] * sizeof(float), cudaMemcpyHostToDevice));
		ErrChk(cudaMemcpy(d_B[i], h_B[i], k_array[i] * n_array[i] * sizeof(float), cudaMemcpyHostToDevice));
		h_C_cuBLAS[i] = (float *)malloc(m_array[i] * n_array[i] * sizeof(float));
		alpha_array[i] = 1.0;
		beta_array[i] = 0.0;
		transa_array[i] = CUBLAS_OP_N;
		transb_array[i] = CUBLAS_OP_N;
		lda_array[i] = m_array[i];
		ldb_array[i] = k_array[i];
		ldc_array[i] = m_array[i];
		group_size[i] = 1; // batch size for each group
	}

	ErrChk(
		cudaMalloc(reinterpret_cast<void **>(&d_A_array), sizeof(float *) * group_count));
	ErrChk(
		cudaMalloc(reinterpret_cast<void **>(&d_B_array), sizeof(float *) * group_count));
	ErrChk(
		cudaMalloc(reinterpret_cast<void **>(&d_C_array), sizeof(float *) * group_count));

	ErrChk(cudaMemcpy(d_A_array, d_A.data(), sizeof(float *) * group_count,
					  cudaMemcpyHostToDevice));
	ErrChk(cudaMemcpy(d_B_array, d_B.data(), sizeof(float *) * group_count,
					  cudaMemcpyHostToDevice));
	ErrChk(cudaMemcpy(d_C_array, d_C.data(), sizeof(float *) * group_count,
					  cudaMemcpyHostToDevice));

	float elapsedTime = 0.f;
	double arithmetic = 0.f;
	for (int i = 0; i < group_count; ++i)
		arithmetic += ((2 * int64_t(m_array[i]) * int64_t(n_array[i]) * int64_t(k_array[i])) + (2 * int64_t(m_array[i]) * int64_t(n_array[i]))) * 1.0;

	// warm-up and get results

	ErrChk(cublasSgemmGroupedBatched(
		cublasH, transa_array, transb_array, m_array, n_array, k_array,
		alpha_array, d_A_array, lda_array, d_B_array, ldb_array, beta_array,
		d_C_array, ldc_array, group_count, group_size));
	for (int i = 0; i < group_count; ++i)
		ErrChk(cudaMemcpy(h_C_cuBLAS[i], d_C[i], m_array[i] * n_array[i] * sizeof(float), cudaMemcpyDeviceToHost));
	ErrChk(cudaDeviceSynchronize());

	cudaEvent_t start, stop;
	ErrChk(cudaEventCreate(&start));
	ErrChk(cudaEventRecord(start, 0));
	for (int run = 0; run < N_RUNS; ++run)
	{
		cublasSgemmGroupedBatched(
			cublasH, transa_array, transb_array, m_array, n_array, k_array,
			alpha_array, d_A_array, lda_array, d_B_array, ldb_array, beta_array,
			d_C_array, ldc_array, group_count, group_size);
	}
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	double cuBLAS_time = elapsedTime / N_RUNS;
	cuBLAS_time /= 1.0e3; // convert time unit from millisecond to second
	float cuBLAS_gflops = (arithmetic / 1e9) / cuBLAS_time;

	// VBATS

	// Tiling strategy choice, order by thread compute intensity
	const int tile_strategy[8][2] = {
		16, 16,
		32, 32,
		128, 32,
		32, 128,
		64, 64,
		128, 64,
		64, 128,
		128, 128};

	// tiling strategy
	std::vector<int> t_strategy(group_count, 0);
	int *dev_t_strategy;
	ErrChk(cudaMalloc((void **)&dev_t_strategy, group_count * sizeof(int)));
	// splitting strategy
	std::vector<int> s_strategy(group_count, 1);
	int *dev_s_strategy;
	ErrChk(cudaMalloc((void **)&dev_s_strategy, group_count * sizeof(int)));

	float total_arithmetic = 0;
	float max_arithmetic = 0;
	for (int j = 0; j < group_count; ++j)
	{
		// total GEMM Operation Number Ratio
		total_arithmetic += m_array[j] * n_array[j] * k_array[j] * 2;
		max_arithmetic = m_array[j] * n_array[j] > max_arithmetic ? m_array[j] * n_array[j] : max_arithmetic;
	}

	// Tiling stage
	for (int j = 0; j < group_count; ++j)
	{
		// Step 1. Compute TBs threshold.
		float arithmetic_ratio = m_array[j] * n_array[j] * k_array[j] * 2 / total_arithmetic;
		float optBlockNum = arithmetic_ratio * SM_NUM;
		// Step 2. Traverse all tiling strategy.
		for (int i = 1; i < 8; ++i)
		{
			// Compute previous selected tiling strategy TBs and the ratio of active threads
			float preBlockNum_M = (DIV_CEIL(m_array[j], tile_strategy[t_strategy[j]][0]));
			float preBlockNum_N = (DIV_CEIL(n_array[j], tile_strategy[t_strategy[j]][1]));
			float preBlockNum = preBlockNum_M * preBlockNum_N;
			float preRatio_M = (m_array[j]) / (preBlockNum_M * tile_strategy[t_strategy[j]][0]);
			float preRatio_N = (n_array[j]) / (preBlockNum_N * tile_strategy[t_strategy[j]][1]);
			float preRatio = preRatio_M * preRatio_N;
			preBlockNum = preBlockNum >= optBlockNum ? preBlockNum / preRatio : preBlockNum * preRatio;
			// Compute current selected tiling strategy TBs and the ratio of active threads
			float curBlockNum_M = (DIV_CEIL(m_array[j], tile_strategy[i][0]));
			float curBlockNum_N = (DIV_CEIL(n_array[j], tile_strategy[i][1]));
			float curBlockNum = curBlockNum_M * curBlockNum_N;
			float curRatio_M = (m_array[j]) / (curBlockNum_M * tile_strategy[i][0]);
			float curRatio_N = (n_array[j]) / (curBlockNum_N * tile_strategy[i][1]);
			float curRatio = curRatio_M * curRatio_N;
			curBlockNum = curBlockNum >= optBlockNum ? curBlockNum / curRatio : curBlockNum * curRatio;

			// Step 3. Compute the TBs number and threshold.
			if (abs(curBlockNum - optBlockNum) < abs(preBlockNum - optBlockNum))
			{
				t_strategy[j] = i;
			}
		}
	}

	int maxsplitting = 1;
	// Splitting stage
	for (int j = 0; j < group_count; ++j)
	{
		float arithmetic_ratio = m_array[j] * n_array[j] * k_array[j] * 2 / total_arithmetic;
		float optBlockNum = arithmetic_ratio * SM_NUM;
		s_strategy[j] = 1;
		// Step 1. Determine whether K is suitable for splitting
		if (k_array[j] < MAX_K_TILE)
			continue;
		for (int k = 2; k < MAX_K / MAX_K_TILE && k_array[j] / k  >= MAX_K_TILE; k *= 2)
		{
			for (int i = t_strategy[j] + 1; i < 8; ++i)
			{
				// Compute previous selected tiling strategy and splitting strategy TBs and the ratio of active threads
				float preBlockNum_M = (DIV_CEIL(m_array[j], tile_strategy[t_strategy[j]][0]));
				float preBlockNum_N = (DIV_CEIL(n_array[j], tile_strategy[t_strategy[j]][1]));
				float preBlockNum = preBlockNum_M * preBlockNum_N * s_strategy[j];
				float preRatio_M = (m_array[j]) / (preBlockNum_M * tile_strategy[t_strategy[j]][0]);
				float preRatio_N = (n_array[j]) / (preBlockNum_N * tile_strategy[t_strategy[j]][1]);
				float preRatio = preRatio_M * preRatio_N;
				preBlockNum = preBlockNum >= optBlockNum ? preBlockNum / preRatio : preBlockNum * preRatio;
				// Compute current selected tiling strategy TBs and the ratio of active threads
				float curBlockNum_M = (DIV_CEIL(m_array[j], tile_strategy[i][0]));
				float curBlockNum_N = (DIV_CEIL(n_array[j], tile_strategy[i][1]));
				float curBlockNum = curBlockNum_M * curBlockNum_N * k;
				float curRatio_M = (m_array[j]) / (curBlockNum_M * tile_strategy[i][0]);
				float curRatio_N = (n_array[j]) / (curBlockNum_N * tile_strategy[i][1]);
				float curRatio = curRatio_M * curRatio_N;
				curBlockNum = curBlockNum >= optBlockNum ? curBlockNum / curRatio : curBlockNum * curRatio;
				// Step 3. Compute the TBs number and threshold.
				if (abs(preBlockNum - optBlockNum) >= abs(curBlockNum - optBlockNum))
				{
					t_strategy[j] = i;
					s_strategy[j] = k;
				}
			}
			maxsplitting = s_strategy[j] > maxsplitting ? s_strategy[j] : maxsplitting;
		}
	}
	ErrChk(cudaMemcpy(dev_t_strategy, t_strategy.data(), group_count * sizeof(int), cudaMemcpyHostToDevice));
	ErrChk(cudaMemcpy(dev_s_strategy, s_strategy.data(), group_count * sizeof(int), cudaMemcpyHostToDevice));

	// Unified 1D tile mapping
	std::vector<int> unified_n_tile(group_count, 0);
	int *dev_unified_n_tile;
	ErrChk(cudaMalloc((void **)&dev_unified_n_tile, group_count * sizeof(int)));
	int unified_tile = 0;

	for (int j = 0; j < group_count; ++j)
	{
		int curBlockNum = ((int)DIV_CEIL(m_array[j], tile_strategy[t_strategy[j]][0]) * (int)DIV_CEIL(n_array[j], tile_strategy[t_strategy[j]][1]));
		unified_n_tile[j] = (int)DIV_CEIL(m_array[j], tile_strategy[t_strategy[j]][0]);
		unified_tile = unified_tile > curBlockNum ? unified_tile : curBlockNum;
	}
	ErrChk(cudaMemcpy(dev_unified_n_tile, unified_n_tile.data(), group_count * sizeof(int), cudaMemcpyHostToDevice));

	dim3 block_size(256, 1, 1);
	dim3 grid_size;
	grid_size.x = unified_tile;
	grid_size.y = maxsplitting;
	grid_size.z = group_count;

	float **h_C_vbats = (float **)malloc(group_count * sizeof(float *));
	for (int i = 0; i < group_count; ++i){
		h_C_vbats[i] = (float *)malloc(m_array[i] * n_array[i] * sizeof(float));
	}

	int *dev_m_array, *dev_n_array, *dev_k_array;
	ErrChk(cudaMalloc((void **)&dev_m_array, group_count * sizeof(int)));
	ErrChk(cudaMalloc((void **)&dev_n_array, group_count * sizeof(int)));
	ErrChk(cudaMalloc((void **)&dev_k_array, group_count * sizeof(int)));

	ErrChk(cudaMemcpy(dev_m_array, m_array, group_count * sizeof(int), cudaMemcpyHostToDevice));
	ErrChk(cudaMemcpy(dev_n_array, n_array, group_count * sizeof(int), cudaMemcpyHostToDevice));
	ErrChk(cudaMemcpy(dev_k_array, k_array, group_count * sizeof(int), cudaMemcpyHostToDevice));
	
	dim3 grid_size_reduce;
	grid_size_reduce.x = (int)DIV_CEIL(max_arithmetic, 256);
	grid_size_reduce.y = 1;
	grid_size_reduce.z = group_count;
	// warm-up
	gemm_vbats<<<grid_size, block_size, SMEM_SIZE>>>(dev_m_array, dev_n_array, dev_k_array, d_A_array, d_B_array, d_C_array, dev_t_strategy, dev_s_strategy, dev_unified_n_tile);
	if (maxsplitting != 1)
	{
		reduce_256<<<grid_size_reduce, 256>>>(dev_m_array, dev_n_array, dev_s_strategy, d_C_array);
	}
	for (int i = 0; i < group_count; ++i)
		ErrChk(cudaMemcpy(h_C_vbats[i], d_C[i], m_array[i] * n_array[i] * sizeof(float), cudaMemcpyDeviceToHost));

	ErrChk(cudaEventCreate(&start));
	ErrChk(cudaEventRecord(start, 0));
	for (int run = 0; run < N_RUNS; ++run)
	{
		gemm_vbats<<<grid_size, block_size, SMEM_SIZE>>>(dev_m_array, dev_n_array, dev_k_array, d_A_array, d_B_array, d_C_array, dev_t_strategy, dev_s_strategy, dev_unified_n_tile);
		if (maxsplitting != 1)
		{
			reduce_256<<<grid_size_reduce, 256>>>(dev_m_array, dev_n_array, dev_s_strategy, d_C_array);
		}
	}
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	double vbats_time = elapsedTime / N_RUNS;
	vbats_time /= 1.0e3; // convert time unit from millisecond to second
	double vbats_gflops = (arithmetic / 1e9) / vbats_time;

#if (DETAIL_PRINT)
	for (int j = 0; j < group_count; ++j)
	{
		int m = m_array[j];
		int n = n_array[j];
		int k = k_array[j];
		float arithmetic_ratio = (float)(m_array[j] * n_array[j] * k_array[j] * 2) / (float)total_arithmetic;
		float optTBNum = arithmetic_ratio * SM_NUM;
		int curTBNum = (int)(DIV_CEIL(m_array[j], tile_strategy[t_strategy[j]][0])) * (int)(DIV_CEIL(n_array[j], tile_strategy[t_strategy[j]][1]));
		int tiling_strategy_m = tile_strategy[t_strategy[j]][0];
		int tiling_strategy_n = tile_strategy[t_strategy[j]][1];
		int splitting_strategy = s_strategy[j];
    	printf("Group:%3d,M:%4d,N:%4d,K:%4d arithmetic ratio:%8.2f%% optTBnum:%8.2f curTBnum:%5d tiling strategy: %3dx%d\tsplitting strategy:%5d\n",
			   j, m, n, k, arithmetic_ratio * 100.0f, optTBNum, curTBNum, tiling_strategy_m, tiling_strategy_n, splitting_strategy);
	}
#endif

	printf("cuBLAS: %8.2f Gflops vbats: %8.2f Gflops  Speedup: %8.2f\n", cuBLAS_gflops, vbats_gflops, vbats_gflops / cuBLAS_gflops);

#if (VERIFICATION)
	bool results = 1;
	for (int i = 0; i < group_count; ++i)
	{
		results &= CHECK(h_C_cuBLAS[i], h_C_vbats[i], m_array[i] * n_array[i]);
	}
	if(results){
    	printf("Verification passed!\n");
	}else{
    	printf("Verification failed!\n");
	}
#endif
	for (int i = 0; i < group_count; ++i)
	{
		free(h_A[i]);
		free(h_B[i]);
		ErrChk(cudaFree(d_A[i]));
		ErrChk(cudaFree(d_B[i]));
		ErrChk(cudaFree(d_C[i]));
	}

	free(m_array);
	free(n_array);
	free(k_array);
	free(lda_array);
	free(ldb_array);
	free(ldc_array);
	free(group_size);
	free(alpha_array);
	free(beta_array);
	free(h_A);
	free(h_B);
	free(h_C_cuBLAS);
	free(transa_array);
	free(transb_array);
	free(h_C_vbats);

	ErrChk(cudaFree(dev_t_strategy));
	ErrChk(cudaFree(dev_s_strategy));
	ErrChk(cudaFree(dev_unified_n_tile));

	return 0;
}
