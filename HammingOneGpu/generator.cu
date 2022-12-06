#include "common.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <iostream>

#include <curand_kernel.h>

__global__ void generateData(bool* data, const int N, const int L, const int K)
{
	const int size = N * L;
	const int part = size * K / 10;
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t state{};
	curand_init((unsigned long long)clock(), tid, 0, &state);
	if (tid < size - part) {
		const bool a = curand(&state) % 2;
		data[tid] = a;
		if (tid < part) {
			data[size - part + tid] = a;
		}
	}
}

__global__ void mutateData(bool* data, const int N, const int L, const int K)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t state{};
	curand_init((unsigned long long)clock(), tid, 0, &state);
	if (tid < N * K / 10) {
		int pos = curand(&state) % L;
		data[L * tid + pos] ^= 1;
	}
}

void writeData(const std::string fileName, const bool* a, const int N, const int L) {
	std::ofstream file(fileName);
	if (!file.is_open()) {
		fprintf(stderr, "ofstream failed!");
		exit(EXIT_FAILURE);
	}
	file << N << " " << L << "\n";
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < L; ++j) {
			file << a[i * L + j];
		}
		file << "\n";
	}
}

int main()
{
	constexpr int device = 0;

	// Choose which GPU to run on, change this on a multi-GPU system
	cudaCheckErrors(cudaSetDevice(device));

	constexpr int N = 100'000;
	constexpr int L = 1'000;

	bool* data = nullptr;

	// Allocate a GPU buffer for one vector
	cudaCheckErrors(cudaMallocManaged((void**)&data, N * L * sizeof(*data)));

	cudaEvent_t start{};
	cudaEvent_t end{};

	cudaCheckErrors(cudaEventCreate(&start));
	cudaCheckErrors(cudaEventCreate(&end));

	cudaDeviceProp prop{};

	cudaCheckErrors(cudaGetDeviceProperties(&prop, device));

	int numThreads = prop.maxThreadsPerBlock;
	int numBlocks = (int)ceil((double)N * L / numThreads);
	constexpr int K = 3;

	cudaCheckErrors(cudaEventRecord(start));

	// Launch a kernel on the GPU
	generateData << <numBlocks, numThreads >> > (data, N, L, K);

	// Check for any errors launching the kernel
	cudaCheckErrors(cudaGetLastError());

	// Launch a kernel on the GPU
	numBlocks = (int)ceil(N * K / 10.0 / numThreads);
	mutateData << <numBlocks, numThreads >> > (data, N, L, K);

	// Check for any errors launching the kernel
	cudaCheckErrors(cudaGetLastError());

	cudaCheckErrors(cudaEventRecord(end));
	cudaCheckErrors(cudaEventSynchronize(end));

	const std::string fileName = "hamming_one.txt";

	writeData(fileName, data, N, L);

	float time = .0f;

	cudaCheckErrors(cudaEventElapsedTime(&time, start, end));

	std::cout << "Generating " << N << " binary sequences of length " << L << " took " << time << " ms\n";

	cudaCheckErrors(cudaEventDestroy(start));
	cudaCheckErrors(cudaEventDestroy(end));

	cudaCheckErrors(cudaFree(data));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces
	cudaCheckErrors(cudaDeviceReset());

	return EXIT_SUCCESS;
}