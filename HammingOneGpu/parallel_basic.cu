#include "common.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <iostream>

__global__ void findHammingOne(int* data, const int N, const int L)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid + 1; i < N; ++i) {
		int distance = 0;
		for (int j = 0; j < L && distance <= 1; ++j) {
			int diff = data[tid * L + j] ^ data[i * L + j];
			if (diff) {
				if (diff & (diff - 1)) {
					distance = INT_MAX;
				}
				else {
					++distance;
				}
			}
		}
		if (distance == 1) {
			printf("(%d, %d)\n", tid, i);
		}
	}
}

void readFile(int*& data, int& N, int& L, const std::string fileName)
{
	constexpr int bits = CHAR_BIT * sizeof(*data);
	std::ifstream file(fileName);
	if (!file.is_open()) {
		fprintf(stderr, "ifstream failed!\n");
		exit(EXIT_FAILURE);
	}
	file >> N >> L;
	int newL = (int)ceil((double)L / bits);
	cudaCheckErrors(cudaMallocManaged((void**)&data, N * L * sizeof(*data)));
	memset(data, 0, N * newL * sizeof(*data));
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < newL; ++j) {
			for (int k = 0; k < bits && j * bits + k < L; ++k) {
				char b;
				file >> b;
				if (b == '0') {
					data[i * newL + j] = data[i * newL + j] << 1;
				}
				else if (b == '1') {
					data[i * newL + j] = (data[i * newL + j] << 1) + 1;
				}
			}
		}
	}
	L = newL;
}

int main()
{
	constexpr int device = 0;

	cudaCheckErrors(cudaSetDevice(device));
	cudaCheckErrors(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, ULONG_MAX));

	cudaEvent_t start{};
	cudaEvent_t end{};

	cudaCheckErrors(cudaEventCreate(&start));
	cudaCheckErrors(cudaEventCreate(&end));

	const std::string fileName = "hamming_one.txt";
	int N = 0;
	int L = 0;
	int* data = nullptr;

	readFile(data, N, L, fileName);

	cudaDeviceProp prop{};

	cudaCheckErrors(cudaGetDeviceProperties(&prop, device));

	int numThreads = prop.maxThreadsPerBlock;
	int numBlocks = (int)ceil((double)N / numThreads);

	cudaCheckErrors(cudaEventRecord(start));
	findHammingOne << <numBlocks, numThreads >> > (data, N, L);
	cudaCheckErrors(cudaGetLastError());
	cudaCheckErrors(cudaEventRecord(end));
	cudaCheckErrors(cudaEventSynchronize(end));

	float time = .0f;

	cudaCheckErrors(cudaEventElapsedTime(&time, start, end));

	printf("Searching for pairs with the Hamming distance equal to one took %f ms\n", time);

	cudaCheckErrors(cudaEventDestroy(start));
	cudaCheckErrors(cudaEventDestroy(end));
	cudaCheckErrors(cudaFree(data));
	cudaCheckErrors(cudaDeviceReset());
	return EXIT_SUCCESS;
}