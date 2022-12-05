#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

// Rabin-Karp rolling hash
__global__ void computeHashes(thrust::pair<size_t, int>* const hashes, const uint_fast8_t* data, const int N, const int L, const size_t multiplier, const size_t modulus)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N) {
		size_t hash = 0;
		size_t a = multiplier;
		for (int i = 0; i < L; ++i) {
			hash = (hash + a * data[tid * L + i]) % modulus;
			a = a * multiplier % modulus;
		}
		thrust::pair<size_t, int> el{ hash, tid };
		hashes[tid] = el;
	}
}

__device__ int binarySearch(const thrust::pair<size_t, int>* const hashes, const size_t hash, const int N)
{
	int left = 0;
	int right = N - 1;
	while (left != right) {
		int middle = (int)ceil(((double)left + right) / 2);
		if (hashes[middle].first > hash) {
			right = middle - 1;
		}
		else {
			left = middle;
		}
	}
	if (hashes[left].first == hash) {
		return left;
	}
	return -1;
}

__global__ void findHammingOne(const thrust::pair<size_t, int>* const hashes, const uint_fast8_t* const data, const int N, const int L, const size_t multiplier, const size_t modulus)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N) {
		size_t a = multiplier;
		for (int i = 0; i < L; ++i) {
			thrust::pair<size_t, int> el = hashes[tid];
			int s = data[el.second * L + i] == 1 ? 1 : -1;
			size_t newHash = (el.first + s * a + modulus) % modulus;
			int j = binarySearch(hashes, newHash, N);
			while (j >= 0) {
				thrust::pair<size_t, int> newEl = hashes[j];
				if (newEl.first != newHash) {
					break;
				}
				if (el.second < newEl.second) {
					printf("(%d, %d)\n", el.second, newEl.second);
				}
				--j;
			}
			a = a * multiplier % modulus;
		}
	}
}

int readFile(uint_fast8_t*& data, int& N, int& L, const char* fileName)
{
	std::ifstream file(fileName);
	if (!file.is_open()) {
		fprintf(stderr, "ofstream failed!");
		return -1;
	}
	file >> N >> L;
	cudaError_t cudaStatus = cudaMallocManaged((void**)&data, N * L * sizeof(*data));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocManaged failed!\n");
		return -1;
	}
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < L; ++j) {
			char b;
			file >> b;
			if (b == '0') {
				data[i * L + j] = 2;
			}
			else if (b == '1') {
				data[i * L + j] = 1;
			}
		}
	}
	return 0;
}

int main()
{
	constexpr size_t multiplier = 16807;
	constexpr size_t modulus = 2147483647;

	constexpr const char* fileName = "hamming_one.txt";
	int N = 0;
	int L = 0;

	uint_fast8_t* data = nullptr;
	thrust::pair<size_t, int>* hashes = nullptr;
	int numThreads = 0;
	int numBlocks = 0;

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	if (readFile(data, N, L, fileName)) {
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&hashes, N * sizeof(*hashes));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocManaged failed!\n");
		goto Error;
	}

	numThreads = 1024;
	numBlocks = (int)ceil((double)N / numThreads);

	computeHashes << <numBlocks, numThreads >> > (hashes, data, N, L, multiplier, modulus);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "computeHashes launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	thrust::sort(thrust::device, hashes, hashes + N);

	findHammingOne << <numBlocks, numThreads >> > (hashes, data, N, L, multiplier, modulus);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching computeHashes!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		goto Error;
	}

Error:
	cudaFree(data);
	cudaFree(hashes);

	return 0;
}