#include "common.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <fstream>

#include <thrust/execution_policy.h>
#include <thrust/pair.h>
#include <thrust/sort.h>

constexpr uint_fast64_t firstMultiplier = 16807;
constexpr uint_fast64_t secondMultiplier = 8121;
constexpr uint_fast64_t firstModulus = 2147483647;
constexpr uint_fast64_t secondModulus = 2305843009213693951;

// Build a polynomial rolling hash
__global__ void computeHashes(thrust::pair<thrust::pair<uint_fast64_t, uint_fast64_t>, int>* const hashes, const char* d_data, const int N, const int L)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N) {
		uint_fast64_t firstHash = 0;
		uint_fast64_t secondHash = 0;
		uint_fast64_t firstCoefficient = firstMultiplier;
		uint_fast64_t secondCoefficient = 1;
		for (int i = 0; i < L; ++i) {
			firstHash = (firstHash + firstCoefficient * d_data[tid * L + i]) % firstModulus;
			secondHash = (secondHash + secondCoefficient * d_data[tid * L + i]) % secondModulus;
			firstCoefficient = firstCoefficient * firstMultiplier % firstModulus;
			secondCoefficient = secondCoefficient * secondMultiplier % secondModulus;
		}
		thrust::pair< thrust::pair<uint_fast64_t, uint_fast64_t>, int> el{ {firstHash, secondHash}, tid };
		// Allow multiple entries with the same key (multimap)
		hashes[tid] = el;
	}
}

// Return the index of the rightmost element if there are duplicates
__device__ int binarySearch(const thrust::pair<thrust::pair<uint_fast64_t, uint_fast64_t>, int>* const hashes, const thrust::pair<uint_fast64_t, uint_fast64_t> newHashPair, const int N)
{
	int left = 0;
	int right = N - 1;
	thrust::pair<uint_fast64_t, uint_fast64_t> hashPair{};
	while (left != right) {
		int middle = (int)ceil(((double)left + right) / 2);
		hashPair = hashes[middle].first;
		if (hashPair.first > newHashPair.first
			|| (hashPair.first == newHashPair.first && hashPair.second > newHashPair.second)) {
			right = middle - 1;
		}
		else {
			left = middle;
		}
	}
	hashPair = hashes[left].first;
	if (hashPair.first == newHashPair.first || hashPair.second == newHashPair.second) {
		return left;
	}
	return -1;
}

// Find pairs with the Hamming distance equal to one in O(N*L*log(N)) time
__global__ void findHammingOne(const thrust::pair<thrust::pair<uint_fast64_t, uint_fast64_t>, int>* const hashes, const char* const d_data, const int N, const int L)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N) {
		const thrust::pair<thrust::pair<uint_fast64_t, uint_fast64_t>, int> el = hashes[tid];
		uint_fast64_t firstCoefficient = firstMultiplier;
		uint_fast64_t secondCoefficient = 1;
		for (int i = 0; i < L; ++i) {
			// Update the hash in constant time
			int s = d_data[el.second * L + i] == 1 ? 1 : -1;
			thrust::pair<uint_fast64_t, uint_fast64_t> newHashPair{
				(el.first.first + s * firstCoefficient + firstModulus) % firstModulus,
				(el.first.second + s * secondCoefficient + secondModulus) % secondModulus
			};
			int j = binarySearch(hashes, newHashPair, N);
			while (j >= 0) {
				thrust::pair<thrust::pair<uint_fast64_t, uint_fast64_t>, int> newEl = hashes[j];
				if (newEl.first.first != newHashPair.first || newEl.first.second != newHashPair.second) {
					break;
				}
				// Print pairs only once
				if (el.second < newEl.second) {
					printf("(%d, %d)\n", el.second, newEl.second);
				}
				--j;
			}
			firstCoefficient = firstCoefficient * firstMultiplier % firstModulus;
			secondCoefficient = secondCoefficient * secondMultiplier % secondModulus;
		}
	}
}

// Read data from an existing well-formatted file
void readFile(char*& h_data, int& N, int& L, const std::string fileName)
{
	std::ifstream file(fileName);
	if (!file.is_open()) {
		fprintf(stderr, "ifstream failed!\n");
		exit(EXIT_FAILURE);
	}
	file >> N >> L;
	h_data = new char[N * L];
	if (!h_data) {
		fprintf(stderr, "new failed!\n");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < L; ++j) {
			char b;
			file >> b;
			if (b == '0') {
				// Replace 0's with 2's to get good hashing
				h_data[i * L + j] = 2;
			}
			else if (b == '1') {
				h_data[i * L + j] = 1;
			}
		}
	}
}

// Write time measurements to a unique file
void writeStats(const int N, const int L, const float readTime, const float memcpyTime, const float computeTime, const float sortTime, const float findTime)
{
	std::string now = std::to_string(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
	std::ofstream file("stats_" + now + ".txt");
	if (!file.is_open()) {
		fprintf(stderr, "ofstream failed!\n");
		exit(1);
	}
	file << "Reading " << N << " binary sequences of length " << L << ": " << readTime
		<< " ms\nCopying data from host to device memory: " << memcpyTime
		<< " ms\nComputing hashes: " << computeTime
		<< " ms\nSorting hashes: " << sortTime
		<< " ms\nSearching for pairs with the Hamming distance equal to one: " << findTime << " ms\n";
}

int main()
{
	constexpr int device = 0;

	cudaCheckErrors(cudaSetDevice(device));
	cudaCheckErrors(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, ULONG_MAX));

	cudaEvent_t readStart{};
	cudaEvent_t readEnd{};
	cudaEvent_t memcpyStart{};
	cudaEvent_t memcpyEnd{};
	cudaEvent_t computeStart{};
	cudaEvent_t computeEnd{};
	cudaEvent_t sortStart{};
	cudaEvent_t sortEnd{};
	cudaEvent_t findStart{};
	cudaEvent_t findEnd{};

	cudaCheckErrors(cudaEventCreate(&readStart));
	cudaCheckErrors(cudaEventCreate(&readEnd));
	cudaCheckErrors(cudaEventCreate(&memcpyStart));
	cudaCheckErrors(cudaEventCreate(&memcpyEnd));
	cudaCheckErrors(cudaEventCreate(&computeStart));
	cudaCheckErrors(cudaEventCreate(&computeEnd));
	cudaCheckErrors(cudaEventCreate(&sortStart));
	cudaCheckErrors(cudaEventCreate(&sortEnd));
	cudaCheckErrors(cudaEventCreate(&findStart));
	cudaCheckErrors(cudaEventCreate(&findEnd));

	const std::string fileName = "hamming_one.txt";
	int N = 0;
	int L = 0;
	char* h_data = nullptr;

	cudaCheckErrors(cudaEventRecord(readStart));
	readFile(h_data, N, L, fileName);
	cudaCheckErrors(cudaEventRecord(readEnd));

	char* d_data = nullptr;

	cudaCheckErrors(cudaMalloc((void**)&d_data, N * L * sizeof(*d_data)));

	cudaCheckErrors(cudaEventRecord(memcpyStart));
	cudaCheckErrors(cudaMemcpy(d_data, h_data, N * L * sizeof(*d_data), cudaMemcpyHostToDevice));
	cudaCheckErrors(cudaEventRecord(memcpyEnd));

	thrust::pair<thrust::pair<uint_fast64_t, uint_fast64_t>, int>* hashes = nullptr;

	cudaCheckErrors(cudaMalloc((void**)&hashes, N * sizeof(*hashes)));

	cudaDeviceProp prop{};

	cudaCheckErrors(cudaGetDeviceProperties(&prop, device));

	int numThreads = prop.maxThreadsPerBlock;
	int numBlocks = (int)ceil((double)N / numThreads);

	cudaCheckErrors(cudaEventRecord(computeStart));
	computeHashes << <numBlocks, numThreads >> > (hashes, d_data, N, L);
	cudaCheckErrors(cudaGetLastError());
	cudaCheckErrors(cudaEventRecord(computeEnd));

	cudaCheckErrors(cudaEventRecord(sortStart));
	// Sort in ascending order (the first element of a pair is compared first)
	thrust::sort(thrust::device, hashes, hashes + N);
	cudaCheckErrors(cudaEventRecord(sortEnd));

	cudaCheckErrors(cudaEventRecord(findStart));
	findHammingOne << <numBlocks, numThreads >> > (hashes, d_data, N, L);
	cudaCheckErrors(cudaGetLastError());
	cudaCheckErrors(cudaEventRecord(findEnd));
	cudaCheckErrors(cudaEventSynchronize(findEnd));

	float readTime = .0f;
	float memcpyTime = .0f;
	float computeTime = .0f;
	float sortTime = .0f;
	float findTime = .0f;

	cudaCheckErrors(cudaEventElapsedTime(&readTime, readStart, readEnd));
	cudaCheckErrors(cudaEventElapsedTime(&memcpyTime, memcpyStart, memcpyEnd));
	cudaCheckErrors(cudaEventElapsedTime(&computeTime, computeStart, computeEnd));
	cudaCheckErrors(cudaEventElapsedTime(&sortTime, sortStart, sortEnd));
	cudaCheckErrors(cudaEventElapsedTime(&findTime, findStart, findEnd));

	writeStats(N, L, readTime, memcpyTime, computeTime, sortTime, findTime);

	delete[] h_data;
	cudaCheckErrors(cudaEventDestroy(readStart));
	cudaCheckErrors(cudaEventDestroy(readEnd));
	cudaCheckErrors(cudaEventDestroy(memcpyStart));
	cudaCheckErrors(cudaEventDestroy(memcpyEnd));
	cudaCheckErrors(cudaEventDestroy(computeStart));
	cudaCheckErrors(cudaEventDestroy(computeEnd));
	cudaCheckErrors(cudaEventDestroy(sortStart));
	cudaCheckErrors(cudaEventDestroy(sortEnd));
	cudaCheckErrors(cudaEventDestroy(findStart));
	cudaCheckErrors(cudaEventDestroy(findEnd));
	cudaCheckErrors(cudaFree(d_data));
	cudaCheckErrors(cudaFree(hashes));
	cudaCheckErrors(cudaDeviceReset());
	return EXIT_SUCCESS;
}