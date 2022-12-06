#include "common.h"

#include <fstream>

void check(const cudaError_t cudaStatus, const char* func, const char* file, const int line) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", file, line, cudaStatus, cudaGetErrorString(cudaStatus), func);
		exit(EXIT_FAILURE);
	}
}