#ifndef COMMON_H
#define COMMON_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define cudaCheckErrors(val) check((val), #val, __FILE__, __LINE__);

void check(const cudaError_t cudaStatus, const char* func, const char* file, const int line);

#endif