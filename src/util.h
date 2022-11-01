#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>
#ifdef __NVCC__
#include "helper_cuda.h"
#endif

typedef uint8_t  u8;
typedef uint32_t u32;
typedef int8_t   i8;
typedef int32_t  i32;

#define IDX(x, y, w) ((y)*(w)+(x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(x, mi, ma) (MIN(MAX(x, mi), ma))
#define SWAP(x, y, T)   \
    do {                \
        T tmp = (x);    \
        (x) = (y);      \
        (y) = tmp;      \
    } while (0)

#ifdef __NVCC__
__host__ __device__
#endif
static inline int idx(int x, int y, int w)
{
    x = (x + w) % w;
    y = (y + w) % w;
    return y * w + x;
}

static inline void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (!p) {
        fprintf(stderr, "error: out of memory\n");
        exit(1);
    }
    memset(p, 0, size);
    return p;
}

#define ALLOCATE(type, count) \
    (type *) xmalloc(sizeof(type) * (count))

static inline double parse_double(const char *s, double *n)
{
    char *endptr;
    *n = strtod(s, &endptr);
    return *n == 0 && endptr == s;
}

static inline int parse_int(const char *s, int *n)
{
    char *endptr;
    *n = strtol(s, &endptr, 0);
    return *n == 0 && endptr == s;
}

/*
 * performs an integer division, rounding to the higher integer instead of the lower one.
 * e.g. 1/2 = 1, 5/3 = 2
 */
#ifdef __NVCC__
__host__ __device__
#endif
static inline int ceil_div(int x, int y)
{
    return (x + y - 1) / y;
}

// CUDA specific helper functions

#ifdef __NVCC__

static inline void *cuda_xmalloc(size_t size)
{
    void *p;
    checkCudaErrors(cudaMalloc(&p, size));
    if (!p) {
        fprintf(stderr, "error: out of memory\n");
        exit(1);
    }
    checkCudaErrors(cudaMemset(p, 0, size));
    return p;
}

static inline void *make_gpu_copy(void *hp, size_t size)
{
    void *dp = cuda_xmalloc(size);
    checkCudaErrors(cudaMemcpy(dp, hp, size, cudaMemcpyHostToDevice));
    return dp;
}

static inline void *make_host_copy(void *hp, size_t size)
{
    void *dp = cuda_xmalloc(size);
    checkCudaErrors(cudaMemcpy(dp, hp, size, cudaMemcpyDeviceToHost));
    return dp;
}

#define ALLOCATE_GPU(type, count) \
    (type *) cuda_xmalloc(sizeof(type) * count)

#define MAKE_GPU_COPY(type, p, count) \
    (type *) make_gpu_copy(p, sizeof(type) * count)

#define MAKE_HOST_COPY(type, p, count) \
    (type *) make_host_copy(p, sizeof(type) * count)

#endif

#endif
