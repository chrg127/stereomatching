#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdbool.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
#ifdef __NVCC__
#include "helper_cuda.h"
#endif

typedef uint8_t  u8;
typedef uint32_t u32;
typedef int8_t   i8;
typedef int32_t  i32;

#define ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))
#define IDX(x, y, w) ((y)*(w)+(x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(x, mi, ma) (MIN(MAX(x, mi), ma))
#define SWAP(T, x, y)   \
    do {                \
        T tmp = (x);    \
        (x) = (y);      \
        (y) = tmp;      \
    } while (0)

#ifdef __NVCC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

HOST DEVICE static inline int idx(int x, int y, int w, int h)
{
    x = (x + w) % w;
    y = (y + h) % h;
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
HOST DEVICE static inline int ceil_div(int x, int y)
{
    return (x + y - 1) / y;
}

HOST DEVICE static inline i32 array_max(i32 *a, size_t s)
{
    i32 m = INT_MIN;
    for (size_t i = 0; i < s; i++)
        m = MAX(a[i], m);
    return m;
}

HOST DEVICE static inline i32 array_min(i32 *a, size_t s)
{
    i32 m = INT_MAX;
    for (size_t i = 0; i < s; i++)
        m = MIN(a[i], m);
    return m;
}

static inline double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + (double) ts.tv_nsec / 1e9;
}



// CUDA specific helper functions

#ifdef __NVCC__

#define BLOCK_DIM 1024
#define BLOCK_DIM_SIDE 32
#define BLOCK_DIM_2D dim3(BLOCK_DIM_SIDE, BLOCK_DIM_SIDE)

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

static inline void *make_host_copy(void *dp, size_t size)
{
    void *hp = xmalloc(size);
    checkCudaErrors(cudaMemcpy(hp, dp, size, cudaMemcpyDeviceToHost));
    return hp;
}

#define ALLOCATE_GPU(type, count) \
    (type *) cuda_xmalloc(sizeof(type) * count)

#define MAKE_GPU_COPY(type, p, count) \
    (type *) make_gpu_copy(p, sizeof(type) * count)

#define MAKE_HOST_COPY(type, p, count) \
    (type *) make_host_copy(p, sizeof(type) * count)

__global__ void fill_array_double(double *arr, size_t size, double value);
__global__ void fill_array_i32(i32 *arr, size_t size, i32 value);
__global__ void fill_array_u8(u8 *arr, size_t size, u8 value);

i32 array_max_gpu(i32 *arr, size_t n);
i32 array_min_gpu(i32 *arr, size_t n);

#endif // __NVCC__

#endif
