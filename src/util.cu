#include "util.h"

#define FILL_ARRAY_DEFINITION(type) \
__global__ void fill_array_##type(type *arr, size_t size, type value)   \
{                                                                       \
    int i = threadIdx.x + blockIdx.x * blockDim.x;                      \
    if (i < size)                                                       \
        arr[i] = value;                                                 \
}

FILL_ARRAY_DEFINITION(double)
FILL_ARRAY_DEFINITION(i32)
FILL_ARRAY_DEFINITION(u8)

#define REDUCTION_OP(name, func_name, atomic_func_name, start_value) \
__global__ void _##name(i32 *a, size_t n, i32 *r)               \
{                                                               \
    __shared__ i32 tmp[BLOCK_DIM];                              \
    int li = threadIdx.x,                                       \
        gi = threadIdx.x + blockIdx.x * blockDim.x,             \
        block_size = blockDim.x / 2;                            \
    tmp[li] = gi < n ? a[gi] : start_value;                     \
    __syncthreads();                                            \
    while (block_size > 0) {                                    \
        if (li < block_size)                                    \
            tmp[li] = func_name(tmp[li], tmp[li + block_size]); \
        block_size /= 2;                                        \
        __syncthreads();                                        \
    }                                                           \
    if (li == 0)                                                \
        atomic_func_name(r, tmp[0]);                            \
}                                                               \
                                                                \
i32 name(i32 *a, size_t n)                                      \
{                                                               \
    i32 r = start_value;                                        \
    i32 *rd = MAKE_GPU_COPY(i32, &r, 1);                        \
    _##name<<<ceil_div(n, BLOCK_DIM), BLOCK_DIM>>>(a, n, rd);   \
    checkCudaErrors(cudaMemcpy(&r, rd, sizeof(r), cudaMemcpyDeviceToHost)); \
    cudaFree(rd);                                               \
    return r;                                                   \
}

REDUCTION_OP(array_max_gpu, MAX, atomicMax, INT_MIN)
REDUCTION_OP(array_min_gpu, MIN, atomicMin, INT_MAX)

