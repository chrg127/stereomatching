#include "util.h"

#define FILL_ARRAY_DEFINITION(type) \
__global__ void fill_array_##type(type *arr, size_t size, type value) \
{                                                                       \
    int i = threadIdx.x + blockIdx.x * blockDim.x;                      \
    if (i < size)                                                       \
        arr[i] = value;                                                 \
}

FILL_ARRAY_DEFINITION(double)
FILL_ARRAY_DEFINITION(i32)
FILL_ARRAY_DEFINITION(u8)

