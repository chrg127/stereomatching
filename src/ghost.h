#ifndef GHOST_H_INCLUDED
#define GHOST_H_INCLUDED

#include "util.h"

static inline void *ghost_to_real(void *p, int elem_size, int width, int ghost_size)
{
    return ((u8 *) p) - IDX(ghost_size, ghost_size, width + ghost_size * 2) * elem_size;
}

static inline void *real_to_ghost(void *p, int elem_size, int width, int ghost_size)
{
    return ((u8 *) p) + IDX(ghost_size, ghost_size, width + ghost_size * 2) * elem_size;
}

#define GHOST_ALLOC_DEFINITION(type) \
static inline type *ghost_alloc_##type(int width, int height, int ghost_size, type start_value) \
{                                                                                               \
    size_t array_size = (width + ghost_size * 2) * (height + ghost_size * 2);                   \
    type *p = ALLOCATE(type, array_size);                                                       \
    for (size_t i = 0; i < array_size; i++)                                                     \
        p[i] = start_value;                                                                     \
    return (type *) real_to_ghost(p, sizeof(type), width, ghost_size);                          \
}                                                                                               \

static inline void ghost_free(void *p, int elem_size, int width, int ghost_size)
{
    free(ghost_to_real(p, elem_size, width, ghost_size));
}

#define GHOST_FREE(type, p, width, ghost_size) \
    ghost_free(p, sizeof(type), width, ghost_size)

#define GHOST_ADD_DEFINITION(type) \
static inline type *ghost_add_##type(type *p, int width, int height, int ghost_size, type start_value)  \
{                                                                                                       \
    type *np = ghost_alloc_##type(width, height, ghost_size, start_value);                              \
    type *op = p;                                                                                       \
    for (int y = 0; y < height; y++)                                                                    \
        memcpy(np + y * (width + ghost_size * 2),                                                       \
               op + y *  width,                                                                         \
               width * sizeof(type));                                                                   \
    return np;                                                                                          \
}

GHOST_ALLOC_DEFINITION(double)
GHOST_ALLOC_DEFINITION(i32)
GHOST_ALLOC_DEFINITION(u8)

GHOST_ADD_DEFINITION(double)
GHOST_ADD_DEFINITION(i32)
GHOST_ADD_DEFINITION(u8)

#define IGX(x, y, w, g) \
    ((y) * ((w) + (g) * 2) + (x))



#ifdef __NVCC__

#define GHOST_ALLOC_GPU_DEFINITION(type) \
static inline type *ghost_alloc_gpu_##type(int width, int height, int ghost_size, type start_value) \
{                                                                                                   \
    size_t array_size = (width + ghost_size * 2) * (height + ghost_size * 2);                       \
    type *p = ALLOCATE_GPU(type, array_size);                                                       \
    fill_array_##type<<<ceil_div(array_size, BLOCK_DIM), BLOCK_DIM>>>(p, array_size, start_value);  \
    return (type *) real_to_ghost(p, sizeof(type), width, ghost_size);                              \
}                                                                                                   \

static inline void ghost_free_gpu(void *p, int elem_size, int width, int ghost_size)
{
    cudaFree(ghost_to_real(p, elem_size, width, ghost_size));
}

#define GHOST_FREE_GPU(type, p, width, ghost_size) \
    ghost_free_gpu(p, sizeof(type), width, ghost_size)

#define GHOST_ADD_GPU_DEFINITION(type) \
static inline type *ghost_add_gpu_##type(type *p, int width, int height, int ghost_size,    \
                                         type start_value, enum cudaMemcpyKind kind)        \
{                                                                                           \
    type *np = ghost_alloc_gpu_##type(width, height, ghost_size, start_value);              \
    type *op = p;                                                                           \
    for (int y = 0; y < height; y++)                                                        \
        cudaMemcpy(np + y * (width + ghost_size * 2),                                       \
                   op + y *  width,                                                         \
                   width * sizeof(type),                                                    \
                   kind);                                                                   \
    return np;                                                                              \
}

GHOST_ALLOC_GPU_DEFINITION(double)
GHOST_ALLOC_GPU_DEFINITION(i32)
GHOST_ALLOC_GPU_DEFINITION(u8)

GHOST_ADD_GPU_DEFINITION(double)
GHOST_ADD_GPU_DEFINITION(i32)
GHOST_ADD_GPU_DEFINITION(u8)

#endif // __NVCC__

#endif
