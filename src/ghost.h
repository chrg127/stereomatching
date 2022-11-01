#ifndef GHOST_H_INCLUDED
#define GHOST_H_INCLUDED

#include "util.h"

#define GHOST_ALLOC_DEFINITION(type) \
static inline type *ghost_alloc_##type(int width, int height, int ghost_size, type start_value) \
{                                                                                               \
    size_t array_size = (width + ghost_size * 2) * (height + ghost_size * 2);                   \
    type *p = ALLOCATE(type, array_size);                                                       \
    for (size_t i = 0; i < array_size; i++)                                                     \
        p[i] = start_value;                                                                     \
    return p + (width + ghost_size * 2 + ghost_size);                                           \
}                                                                                               \

GHOST_ALLOC_DEFINITION(double)
GHOST_ALLOC_DEFINITION(i32)
GHOST_ALLOC_DEFINITION(u8)

static inline void ghost_free(void *p, int elem_size, int width, int ghost_size)
{
    u8 *realp = ((u8 *) p) - (width + ghost_size * 2 + ghost_size) * elem_size;
    free(realp);
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

GHOST_ADD_DEFINITION(double)
GHOST_ADD_DEFINITION(i32)
GHOST_ADD_DEFINITION(u8)

#define IGX(x, y, w, g) \
    ((y) * ((w) + (g) * 2) + (x))

#endif
