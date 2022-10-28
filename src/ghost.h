#ifndef GHOST_H_INCLUDED
#define GHOST_H_INCLUDED

#include "util.h"

static inline void *ghost_alloc(size_t elem_size, int width, int height, int ghost_size, int starting_value)
{
    size_t array_size = elem_size * (width + ghost_size * 2) * (height * ghost_size * 2);
    u8 *p = ALLOCATE(u8, array_size);
    memset(p, starting_value, array_size);
    return (void *) (p + (width + ghost_size * 2 + ghost_size) * elem_size);
}

#define ALLOCATE_GHOST(type, width, height, ghost_size, starting_value) \
    (type *) alloc_ghost(sizeof(type), width, height, ghost_size, starting_value)

#define IGX(x, y, w, g) \
    ((y) * ((w) + (g) * 2) + (x))

static inline void ghost_free(void *p, int elem_size, int width, int ghost_size)
{
    u8 *realp = ((u8 *) p) - (width + ghost_size * 2 + ghost_size) * elem_size;
    free(realp);
}

static inline void *ghost_add(void *p, size_t elem_size, int width, int height, int ghost_size, int starting_value)
{
    u8 *np = (u8 *) ghost_alloc(elem_size, width, height, ghost_size, starting_value);
    u8 *op = p;
    for (int y = 0; y < height; y++)
        memcpy(np + y * (width + ghost_size * 2) * elem_size,
               op + y *  width                   * elem_size,
               width * elem_size);
    return np;
}

#define ADD_GHOST(type, p, width, height, ghost_size, starting_value) \
    (type *) ghost_add(p, sizeof(type), width, height, ghost_size, starting_value)

#endif
