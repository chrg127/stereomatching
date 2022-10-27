#ifndef GHOST_H_INCLUDED
#define GHOST_H_INCLUDED

#include "util.h"

static inline void *alloc_ghost(size_t size, int width, int height, int ghost_size, int starting_value)
{
    void *p = xmalloc(size * (width + ghost_size * 2) * (height * ghost_size * 2));
    memset(p, starting_value, size * (width + ghost_size * 2) * (height * ghost_size * 2));
    return p + (width + ghost_size * 2 + ghost_size);
}

#define ALLOCATE_GHOST_AREA(type, width, height, ghost_size, starting_value) \
    (type *) alloc_ghost(sizeof(type), width, height, ghost_size, starting_value)

static inline void free_ghost(void *p, int width, int ghost_size)
{
    void *realp = p - (width + ghost_size * 2 + ghost_size);
    free(realp);
}

#endif
