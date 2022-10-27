#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdbool.h>

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

#endif
