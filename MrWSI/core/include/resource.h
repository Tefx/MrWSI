#ifndef MRWSI_RESOURCE_H
#define MRWSI_RESOURCE_H

#include <malloc.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define iMAX(x, y) (x) = MAX(x, y)
#define iMIN(x, y) (x) = MIN(x, y)

typedef int res_t;
#define MRWSI_RES_UNIT_IS_INT
#define RES_DIM_MAX 2

#ifdef MRWSI_RES_UNIT_IS_INT
#define EPSILON 0
#define r_eq(x, y) ((x) == (y))
#define r_ne(x, y) ((x) != (y))
#define r_le(x, y) ((x) <= (y))
#define r_lt(x, y) ((x) < (y))
#define r_ge(x, y) ((x) >= (y))
#define r_gt(x, y) ((x) > (y))
#else
#define EPSILON 0.001
#define r_eq(x, y) (fabs((x) - (y)) <= EPSILON)
#define r_ne(x, y) (fabs((x) - (y)) > EPSILON)
#define r_fle(x, y) ((x) <= ((y) + EPSILON))
#define r_flt(x, y) ((x) < ((y)-EPSILON))
#define r_fge(x, y) ((x) >= ((y)-EPSILON))
#define r_fgt(x, y) ((x) > ((y) + EPSILON))
#endif

inline bool mr_le(res_t* a, res_t* b, int dim) {
    for (int i = 0; i < dim; ++i)
        if (r_gt(a[i], b[i])) return false;
    return true;
}

inline bool mr_le_precise(res_t* a, res_t* b, int dim) {
    for (int i = 0; i < dim; ++i)
        if (a[i] > b[i]) return false;
    return true;
}

inline bool mr_lt(res_t* a, res_t* b, int dim) {
    for (int i = 0; i < dim; ++i)
        if (r_ge(a[i], b[i])) return false;
    return true;
}

inline bool mr_eq(res_t* a, res_t* b, int dim) {
    for (int i = 0; i < dim; ++i)
        if (!r_eq(a[i], b[i])) return false;
    return true;
}

#define mr_ineg(a, dim)                                        \
    {                                                          \
        for (int _i = 0; _i < (dim); ++_i) (a)[_i] = -(a)[_i]; \
    }

#define mr_iadd(a, b, dim)                                     \
    {                                                          \
        for (int _i = 0; _i < (dim); ++_i) (a)[_i] += (b)[_i]; \
    }

#define mr_isub(a, b, dim)                                     \
    {                                                          \
        for (int _i = 0; _i < (dim); ++_i) (a)[_i] -= (b)[_i]; \
    }

#define mr_sub(c, a, b, dim)                                            \
    {                                                                   \
        for (int _i = 0; _i < (dim); ++_i) (c)[_i] = (a)[_i] - (b)[_i]; \
    }

#define mr_iadd_v(a, v, dim)                               \
    {                                                      \
        for (int _i = 0; _i < (dim); ++_i) (a)[_i] += (v); \
    }
#define mr_imax(a, b, dim)                                         \
    {                                                              \
        for (int _i = 0; _i < (dim); ++_i) iMAX((a)[_i], (b)[_i]); \
    }

#define mr_set(a, v, dim)                             \
    {                                                 \
        for (int i = 0; i < (dim); ++i) (a)[i] = (v); \
    }

bool mr_richcmp(res_t* r0, res_t* r1, int op);

inline res_t* mr_alloc(int dim) { return (res_t*)malloc(sizeof(res_t) * dim); }

inline void mr_copy(res_t* dest, res_t* src, int dim) {
    memcpy(dest, src, sizeof(res_t) * dim);
}

#endif /* ifndef MRWSI_RESOURCE_H */
