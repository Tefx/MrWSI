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

typedef long res_t;
#define RES_MAX LONG_MAX
#define MRWSI_RES_UNIT_IS_INT

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

inline bool mr_ge(res_t* a, res_t* b, int dim) {
    for (int i = 0; i < dim; ++i)
        if (r_lt(a[i], b[i])) return false;
    return true;
}

inline bool mr_gt(res_t* a, res_t* b, int dim) {
    for (int i = 0; i < dim; ++i)
        if (r_le(a[i], b[i])) return false;
    return true;
}

inline bool mr_eq(res_t* a, res_t* b, int dim) {
    for (int i = 0; i < dim; ++i)
        if (!r_eq(a[i], b[i])) return false;
    return true;
}

inline bool mr_not_empty(res_t* a, int dim) {
    for (int i = 0; i < dim; ++i)
        if (r_gt(a[i], 0)) return true;
    return false;
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

bool mr_richcmp(res_t* r0, res_t* r1, int op, int dim);

#define mr_alloc(dim) ((res_t*)malloc(sizeof(res_t) * (dim)))
#define mr_free(r) (free(r))
#define mr_copy(dest, src, dim) memcpy((dest), (src), sizeof(res_t) * (dim))

#endif /* ifndef MRWSI_RESOURCE_H */
