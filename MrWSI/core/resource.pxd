from libcpp cimport bool

DEF MULTIRES_DIM=3

cdef extern from "resource.h":
    ctypedef long res_t

    bool r_eq(res_t* a, res_t* b)
    bool r_ne(res_t* a, res_t* b)
    bool r_le(res_t* a, res_t* b)
    bool r_lt(res_t* a, res_t* b)
    bool r_ge(res_t* a, res_t* b)
    bool r_gt(res_t* a, res_t* b)

    bool mr_le(res_t* a, res_t* b, int dim)
    bool mr_le_precise(res_t* a, res_t* b, int dim)
    bool mr_lt(res_t* a, res_t* b, int dim)
    bool mr_eq(res_t* a, res_t* b, int dim)

    void mr_ineg(res_t* a, int dim)
    void mr_iadd(res_t* a, res_t* b, int dim)
    void mr_isub(res_t* a, res_t* b, int dim)
    void mr_sub(res_t* c, res_t* a, res_t* b, int dim)
    void mr_iadd_v(res_t* a, int v, int dim)
    void mr_imax(res_t* a, res_t* b, int dim)
    void mr_set(res_t* a, int v, int dim)

    bool mr_richcmp(res_t* r0, res_t* r1, int op, int dim)

    res_t* mr_alloc(int dim);
    void mr_copy(res_t* dest, res_t* src, int dim)

cdef class MultiRes:
    cdef res_t* c
    cdef int dimension

cdef mr_wrap_c(res_t* c, int dimension)
