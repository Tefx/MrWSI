from libc.string cimport memcpy

DEF MULTIRES_DIM=3

cdef class MultiRes:
    @classmethod
    def zero(cls):
        mr = MultiRes()
        mr_set(mr.c, 0, MULTIRES_DIM)
        return mr

    def __iadd__(MultiRes self, MultiRes other):
        mr_iadd(self.c, other.c, MULTIRES_DIM)
        return self

    def __isub__(MultiRes self, MultiRes other):
        mr_isub(self.c, other.c, MULTIRES_DIM)
        return self

    def __add__(MultiRes self, MultiRes other):
        mr = mr_wrap_c(self.c)
        mr += other
        return mr

    def imax(MultiRes self, MultiRes other):
        mr_imax(self.c, other.c, MULTIRES_DIM)

    @classmethod
    def max(cls, mrs):
        result = MultiRes.zero()
        for mr in mrs:
            result.imax(mr)
        return result

    def __copy(MultiRes self):
        return mr_wrap_c(self.c)

    def __richcmp__(MultiRes self, MultiRes other, int op):
        return mr_richcmp(self.c, other.c, op, MULTIRES_DIM)

    def __getitem__(self, int index):
        return self.c[index]

    def __repr__(self):
        return str(self.c)

cdef mr_wrap_c(res_t* c):
    mr = MultiRes()
    memcpy(mr.c, c, sizeof(res_t) * MULTIRES_DIM)
    return mr

