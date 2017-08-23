from libc.string cimport memcpy

cdef class MultiRes:
    def __cinit__(self, dimension):
        self.c = mr_alloc(dimension)
        self.dimension = dimension

    def __dealloc__(self):
        mr_free(self.c)

    @classmethod
    def zero(cls, dimension):
        mr = MultiRes(dimension)
        mr_set(mr.c, 0, dimension)
        return mr

    def __iadd__(MultiRes self, MultiRes other):
        mr_iadd(self.c, other.c, self.dimension)
        return self

    def __isub__(MultiRes self, MultiRes other):
        mr_isub(self.c, other.c, self.dimension)
        return self

    def __add__(MultiRes self, MultiRes other):
        mr = mr_wrap_c(self.c, self.dimension)
        mr += other
        return mr

    def __sub__(MultiRes self, MultiRes other):
        mr = mr_wrap_c(self.c, self.dimension)
        mr -= other
        return mr

    def imax(MultiRes self, MultiRes other):
        mr_imax(self.c, other.c, self.dimension)

    @classmethod
    def max(cls, MultiRes mr_0, MultiRes mr_1):
        result = mr_0.__copy__()
        result.imax(mr_1)
        return result

    def __copy__(MultiRes self):
        return mr_wrap_c(self.c, self.dimension)

    def __richcmp__(MultiRes self, MultiRes other, int op):
        return mr_richcmp(self.c, other.c, op, self.dimension)

    def __getitem__(self, int index):
        return self.c[index]

    def __setitem__(self, int index, res_t value):
        self.c[index] = value

    def __repr__(self):
        return str([self.c[i] for i in range(self.dimension)])

cdef mr_wrap_c(res_t* c, int dimension):
    mr = MultiRes(dimension)
    mr_copy(mr.c, c, dimension)
    return mr

