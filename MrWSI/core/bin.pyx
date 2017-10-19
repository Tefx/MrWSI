from MrWSI.core.problem cimport Problem
from MrWSI.core.resource cimport mr_wrap_c, MultiRes
from cpython cimport array
import array
from libc.math cimport ceil

cdef class BinNode:
    @property
    def time(self):
        return node_time(self.c_ptr)

cdef class MemPool:
    def __cinit__(self, int dimension, int buffer_size, content="node"):
        if content == "node":
            self.c_ptr = bin_create_node_pool(dimension, buffer_size)
        else:
            self.c_ptr = bin_create_item_pool(dimension, buffer_size)

    def __dealloc__(self):
        mp_free_pool(self.c_ptr)

cdef bn_wrap_c(bin_node_t* node):
    cdef BinNode bn = BinNode()
    bn.c_ptr = node;
    return bn

cdef item_wrap_c(item_t* item):
    cdef Item it = Item()
    it.c_ptr = item;
    return it

cdef class Bin:
    def __init__(Bin self, int dimension, MemPool node_pool, MemPool item_pool):
        self.c_ptr = bin_create(dimension, node_pool.c_ptr, item_pool.c_ptr)

    def __dealloc__(Bin self):
        bin_free(self.c_ptr)

    def print_list(Bin self):
        bin_print(self.c_ptr)

    def is_empty(Bin self):
        return bin_is_empty(self.c_ptr)

    def open_time(Bin self):
        return bin_open_time(self.c_ptr)

    def close_time(Bin self):
        return bin_close_time(self.c_ptr)

    def span(Bin self):
        return bin_span(self.c_ptr)

    def peak_usage(Bin self, force=False):
        return mr_wrap_c(bin_peak_usage(self.c_ptr, force), bin_dimension(self.c_ptr))

    cpdef current_block(Bin self, int time, BinNode node):
        cdef MultiRes mr = MultiRes(bin_dimension(self.c_ptr))
        cdef BinNode bn = BinNode()
        cdef int length = bin_current_block(self.c_ptr, time, mr.c, node.c_ptr if node else NULL, &(bn.c_ptr))
        return mr, length, bn

    def find_idle_common_slots(Bin self, Bin other_bin, int st, long ds, int vi0, int vi1, int bandwidth):
        cdef long length, len_0, len_1
        cdef MultiRes vol_0, vol_1
        cdef BinNode bn_0 = None
        cdef BinNode bn_1 = None
        cdef list crs = []
        cdef long runtime = 0
        while ds > 0:
            vol_0, len_0, bn_0 = self.current_block(st, bn_0)
            vol_1, len_1, bn_1 = other_bin.current_block(st, bn_1)
            length = len_0 if len_0 < len_1 else len_1
            if not (vol_0[vi0] or vol_1[vi1]):
                ds -= length * bandwidth
                if ds < 0:
                    length += int(ceil(ds / bandwidth))
                crs.append((length, bandwidth))
                runtime += length
            elif crs:
                crs.append((length, 0))
                runtime += length
            st += length
        return st - runtime, st, crs


    def earliest_slot(Bin self, MultiRes capacities, MultiRes demands, int length,
                      int est, bool only_forward=False):
        cdef bin_node_t*start_node;
        cdef st = bin_earliest_slot(self.c_ptr, capacities.c, demands.c, length,
                                    est, &start_node, only_forward)
        return st, bn_wrap_c(start_node)

    def earliest_slot_2(Bin self, Bin other,
                        MultiRes capacities, MultiRes capacities_other,
                        MultiRes demands_x, MultiRes demands_y, int length, int est):
        cdef bin_node_t* start_node_x
        cdef bin_node_t* start_node_y
        cdef st = bin_earliest_slot_2(self.c_ptr, other.c_ptr,
                                      capacities.c, capacities_other.c,
                                      demands_x.c, demands_y.c, length, est,
                                      &start_node_x, &start_node_y)
        return st, bn_wrap_c(start_node_x), bn_wrap_c(start_node_y)

    def alloc_item(Bin self, int start_time, MultiRes demands, int length,
                   BinNode start_node=None):
        return item_wrap_c(bin_alloc_item(self.c_ptr, start_time, demands.c, length,
                                          start_node.c_ptr if start_node else NULL))

    def free_item(Bin self, Item item):
        bin_free_item(self.c_ptr, item.c_ptr)

    def extendable_interval(Bin self, Item item, MultiRes capacities):
        cdef int begin, end
        bin_extendable_interval(self.c_ptr, item.c_ptr, capacities.c, &begin, &end)
        return begin, end

    def extend_item(Bin self, Item item, int st, int ft):
        return item_wrap_c(bin_extend_item(self.c_ptr, item.c_ptr, st, ft))

    def usages(Bin self, int dim_index):
        cdef int length = bin_length(self.c_ptr)
        cdef array.array sts = array.array("i", [])
        cdef array.array usages = array.array("l", [])
        array.resize(sts, length)
        array.resize(usages, length)
        bin_to_array(self.c_ptr, sts.data.as_ints, usages.data.as_longs, dim_index)
        return list(zip(sts, usages))

    def eft_for_demand(Bin self, int demand, int di, int st, MultiRes capacities):
        return bin_finish_time_for_demands(self.c_ptr, capacities.c, demand, di, st)
