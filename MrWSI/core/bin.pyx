from MrWSI.core.problem cimport Problem
from MrWSI.core.resource cimport mr_wrap_c, MultiRes
from cpython cimport array
import array

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
    def __init__(self, int dimension, MemPool node_pool, MemPool item_pool):
        self.c_ptr = bin_create(dimension, node_pool.c_ptr, item_pool.c_ptr)

    def __dealloc__(self):
        bin_free(self.c_ptr)

    def print_list(self):
        bin_print(self.c_ptr)

    def is_empty(self):
        return bin_is_empty(self.c_ptr)

    def open_time(self):
        return bin_open_time(self.c_ptr)

    def close_time(self):
        return bin_close_time(self.c_ptr)

    def span(self):
        return bin_span(self.c_ptr)

    def peak_usage(self, force=False):
        return mr_wrap_c(bin_peak_usage(self.c_ptr, force), bin_dimension(self.c_ptr))

    def current_block(self, int time, BinNode node=None):
        cdef MultiRes mr = MultiRes(bin_dimension(self.c_ptr))
        cdef BinNode bn = BinNode()
        cdef int length = bin_current_block(self.c_ptr, time, mr.c, node.c_ptr if node else NULL, &(bn.c_ptr))
        return mr, length, bn

    def earliest_slot(self, MultiRes capacities, MultiRes demands, int length,
                      int est, bool only_forward=False):
        cdef bin_node_t*start_node;
        cdef st = bin_earliest_slot(self.c_ptr, capacities.c, demands.c, length,
                                    est, &start_node, only_forward)
        return st, bn_wrap_c(start_node)

    def earliest_slot_2(self, Bin other,
                        MultiRes capacities, MultiRes capacities_other,
                        MultiRes demands_x, MultiRes demands_y, int length, int est):
        cdef bin_node_t* start_node_x
        cdef bin_node_t* start_node_y
        cdef st = bin_earliest_slot_2(self.c_ptr, other.c_ptr,
                                      capacities.c, capacities_other.c,
                                      demands_x.c, demands_y.c, length, est,
                                      &start_node_x, &start_node_y)
        return st, bn_wrap_c(start_node_x), bn_wrap_c(start_node_y)

    def alloc_item(self, int start_time, MultiRes demands, int length,
                   BinNode start_node=None):
        return item_wrap_c(bin_alloc_item(self.c_ptr, start_time, demands.c, length,
                                          start_node.c_ptr if start_node else NULL))

    def free_item(self, Item item):
        bin_free_item(self.c_ptr, item.c_ptr)

    def extendable_interval(self, Item item, MultiRes capacities):
        cdef int begin, end
        bin_extendable_interval(self.c_ptr, item.c_ptr, capacities.c, &begin, &end)
        return begin, end

    def extend_item(self, Item item, int st, int ft):
        return item_wrap_c(bin_extend_item(self.c_ptr, item.c_ptr, st, ft))

    def usages(self, dim_index):
        cdef int length = bin_length(self.c_ptr)
        cdef array.array sts = array.array("i", [])
        cdef array.array usages = array.array("l", [])
        array.resize(sts, length)
        array.resize(usages, length)
        bin_to_array(self.c_ptr, sts.data.as_ints, usages.data.as_longs, dim_index)
        return list(zip(sts, usages))
