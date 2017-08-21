#include "../include/bin.h"
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

#define _alloc_node(bin) ((bin_node_t*)mp_alloc((bin)->_node_pool))
#define _alloc_item(bin) ((item_t*)mp_alloc((bin)->_item_pool))

static inline void _node_init(bin_node_t* node, int time, res_t* demands,
                              bin_t* bin) {
    node->time = time;
    memcpy(_node_usage(node), demands, sizeof(res_t) * bin->dimension);
    node->start_items = _alloc_item(bin);
    node->finish_items = _alloc_item(bin);
    list_init_head(&node->start_items->start_list);
    list_init_head(&node->finish_items->finish_list);
}

static inline void _node_destory(bin_t* bin, bin_node_t* node) {
    mp_free(bin->_item_pool, node->start_items);
    mp_free(bin->_item_pool, node->finish_items);
}

static inline bin_node_t* _clone_node(bin_t* bin, bin_node_t* prev_node,
                                      int time) {
    bin_node_t* node = _alloc_node(bin);
    _node_init(node, time, _node_usage(prev_node), bin);
    list_insert_after(&prev_node->list, &node->list);
    bin->num ++;
    return node;
}

static inline void _delete_node(bin_t* bin, bin_node_t* node) {
    list_delete(&(node)->list);
    bin->num --;
    _node_destory(bin, node);
    mp_free(bin->_node_pool, node);
}

void _node_print(bin_node_t* node, int dim) {
    printf("[%d, <", node->time);
    res_t* usage = _node_usage(node);
    for (int i = 0; i < dim; ++i) printf("%ld,", usage[i]);
    printf("\b>]");
}

item_t* _add_item(bin_t* bin, res_t* demands, bin_node_t* start_node,
                  bin_node_t* finish_node) {
    item_t* item = (item_t*)mp_alloc(bin->_item_pool);
    mr_copy(_item_demands(item), demands, bin->dimension);
    item->start_node = start_node;
    item->finish_node = finish_node;
    list_insert_after(&start_node->start_items->start_list, &item->start_list);
    list_insert_after(&finish_node->finish_items->finish_list,
                      &item->finish_list);
    return item;
}

void _delete_item(bin_t* bin, item_t* item) {
    item_t* tmp = item->start_node->start_items;
    while (tmp != item) tmp = _item_next_start(tmp);
    list_delete(&tmp->start_list);
    tmp = item->finish_node->finish_items;
    while (tmp != item) tmp = _item_next_finish(tmp);
    list_delete(&tmp->finish_list);
    mp_free(bin->_item_pool, item);
}

mempool_t* bin_create_node_pool(int dimension, size_t buffer_size) {
    return mp_create_pool(size_of_node_t(dimension), buffer_size);
}

mempool_t* bin_create_item_pool(int dimension, size_t buffer_size) {
    return mp_create_pool(size_of_item_t(dimension), buffer_size);
}

bin_t* bin_create(int dimension, mempool_t* node_pool, mempool_t* item_pool) {
    bin_t* bin = (bin_t*)malloc(size_of_bin_t(dimension));
    bin->_node_pool = node_pool;
    bin->_item_pool = item_pool;
    bin->num = 0;

    bin->head = _alloc_node(bin);
    bin->head->time = -1;
    mr_set(_node_usage(bin->head), 0, dimension);
    list_init_head(&bin->head->list);

    bin->dimension = dimension;
    mr_set(_peak_usage(bin), 0, dimension);
    bin->_peak_need_update = false;
    bin->_last_start_node = bin->head;
    return bin;
}

void bin_free(bin_t* bin) { free(bin); }

void bin_print(bin_t* bin) {
    bin_node_t* node = bin->head;
    int dim = bin->dimension;
    _node_print(node, dim);
    node = _node_next(node);
    while (node != bin->head) {
        printf("<=>");
        _node_print(node, dim);
        node = _node_next(node);
    }
    printf("\n");
}

int bin_dimension(bin_t* bin) { return bin->dimension; }

bool bin_is_empty(bin_t* bin) { return list_is_empty((bin)->head->list); }

int bin_open_time(bin_t* bin) {
    bin_node_t* node = _node_next((bin)->head);
    return node->time;
}

int bin_close_time(bin_t* bin) {
    bin_node_t* node = _node_prev((bin)->head);
    return node->time;
}

int bin_length(bin_t* bin) {return bin->num;}

void bin_to_array(bin_t* bin, int* sts, res_t* usages, int dim) {
    bin_node_t* node = _node_next(bin->head);
    int i = 0;
    while (node != bin->head){
        sts[i] = node->time;
        usages[i] = _node_usage(node)[dim];
        i++;
        node = _node_next(node);
    }
}

int bin_span(bin_t* bin) { return bin_close_time(bin) - bin_open_time(bin); }

res_t* bin_peak_usage(bin_t* bin, bool force) {
    /*bin->_peak_need_update = true;*/
    if (force || bin->_peak_need_update) {
        mr_set(_peak_usage(bin), 0, bin->dimension);
        bin_node_t* node = _node_next(bin->head);
        while (node != bin->head) {
            mr_imax(_peak_usage(bin), _node_usage(node), bin->dimension);
            node = _node_next(node);
        }
    }
    return _peak_usage(bin);
}

bin_node_t* _bin_search(bin_t* bin, int time) {
    bin_node_t* node = _node_prev(bin->head);
    while (node->time > time) node = _node_prev(node);
    return node;
}

int bin_current_block(bin_t* bin, int time, res_t* volumn) {
    bin_node_t* current_node = _bin_search(bin, time);
    bin_node_t* next_node = _node_next(current_node);
    int finish_time = next_node->time < 0 ? INT_MAX : next_node->time;
    mr_copy(volumn, _node_usage(current_node), bin->dimension);
    return finish_time - fmax(time, current_node->time);
}

bin_node_t* _earliest_slot(bin_node_t* node, res_t* vol, int length, int est,
                           int dim) {
    int ft = est + length;
    bin_node_t* start_node = node;
    mr_iadd_v(vol, EPSILON, dim);
    if (node->time < 0) node = _node_next(node);
    while (node->time >= 0 && node->time < ft) {
        if (!mr_le_precise(_node_usage(node), vol, dim)) {
            start_node = _node_next(node);
            ft = start_node->time + length;
        }
        node = _node_next(node);
    }
    return start_node;
}

int bin_earliest_slot(bin_t* bin, res_t* capacities, res_t* demands, int length,
                      int est, bin_node_t** start_node, bool only_forward) {
    res_t* vol = _vol_tmp(bin);
    mr_sub(vol, capacities, demands, bin->dimension);
    bin_node_t* node =
        only_forward ? bin->_last_start_node : _bin_search(bin, est);
    *start_node = _earliest_slot(node, vol, length, est, bin->dimension);
    return fmax((*start_node)->time, est);
}

int _earliest_slot_2(bin_node_t* node_x, bin_node_t* node_y, res_t* vol_x,
                     res_t* vol_y, int length, int est, int dimension,
                     bin_node_t** start_node_x, bin_node_t** start_node_y) {
    int ft = est + length;
    mr_iadd_v(vol_x, EPSILON, dimension);
    mr_iadd_v(vol_y, EPSILON, dimension);
    *start_node_x = node_x;
    *start_node_y = node_y;
    if (node_x->time < 0) node_x = _node_next(node_x);
    if (node_y->time < 0) node_y = _node_next(node_y);
    while ((node_x->time >= 0 && node_x->time < ft) ||
           (node_y->time >= 0 && node_y->time < ft)) {
        if (node_x->time < 0) {
            while (node_y->time >= 0 && node_y->time < ft) {
                if (!mr_le_precise(_node_usage(node_y), vol_y, dimension)) {
                    *start_node_y = _node_next(node_y);
                    ft = fmax(ft, (*start_node_y)->time + length);
                }
                node_y = _node_next(node_y);
            }
            break;
        }
        if (node_y->time < 0) {
            while (node_x->time >= 0 && node_x->time < ft) {
                if (!mr_le_precise(_node_usage(node_x), vol_x, dimension)) {
                    *start_node_x = _node_next(node_x);
                    ft = fmax(ft, (*start_node_x)->time + length);
                }
                node_x = _node_next(node_x);
            }
            break;
        }
        if (node_x->time <= node_y->time) {
            if (!mr_le_precise(_node_usage(node_x), vol_x, dimension)) {
                *start_node_x = _node_next(node_x);
                ft = fmax(ft, (*start_node_x)->time + length);
            }
            node_x = _node_next(node_x);
            while (node_y->time >= 0 && node_y->time < fmin(ft, node_x->time)) {
                if (!mr_le_precise(_node_usage(node_y), vol_y, dimension)) {
                    *start_node_y = _node_next(node_y);
                    ft = fmax(ft, (*start_node_y)->time + length);
                }
                node_y = _node_next(node_y);
            }
        } else {
            if (!mr_le_precise(_node_usage(node_y), vol_y, dimension)) {
                *start_node_y = _node_next(node_y);
                ft = fmax(ft, (*start_node_y)->time + length);
            }
            node_y = _node_next(node_y);
            while (node_x->time >= 0 && node_x->time < fmin(ft, node_y->time)) {
                if (!mr_le_precise(_node_usage(node_x), vol_x, dimension)) {
                    *start_node_x = _node_next(node_x);
                    ft = fmax(ft, (*start_node_x)->time + length);
                }
                node_x = _node_next(node_x);
            }
        }
    }
    return ft - length;
}

int bin_earliest_slot_2(bin_t* bin_x, bin_t* bin_y, res_t* capacities_x,
                        res_t* capacities_y, res_t* demands_x, res_t* demands_y,
                        int length, int est, bin_node_t** start_node_x,
                        bin_node_t** start_node_y) {
    bin_node_t* node_x = _bin_search(bin_x, est);
    bin_node_t* node_y = _bin_search(bin_y, est);
    res_t* vol_x = _vol_tmp(bin_x);
    res_t* vol_y = _vol_tmp(bin_y);
    mr_sub(vol_x, capacities_x, demands_x, bin_x->dimension);
    mr_sub(vol_y, capacities_y, demands_y, bin_y->dimension);
    return _earliest_slot_2(node_x, node_y, vol_x, vol_y, length, est,
                            bin_x->dimension, start_node_x, start_node_y);
}

#define _update_peak(bin, node, delta)                                      \
    {                                                                       \
        if ((delta)[0] < 0)                                                 \
            (bin)->_peak_need_update = true;                                \
        else if (!(bin)->_peak_need_update)                                 \
            mr_imax(_peak_usage(bin), _node_usage(node), (bin)->dimension); \
    }

item_t* bin_alloc_item(bin_t* bin, int start_time, res_t* demands, int length,
                       bin_node_t* start_node) {
    int finish_time = start_time + length;
    if (!start_node) start_node = _bin_search(bin, start_time);
    if (start_node->time < start_time)
        start_node = _clone_node(bin, start_node, start_time);
    bin_node_t* node = start_node;
    while (node != bin->head && node->time < finish_time) {
        mr_iadd(_node_usage(node), demands, bin->dimension);
        _update_peak(bin, node, demands);
        node = _node_next(node);
    }
    if (node == bin->head || node->time > finish_time) {
        node = _clone_node(bin, _node_prev(node), finish_time);
        mr_isub(_node_usage(node), demands, bin->dimension);
    }
    return _add_item(bin, demands, start_node, node);
}

#define no_item_at_node(node)                          \
    (list_is_empty((node)->start_items->start_list) && \
     list_is_empty((node)->finish_items->finish_list))

void bin_free_item(bin_t* bin, item_t* item) {
    bin_node_t* start_node = item->start_node;
    bin_node_t* finish_node = item->finish_node;
    bin_node_t* node = start_node;
    while (node != finish_node) {
        mr_isub(_node_usage(node), _item_demands(item), bin->dimension);
        node = _node_next(node);
    }
    bin->_peak_need_update = true;
    _delete_item(bin, item);
    if (no_item_at_node(start_node)) _delete_node(bin, start_node);
    if (no_item_at_node(finish_node)) _delete_node(bin, finish_node);
}

void bin_extendable_interval(bin_t* bin, item_t* item, res_t* capacities,
                             int* ei_begin, int* ei_end) {
    res_t* vol = _vol_tmp(bin);
    mr_sub(vol, capacities, _item_demands(item), bin->dimension);
    *ei_begin = item->start_node->time;
    bin_node_t* node = _node_prev(item->start_node);
    while (node->time >= 0 && mr_le(_node_usage(node), vol, bin->dimension)) {
        *ei_begin = node->time;
        node = _node_prev(node);
    }
    node = item->finish_node;
    *ei_end = node->time;
    while (mr_le(_node_usage(node), vol, bin->dimension)) {
        node = _node_next(node);
        if (node->time < 0) break;
        *ei_end = node->time;
    }
}

item_t* bin_extend_item(bin_t* bin, item_t* item, int st, int ft) {
    item_t* new_item =
        bin_alloc_item(bin, st, _item_demands(item), ft - st, NULL);
    bin_free_item(bin, item);
    return new_item;
}
