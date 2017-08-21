#ifndef MRWSI_BIN_H
#define MRWSI_BIN_H

#include "linkedlist.h"
#include "mempool.h"
#include "resource.h"

struct item_t;

typedef struct bin_node_t {
    int time;
    list_node_t list;
    struct item_t* start_items;
    struct item_t* finish_items;
} bin_node_t;

typedef struct item_t {
    bin_node_t* start_node;
    bin_node_t* finish_node;
    list_node_t start_list;
    list_node_t finish_list;
} item_t;

#define size_of_node_t(dimension) \
    (sizeof(bin_node_t) + sizeof(res_t) * (dimension))
#define size_of_item_t(dimension) (sizeof(item_t) + sizeof(res_t) * (dimension))

#define _node_usage(x) ((res_t*)((x) + 1))
#define _item_demands(x) ((res_t*)((x) + 1))

#define _node_next(node) list_entry((node)->list.next, bin_node_t, list)
#define _node_prev(node) list_entry((node)->list.prev, bin_node_t, list)
#define _item_next_start(item) \
    list_entry((item)->start_list.next, item_t, start_list)
#define _item_next_finish(item) \
    list_entry((item)->finish_list.next, item_t, finish_list)

typedef struct bin_t {
    bin_node_t* head;
    int dimension;
    int num;

    mempool_t* _node_pool;
    mempool_t* _item_pool;
    bool _peak_need_update;
    bin_node_t* _last_start_node;
} bin_t;

#define size_of_bin_t(dimension) (sizeof(bin_t) + sizeof(res_t) * (dimension)*2)
#define _peak_usage(bin) ((res_t*)((bin) + 1))
#define _vol_tmp(bin) (_peak_usage(bin) + 1)

mempool_t* bin_create_node_pool(int dimension, size_t buffer_size);
mempool_t* bin_create_item_pool(int dimension, size_t buffer_size);
bin_t* bin_create(int dimension, mempool_t* node_pool, mempool_t* item_pool);
void bin_free(bin_t* bin);
void bin_print(bin_t* bin);

int bin_dimension(bin_t* bin);
bool bin_is_empty(bin_t* bin);
int bin_open_time(bin_t* bin);
int bin_close_time(bin_t* bin);
int bin_length(bin_t* bin);
void bin_to_array(bin_t* bin, int* sts, res_t* usages, int dim);
int bin_span(bin_t* bin);
res_t* bin_peak_usage(bin_t* bin, bool force);

int bin_current_block(bin_t* bin, int time, res_t* volumn);
int bin_earliest_slot(bin_t* bin, res_t* capacities, res_t* demands, int length,
                      int est, bin_node_t** start_node, bool only_forward);
int bin_earliest_slot_2(bin_t* bin_x, bin_t* bin_y, res_t* capacities_x,
                        res_t* capacities_y, res_t* demands_x, res_t* demands_y,
                        int length, int est, bin_node_t** start_node_x,
                        bin_node_t** start_node_y);
item_t* bin_alloc_item(bin_t* bin, int start_time, res_t* demands, int length,
                       bin_node_t* start_node);
void bin_free_item(bin_t* bin, item_t* item);

void bin_extendable_interval(bin_t* bin, item_t* item, res_t* capacities,
                             int* ei_begin, int* ei_end);

item_t* bin_extend_item(bin_t* bin, item_t* item, int st, int ft);

#endif
