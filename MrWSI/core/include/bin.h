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

typedef struct bin_t {
    bin_node_t* head;
    int dimension;
    res_t peak_usage[RES_DIM_MAX];

    mempool_t* _node_pool;
    mempool_t* _item_pool;
    bool _peak_need_update;
    bin_node_t* _last_start_node;
} bin_t;

mempool_t* bin_prepare_pool(int dimension, size_t buffer_size);
void bin_init(bin_t* bin, int dimension, mempool_t* node_pool,
              mempool_t* item_pool);
void bin_print(bin_t* bin);

#define bin_is_empty(bin) (list_is_empty((bin)->head->list))
#define bin_open_time(bin) (_node_next((bin)->head)->time)
#define bin_close_time(bin) (_node_prev((bin)->head)->time)
#define bin_span(bin) (bin_close_time(bin) - bin_open_time(bin))

res_t* bin_peak_usage(bin_t* bin);
int bin_earliest_slot(bin_t* bin, res_t* capacities, res_t* demands, int length,
                      int est, bin_node_t** start_node, bool only_forward);
item_t* bin_alloc_item(bin_t* bin, int start_time, int length, res_t* demands,
                       bin_node_t* start_node);

void bin_extend_interval(bin_t* bin, item_t* item, res_t* capacities,
                         int* ei_begin, int* ei_end);

void bin_extend_item(bin_t* bin, item_t* item, int st, int ft);

#endif
