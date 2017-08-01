//
// Created by zhaomengz on 26/4/17.
//

#ifndef MRWSI_LINKEDLIST_H
#define MRWSI_LINKEDLIST_H

#include <stddef.h>

typedef struct list_node_t {
    struct list_node_t* prev;
    struct list_node_t* next;
} list_node_t;

#define container_of(ptr, type, member) \
    (type*)((char*)(ptr)-offsetof(type, member))

#define list_conn(p, n) \
    (p)->next = (n);    \
    (n)->prev = (p)

#define list_insert_after(p, node) \
    list_conn(node, (p)->next);    \
    list_conn(p, node)

#define list_insert_before(p, node) \
    list_conn((p)->prev, node);     \
    list_conn(node, p)

#define list_delete(node) list_conn((node)->prev, (node)->next)
#define list_init_head(node) (node)->prev = (node)->next = node
#define list_entry(node, type, member) container_of((node), type, member)

#define list_is_empty(list) (list).next == &(list)

#endif  // MRWSI_LINKEDLIST_H
