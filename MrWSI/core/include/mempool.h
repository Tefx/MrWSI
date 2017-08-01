#ifndef MRWSI_MEMPOOL_H
#define MRWSI_MEMPOOL_H

/*#define MRWSI_MPOOL_USE_SYS_ALLOC*/
/*#define MRWSI_MPOOL_ALWAYS_LT_PAGESIZE*/

#include <stddef.h>

typedef struct block_ext_t {
    struct block_ext_t* next;
    int ref;
} block_ext_t;

typedef struct buffer_ext_t { struct buffer_ext_t* next; } buffer_ext_t;

typedef struct mempool_t {
    block_ext_t* free_list;
    buffer_ext_t* buffer_list;
    size_t ncr;  // Number of blocks available in current block
    size_t block_size;
    size_t block_num;
} mempool_t;

void mp_init_pool(mempool_t* pool, size_t block_size, size_t block_num);
void mp_destory_pool(mempool_t* pool);

mempool_t* mp_create_pool(size_t block_size, size_t buffer_size);
void mp_free_pool(mempool_t* pool);

void* mp_alloc(mempool_t* pool);
void mp_free(mempool_t* pool, void* ptr);
void mp_inc_ref(void* ptr);

#endif
