#include "../include/mempool.h"
#include <malloc.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef MRWSI_MPOOL_ALWAYS_LT_PAGESIZE
static size_t page_size_avail = 0;
#else
static int is_initialised = 0;
#endif

#define block_of(ptr) ((block_ext_t*)(ptr)-1)

static inline block_ext_t* _block_at(buffer_ext_t* buffer, size_t index,
                                     size_t block_size) {
    return (block_ext_t*)((char*)(buffer + 1) + index * block_size);
}

static inline void _enlarge(mempool_t* pool) {
    size_t size = pool->block_num * pool->block_size + sizeof(buffer_ext_t);
    buffer_ext_t* buffer = (buffer_ext_t*)malloc(size);
    buffer->next = pool->buffer_list;
    pool->buffer_list = buffer;
    pool->ncr = pool->block_num;
}

void mp_init_pool(mempool_t* pool, size_t block_size, size_t block_num) {
    pool->free_list = NULL;
    pool->buffer_list = NULL;
    pool->block_size = block_size + sizeof(block_ext_t);
    pool->ncr = 0;

#ifdef MRWSI_MPOOL_ALWAYS_LT_PAGESIZE
    if (!page_size_avail)
        page_size_avail = (sysconf(_SC_PAGESIZE) - sizeof(buffer_ext_t));
    size_t max_block_num = page_size_avail / pool->block_size - 16;
    pool->block_num = block_num > max_block_num ? max_block_num : block_size;
#else
    if (!is_initialised) {
        mallopt(M_TRIM_THRESHOLD, -1);
        mallopt(M_MMAP_MAX, 0);
        is_initialised = 1;
    }
    pool->block_num = block_num;
#endif
}

mempool_t* mp_create_pool(size_t block_size, size_t block_num) {
    mempool_t* pool = (mempool_t*)malloc(sizeof(mempool_t));
    mp_init_pool(pool, block_size, block_num);
    return pool;
}

void mp_destory_pool(mempool_t* pool) {
    buffer_ext_t* buffer = pool->buffer_list;
    buffer_ext_t* tmp;
    while (buffer) {
        tmp = buffer;
        buffer = buffer->next;
        free(tmp);
    }
}

void mp_free_pool(mempool_t* pool) {
    mp_destory_pool(pool);
    free(pool);
}

void mp_inc_ref(void* ptr) { block_of(ptr)->ref++; }

#ifdef MRWSI_MPOOL_USE_SYS_ALLOC

void* mp_alloc(mempool_t* pool) {
    block_ext_t* node = (block_ext_t*)malloc(pool->block_size);
    node->ref = 1;
    return node + 1;
}

void mp_free(mempool_t* pool, void* ptr) {
    block_ext_t* block = block_of(ptr);
    if (!(--block->ref)) free(block);
}

#else

void* mp_alloc(mempool_t* pool) {
    block_ext_t* block;
    if (pool->free_list) {
        block = pool->free_list;
        pool->free_list = block->next;
    } else {
        if (!pool->ncr) _enlarge(pool);
        block = _block_at(pool->buffer_list, --pool->ncr, pool->block_size);
    }
    block->ref = 1;
    return (void*)(block + 1);
}

void mp_free(mempool_t* pool, void* ptr) {
    block_ext_t* block = block_of(ptr);
    if (!(--block->ref)) {
        block->next = pool->free_list;
        pool->free_list = block;
    }
}

#endif
