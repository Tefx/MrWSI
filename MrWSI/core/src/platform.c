#include "../include/platform.h"
#include "memory.h"

platform_env_t* platform_create_nev(int num_tasks, int num_res_types,
                                    int vm_lim) {
    platform_env_t* env = (platform_env_t*)malloc(sizeof(platform_create_nev));
    env->num_res_types = num_res_types;
    env->machine_node_pool = bin_create_node_pool(DEMAND_DIM, num_tasks);
    env->machine_item_pool = bin_create_item_pool(DEMAND_DIM, num_tasks);
    env->platform_node_pool = bin_create_node_pool(num_res_types, vm_lim);
    env->platform_item_pool = bin_create_item_pool(num_res_types, vm_lim);
    return env;
}

void platform_free_env(platform_env_t* env) {
    mp_destory_pool(env->machine_node_pool);
    mp_destory_pool(env->machine_item_pool);
    mp_destory_pool(env->platform_node_pool);
    mp_destory_pool(env->platform_item_pool);
}

void machine_init(machine_t* machine, platform_env_t* env){
    bin_init(&machine->bin, DEMAND_DIM, env->machine_node_pool, env->machine_item_pool);
    machine->item = NULL;
    machine->platfrom = NULL;
}

void platform_init(platform_t* platform, platform_env_t* env){
    bin_init(&platform->bin, env->num_res_types, env->platform_node_pool, env->platform_item_pool);
}


