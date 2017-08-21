#include "../include/problem.h"
#include <float.h>

void task_info_init(task_info_t* task, res_t* demands, int* prevs,
                    int num_prevs, int* succs, int num_succs, int* runtimes,
                    int num_types, int* data_sizes, int num_tasks) {
    mr_copy(task->demands, demands, MULTIRES_DIM);
    task->num_prevs = num_prevs;
    task->num_succs = num_succs;
    task->prevs = (int*)malloc(sizeof(int) * num_prevs);
    task->succs = (int*)malloc(sizeof(int) * num_succs);
    task->runtimes = (int*)malloc(sizeof(int) * num_types);
    task->data_sizes = (int*)calloc(num_tasks, sizeof(int));
    memcpy(task->prevs, prevs, sizeof(int) * num_prevs);
    memcpy(task->succs, succs, sizeof(int) * num_succs);
    memcpy(task->runtimes, runtimes, sizeof(int) * num_types);
    memcpy(task->data_sizes, data_sizes, sizeof(int) * num_tasks);
}

void task_info_destory(task_info_t* task) {
    free(task->prevs);
    free(task->succs);
    free(task->runtimes);
    free(task->data_sizes);
}

void type_info_init(type_info_t* type, res_t* capacities, res_t* demands,
                    int platform_limit_dim, double price) {
    type->demands = mr_alloc(platform_limit_dim);
    mr_copy(type->capacities, capacities, MULTIRES_DIM);
    mr_copy(type->demands, demands, platform_limit_dim);
    type->price = price;
}

void type_info_destory(type_info_t* type) { free(type->demands); }

void problem_init(problem_t* problem, int num_tasks, int num_types,
                  int platform_limit_dim) {
    problem->num_tasks = num_tasks;
    problem->num_types = num_types;
    problem->tasks = (task_info_t*)malloc(sizeof(task_info_t) * num_tasks);
    problem->types = (type_info_t*)malloc(sizeof(type_info_t) * num_types);
    problem->platform_limits = mr_alloc(platform_limit_dim);
    problem->platform_limit_dim = platform_limit_dim;
    problem->charge_unit = 1;
    mr_set(problem->platform_limits, RES_MAX, platform_limit_dim);
}

void problem_set_charge_unit(problem_t* problem, int charge_unit) {
    problem->charge_unit = charge_unit;
}

void problem_set_platform_limits(problem_t* problem, res_t* platform_limits,
                                 int platform_limit_dim) {
    mr_copy(problem->platform_limits, platform_limits, platform_limit_dim);
}

void problem_destory(problem_t* problem) {
    for (int i = 0; i < problem->num_tasks; ++i)
        task_info_destory(problem->tasks + i);
    for (int i = 0; i < problem->num_types; ++i)
        type_info_destory(problem->types + i);
    free(problem->tasks);
    free(problem->types);
    free(problem->platform_limits);
}

void problem_reverse_dag(problem_t* problem) {
    int tmp_num;
    int* tmp_dep;
    task_info_t* task;
    for (int task_id = 0; task_id < problem->num_tasks; ++task_id) {
        task = problem->tasks + task_id;
        tmp_num = task->num_succs;
        task->num_succs = task->num_prevs;
        task->num_prevs = tmp_num;
        tmp_dep = task->prevs;
        task->prevs = task->succs;
        task->succs = tmp_dep;
    }
}

int problem_task_average_runtime(problem_t* problem, int task_id) {
    int sum = 0;
    for (int i = 0; i < problem->num_types; ++i)
        sum += problem->tasks[task_id].runtimes[i];
    return sum / problem->num_types;
}

int problem_cheapest_type(problem_t* problem) {
    double min_price = DBL_MAX;
    int cheapest_type = -1;
    for (int type_id = 0; type_id < problem->num_types; ++type_id)
        if (problem_type_price(problem, type_id) < min_price)
            cheapest_type = type_id;
    return cheapest_type;
}

int problem_cheapest_type_for_demands(problem_t* problem, res_t* demands) {
    double min_price = DBL_MAX;
    int cheapest_type = -1;
    for (int type_id = 0; type_id < problem->num_types; ++type_id) {
        if (mr_le(demands, problem_type_capacities(problem, type_id),
                  MULTIRES_DIM) &&
            problem_type_price(problem, type_id) <= min_price) {
            min_price = problem_type_price(problem, type_id);
            cheapest_type = type_id;
        }
    }
    return cheapest_type;
}

#define _div_and_ceil(x, y) (((x) + (y)-1) / (y))

double problem_charge(problem_t* problem, int type_id, int runtime) {
    return problem_type_price(problem, type_id) * problem->charge_unit *
           _div_and_ceil(runtime, problem->charge_unit);
}
