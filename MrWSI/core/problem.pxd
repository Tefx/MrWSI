from MrWSI.core.resource cimport res_t, mr_wrap_c, MultiRes
from libcpp cimport bool

cdef extern from "problem.h":
    struct task_info_t
    struct type_info_t
    struct problem_t:
        int num_tasks
        int num_types
        task_info_t* tasks
        type_info_t* types
        int charge_unit
        res_t* platform_limits
        int platform_limit_dim;

    void task_info_init(task_info_t* task, res_t* demands, int* prevs,
                        int num_prevs, int* succs, int num_succs, int* runtimes,
                        int num_types, long* data_sizes, int num_tasks)

    void type_info_init(type_info_t* type, res_t* capacities, res_t* demands,
                        int platform_limit_dim, double price);

    void problem_init(problem_t* problem, int num_tasks, int num_types,
                      int platform_limit_dim)
    void problem_set_charge_unit(problem_t* problem, int charge_unit)
    void problem_set_platform_limits(problem_t* problem, res_t* platform_limits,
                                     int platform_limit_dim)
    void problem_destory(problem_t* problem)
    void problem_reverse_dag(problem_t* problem)
    bool problem_task_is_entry(problem_t* problem, int task_id)
    bool problem_task_is_exit(problem_t* problem, int task_id)
    res_t* problem_task_demands(problem_t* problem, int task_id)
    int problem_task_runtime(problem_t* problem, int task_id, int type_id)
    int problem_task_num_prevs(problem_t* problem, int task_id)
    int problem_task_num_succs(problem_t* problem, int task_id)
    int* problem_task_prevs(problem_t* problem, int task_id)
    int* problem_task_succs(problem_t* problem, int task_id)

    res_t* problem_type_demands(problem_t* problem, int type_id)
    res_t* problem_type_capacities(problem_t* problem,int  type_id)
    double problem_type_price(problem_t* problem, int type_id)
    long problem_data_size(problem_t* problem, int task_from, int task_to)

    int problem_task_average_runtime(problem_t* problem, int task_id)
    int problem_cheapest_type(problem_t* problem)
    int problem_cheapest_type_for_demands(problem_t* problem, res_t* demands)
    double problem_charge(problem_t* problem, int type_id, int runtime)

cdef class Problem:
    cdef problem_t c
    cdef public list task_str_ids
    cdef public list type_str_ids
    cdef public list tasks
    cdef public list types
    cdef res_t mean_bandwidth

    cdef task_info_t* _ctask_info(self, int task_id)
    cdef type_info_t* _ctype_info(self, int type_id)

