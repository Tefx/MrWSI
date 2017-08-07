from cpython cimport array
import array
import json

DEF MULTIRES_DIM = 3

def calculate_succs(tasks):
    succs = {task_id:[] for task_id in tasks}
    for task_id, task in tasks.items():
        for prev_id in task["prevs"].keys():
            succs[prev_id].append(task_id)
    return succs

cdef class Problem:
    def __cinit__(self, dict tasks, dict types, int platform_limit_dim):
        cdef task_info_t* task_info
        cdef type_info_t* type_info
        cdef array.array demands, prevs, succs, runtimes, data_sizes
        cdef array.array capacities, type_demands

        self.task_str_ids = list(tasks.keys())
        self.type_str_ids = list(types.keys())
        raw_succs = calculate_succs(tasks)

        problem_init(&self.c, len(tasks), len(types), platform_limit_dim)

        for task_id, task_str_id in enumerate(self.task_str_ids):
            task_info = self._ctask_info(task_id)
            raw_task = tasks[task_str_id]
            raw_task["demands"][0] *= 1000
            demands = array.array("l", map(int, raw_task["demands"]))
            prevs = array.array("i", [self.task_str_ids.index(t) for t in raw_task["prevs"].keys()])
            succs = array.array("i", [self.task_str_ids.index(t) for t in raw_succs[task_str_id]])
            runtimes = array.array("i", [int(raw_task["runtime"]/types[p]["speed"]) for p in self.type_str_ids])
            data_sizes = array.array("i", [0 for _ in range(len(tasks))])
            for prev_id, data in raw_task["prevs"].items():
                data_sizes[self.task_str_ids.index(prev_id)] = data
            task_info_init(task_info, demands.data.as_longs, 
                           prevs.data.as_ints, len(prevs),
                           succs.data.as_ints, len(succs),
                           runtimes.data.as_ints, len(types),
                           data_sizes.data.as_ints, len(tasks))

        for type_id, type_str_id in enumerate(self.type_str_ids):
            type_info = self._ctype_info(type_id)
            raw_type = types[type_str_id]
            raw_type["capacities"][0] *= 1000
            capacities = array.array("l", raw_type["capacities"])
            type_demands = array.array("l", [1 for _ in range(platform_limit_dim)])
            type_info_init(type_info, capacities.data.as_longs, type_demands.data.as_longs, platform_limit_dim, raw_type["price"])

    cdef task_info_t* _ctask_info(self, int task_id):
        return self.c.tasks + task_id

    cdef type_info_t* _ctype_info(self, int type_id):
        return self.c.types + type_id

    @property
    def charge_unit(self):
        return self.c.charge_unit

    @charge_unit.setter
    def charge_unit(self, int charge_unit):
        problem_set_charge_unit(&self.c, charge_unit)

    @property
    def platform_limits(self):
        return mr_wrap_c(self.c.platform_limits)

    @platform_limits.setter
    def platform_limits(self, list platform_limits):
        cdef array.array pls = array.array("l", platform_limits)
        problem_set_platform_limits(&self.c, pls.data.as_longs, len(pls))

    def __dealloc__(self):
        problem_destory(&self.c)

    def reverse_dag(self):
        problem_reverse_dag(&self.c)

    @classmethod
    def load(cls, wrk_file, plt_file):
        with open(wrk_file) as f:
            raw_tasks = json.load(f)
        with open(plt_file) as f:
            raw_types = json.load(f)
        return cls(raw_tasks, raw_types, 1)

    @property
    def num_tasks(self):
        return self.c.num_tasks

    @property
    def num_types(self):
        return self.c.num_types

    @property
    def multiresource_dimension(self):
        return MULTIRES_DIM

    @property
    def platform_limit_dimension(self):
        return self.c.platform_limit_dim

    @property
    def tasks(self):
        return list(range(self.c.num_tasks))

    @property
    def types(self):
        return list(range(self.c.num_types))

    def task_demands(self, int task_id):
        return mr_wrap_c(problem_task_demands(&self.c, task_id))

    def task_runtime(self, int task_id, int type_id):
        return problem_task_runtime(&self.c, task_id, type_id)

    def task_prevs(self, int task_id):
        cdef int* prevs = problem_task_prevs(&self.c, task_id)
        return [prevs[i] for i in problem_task_num_prevs(&self.c, task_id)]

    def data_size_between(self, int task_0, int task_1):
        return problem_data_size(&self.c, task_0, task_1)

    def task_succs(self, int task_id):
        cdef int* succs = problem_task_succs(&self.c, task_id)
        return [succs[i] for i in problem_task_num_succs(&self.c, task_id)]

    def type_capacities(self, int type_id):
        return mr_wrap_c(problem_type_capacities(&self.c, type_id))

    def type_price(self, int type_id):
        return problem_type_price(&self.c, type_id)

    def vm_cost(self, int type_id, int runtime):
        return problem_charge(&self.c, type_id, runtime)
