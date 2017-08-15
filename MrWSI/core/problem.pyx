from cpython cimport array
import array
import json
from math import ceil
from itertools import product

DEF MULTIRES_DIM = 3

def calculate_succs(tasks):
    succs = {task_id:[] for task_id in tasks}
    for task_id, task in tasks.items():
        for prev_id in task["prevs"].keys():
            succs[prev_id].append(task_id)
    return succs

cdef class VMType:
    def __cinit__(VMType self, Problem problem, int type_id):
        self.problem = problem
        self.type_id = type_id

    def capacities(VMType self):
        return self.problem.type_capacities(self.type_id)

    def bandwidth(VMType self):
        return self.problem.type_capacities(self.type_id)[2]

    def demands(VMType self):
        return self.problem.type_demands(self.type_id)

    def price(VMType self):
        return self.problem.type_price(self.type_id)

    def charge(VMType self, int runtime):
        return self.problem.vm_cost(self.type_id, runtime)

    def __repr__(self):
        return self.problem.type_str_ids[self.type_id]

class Task(object):
    def __init__(self, problem, task_id):
        self.problem = problem
        self.task_id = task_id

    def demands(self):
        return self.problem.task_demands(self.task_id)

    def runtime(self, typ):
        return self.problem.task_runtime(self.task_id, typ.type_id)

    def prevs(self):
        return self.problem.task_prevs(self.task_id)

    def succs(self):
        return self.problem.task_succs(self.task_id)

    def is_entry(self):
        return not self.prevs()

    def is_exit(self):
        return not self.succs()

    def mean_runtime(self):
        return self.problem.task_mean_runtime(self.task_id)

    def data_size_between(self, to_task):
        return self.problem.data_size_between(self.task_id, to_task.task_id)

    def __repr__(self):
        return self.problem.task_str_ids[self.task_id]

cdef class Problem:
    def __cinit__(Problem self, dict tasks, dict types, int platform_limit_dim):
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
            raw_task["demands"][0] = ceil(raw_task["demands"][0]) * 1000
            # raw_task["demands"][0] *= 1000
            demands = array.array("l", map(int, raw_task["demands"]))
            prevs = array.array("i", [self.task_str_ids.index(t) for t in raw_task["prevs"].keys()])
            succs = array.array("i", [self.task_str_ids.index(t) for t in raw_succs[task_str_id]])
            runtimes = array.array("i", [ceil(raw_task["runtime"]/types[p]["speed"]) for p in self.type_str_ids])
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

        bws = [min(typ0.bandwidth(), typ1.bandwidth()) for typ0, typ1 in product(self.types, self.types)]
        self.mean_bandwidth = int(sum(bws) / len(bws))

    cdef task_info_t* _ctask_info(Problem self, int task_id):
        return self.c.tasks + task_id

    cdef type_info_t* _ctype_info(Problem self, int type_id):
        return self.c.types + type_id

    @property
    def charge_unit(Problem self):
        return self.c.charge_unit

    @charge_unit.setter
    def charge_unit(Problem self, int charge_unit):
        problem_set_charge_unit(&self.c, charge_unit)

    @property
    def platform_limits(Problem self):
        return mr_wrap_c(self.c.platform_limits, self.c.platform_limit_dim)

    @platform_limits.setter
    def platform_limits(Problem self, list platform_limits):
        cdef array.array pls = array.array("l", platform_limits)
        problem_set_platform_limits(&self.c, pls.data.as_longs, len(pls))

    def __dealloc__(Problem self):
        problem_destory(&self.c)

    def reverse_dag(Problem self):
        problem_reverse_dag(&self.c)

    @classmethod
    def load(Problem cls, wrk_file, plt_file, 
             type_family="all", charge_unit=3600,platform_limits=[20]):
        with open(wrk_file) as f:
            raw_tasks = json.load(f)
        with open(plt_file) as f:
            raw_types = json.load(f)
        if type_family != "all":
            raw_types = {name:info for name,info in raw_types.items() if type_family in name}
        problem = cls(raw_tasks, raw_types, 1)
        problem.charge_unit = charge_unit
        problem.platform_limits = platform_limits if isinstance(platform_limits, list) else [platform_limits]
        return problem

    @property
    def num_tasks(Problem self):
        return self.c.num_tasks

    @property
    def num_types(Problem self):
        return self.c.num_types

    @property
    def multiresource_dimension(Problem self):
        return MULTIRES_DIM

    @property
    def platform_limit_dimension(Problem self):
        return self.c.platform_limit_dim

    @property
    def tasks(Problem self):
        return [Task(self, task_id) for task_id in range(self.c.num_tasks)]

    @property
    def types(Problem self):
        return [VMType(self, type_id) for type_id in range(self.c.num_types)]

    def task_demands(Problem self, int task_id):
        return mr_wrap_c(problem_task_demands(&self.c, task_id), MULTIRES_DIM)

    def task_runtime(Problem self, int task_id, int type_id):
        return problem_task_runtime(&self.c, task_id, type_id)

    def task_prevs(Problem self, int task_id):
        cdef int* prevs = problem_task_prevs(&self.c, task_id)
        return [Task(self, prevs[i]) for i in range(problem_task_num_prevs(&self.c, task_id))]

    def task_succs(Problem self, int task_id):
        cdef int* succs = problem_task_succs(&self.c, task_id)
        return [Task(self, succs[i]) for i in range(problem_task_num_succs(&self.c, task_id))]

    def data_size_between(Problem self, int task_0, int task_1):
        return problem_data_size(&self.c, task_0, task_1)

    def type_demands(Problem self, int type_id):
        return mr_wrap_c(problem_type_demands(&self.c, type_id), self.c.platform_limit_dim)

    def type_capacities(Problem self, int type_id):
        return mr_wrap_c(problem_type_capacities(&self.c, type_id), MULTIRES_DIM)

    def type_price(Problem self, int type_id):
        return problem_type_price(&self.c, type_id)

    def vm_cost(Problem self, int type_id, int runtime):
        return problem_charge(&self.c, type_id, runtime)

    def task_mean_runtime(Problem self, int task_id):
        return problem_task_average_runtime(&self.c, task_id)

    def type_mean_bandwidth(Problem self):
        return self.mean_bandwidth
