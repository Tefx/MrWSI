from MrWSI.core.problem import Problem
from MrWSI.core.bin import MemPool, Bin


class Context(object):
    def __init__(self, problem):
        self.problem = problem
        self.machine_node_pool = MemPool(problem.multiresource_dimension,
                                         problem.num_tasks, "node")
        self.machine_item_pool = MemPool(problem.multiresource_dimension,
                                         problem.num_tasks, "item")
        self.platform_node_pool = MemPool(problem.platform_limit_dimension,
                                          problem.num_types, "node")
        self.platform_item_pool = MemPool(problem.platform_limit_dimension,
                                          problem.num_types, "item")


class Task(objective):
    def __init__(self, problem, type_id):
        self.problem = problem
        self.type_id = type_id

    def runtime(self, machine):
        return self.problem.task_runtime(self.task_id, machine.type_id)

    def demands(self, machine):
        return self.problem.task_demands(self.task_id)


class Machine(Bin):
    def __init__(self, type_id, context):
        super().__init__(context.problem.multiresource_dimension,
                         context.machine_node_pool, context.machine_item_pool)
        self.item = None
        self.problem = context.problem
        self.type_id = type_id
        self.tasks = set()

    def capacities(self):
        return self.problem.type_capacities(self.type_id)

    def earliest_slot(self, task, est):
        return super().earliest_slot(self.capacities(),
                                     task.demands(self),
                                     task.runtime(self), est)

    def place_task(self, task, start_time, start_node):
        task.item = self.alloc_item(start_time,
                                    task.demands(self),
                                    task.runtime(self), start_node)

    def extendable_interval(self, task):
        return super().extendable_interval(task.item, self.capacities())


class Platform(Bin):
    def __init__(self, context):
        super().__init__(context.problem.platform_limit_dimension,
                         context.platform_node_pool,
                         context.platform_item_pool)
        self.problem = context.problem
        self.machines = set()

    def extendable_interval(self, machine):
        ei0, ei1 = super().extendable_interval(machine.item,
                                               self.problem.platform_limits)
        if ei0 == self.open_time:
            ei0 = 0
        if ei1 == self.close_time:
            ei1 = int(float("inf"))
        return e10, ei1

    def update_machine(self, machine, start_node=None):
        if machine not in self.machines:
            self.machines.add(machine)
            machine.item = self.alloc_item(
                machine.open_time,
                problem.type_capacities(machine.type_id), machine.span,
                start_node)
        else:
            machine.item = self.extend_item(machine.item, machine.open_time,
                                            machine.close_time)

    def earliest_slot(self, demands, length, est):
        return super().earliest_slot(self.problem.platform_limits, demands,
                                     length, est)
