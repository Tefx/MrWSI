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


class Machine(Bin):
    def __init__(self, vm_type, context):
        super().__init__(context.problem.multiresource_dimension,
                         context.machine_node_pool, context.machine_item_pool)
        self.item = None
        self.problem = context.problem
        self.vm_type = vm_type
        self.tasks = set()

    def capacities(self):
        return self.problem.type_capacities(self.type_id)

    def earliest_slot(self, vm_type, task, est):
        return super().earliest_slot(vm_type.capacities(),
                                     task.demands(), task.runtime(vm_type),
                                     est)

    def place_task(self, task, start_time, start_node=None):
        self.tasks.add(task)
        task.item = self.alloc_item(start_time,
                                    task.demands(),
                                    task.runtime(self.vm_type), start_node)

    def extendable_interval(self, task):
        return super().extendable_interval(task.item, self.capacities())

    def cost(self):
        return self.vm_type.charge(self.span())

    def cost_increase(self, start_time, runtime):
        new_runtime = max(start_time + runtime, self.close_time()) - min(
            start_time, self.open_time())
        return self.vm_type.charge(new_runtime) - self.cost()

    def __iter__(self):
        for task in self.tasks:
            yield task


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
            machine.item = self.alloc_item(machine.open_time(),
                                           machine.vm_type.demands(),
                                           machine.span(), start_node)
        else:
            machine.item = self.extend_item(machine.item, machine.open_time(),
                                            machine.close_time())

    def earliest_slot(self, demands, length, est):
        return super().earliest_slot(self.problem.platform_limits, demands,
                                     length, est)

    def __iter__(self):
        for machine in self.machines:
            yield machine

    def cost(self):
        return sum(machine.cost() for machine in self)

    def span(self):
        return max(machine.close_time()
                   for machine in self) - min(machine.open_time()
                                              for machine in self)
