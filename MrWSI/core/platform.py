from MrWSI.core.problem import Problem, COMM_INPUT, COMM_OUTPUT
from MrWSI.core.bin import MemPool, Bin
from MrWSI.core.resource import MultiRes

from math import inf

def bandwidth2capacities(bw, dimension, comm_type):
    c = MultiRes.zero(dimension)
    c[2 + comm_type] = bw
    return c


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
        self.communications = set()
        self.self_context = not context

    def capacities(self):
        return self.problem.type_capacities(self.type_id)

    def current_available_cr(self, time, vm_type, comm_type, bn=None):
        vol, length, bn = self.current_block(time, bn)
        cr = vm_type.bandwidth - vol[2 + comm_type]
        return cr, length, bn

    def earliest_slot_for_task(self, vm_type, task, est):
        return self.earliest_slot(vm_type.capacities,
                                  task.demands(), task.runtime(vm_type), est)

    def earliest_slot_for_communication(self, to_machine, vm_type, to_vm_type,
                                        communication, bandwidth, est):
        return self.earliest_slot_2(
            to_machine, vm_type.capacities, to_vm_type.capacities,
            bandwidth2capacities(
                bandwidth, self.problem.multiresource_dimension, COMM_OUTPUT),
            bandwidth2capacities(
                bandwidth, self.problem.multiresource_dimension, COMM_INPUT),
            communication.runtime(bandwidth), est)

    def place_task(self, task, start_time, start_node=None):
        self.tasks.add(task)
        task.item = self.alloc_item(start_time,
                                    task.demands(),
                                    task.runtime(self.vm_type), start_node)

    def comm_finish_time_est(self, comm, st, comm_type):
        bandwidth = self.vm_type.bandwidth
        return self.eft_for_demand(bandwidth * comm.runtime(bandwidth),
                                   2 + comm_type, st, self.vm_type.capacities)

    def remove_task(self, task):
        self.tasks.remove(task)
        self.free_item(task.item)

    def place_communication(self, comm, start_time, crs, comm_type):
        self.communications.add(comm)
        comm.items[comm_type] = self.alloc_multi_items(
            start_time, crs, 2 + comm_type)
        # dimension = self.problem.multiresource_dimension
        # comm.items[comm_type] = []
        # for rt, cr in crs:
        # if cr:
        # cap = bandwidth2capacities(cr, dimension, comm_type)
        # item = self.alloc_item(start_time, cap, rt, None)
        # comm.items[comm_type].append(item)
        # start_time += rt

    def remove_communication(self, comm, comm_type):
        self.communications.remove(comm)
        for item in comm.items[comm_type]:
            self.free_item(item)
        comm.items[comm_type] = []

    def extendable_interval(self, task):
        return super().extendable_interval(task.item, self.capacities())

    def cost(self, span=None):
        if not self.vm_type:
            return 0
        if not span:
            span = self.span()
        return self.vm_type.charge(span)

    def cost_increase(self, start_time, runtime, vm_type):
        new_runtime = max(start_time + runtime, self.close_time()) - min(
            start_time, self.open_time())
        return vm_type.charge(new_runtime) - self.cost()

    def __contains__(self, x):
        return x in self.tasks or x in self.communications

    def __repr__(self):
        return "Machine<{}>".format(str(id(self))[-4:])


class Platform(Bin):
    def __init__(self, context):
        super().__init__(context.problem.platform_limit_dimension,
                         context.platform_node_pool,
                         context.platform_item_pool)
        self.problem = context.problem
        self.machines = []
        self.context = context

    def extendable_interval(self, machine):
        ei0, ei1 = super().extendable_interval(machine.item,
                                               self.problem.platform_limits)
        if ei0 == self.open_time:
            ei0 = 0
        if ei1 == self.close_time:
            ei1 = inf
        return ei0, ei1

    def update_machine(self, machine, start_node=None):
        if machine not in self.machines:
            self.machines.append(machine)
            machine.item = self.alloc_item(machine.open_time(),
                                           machine.vm_type.demands(),
                                           machine.span(), start_node)
        else:
            machine.item = self.extend_item(machine.item,
                                            machine.open_time(),
                                            machine.close_time())

    def earliest_slot(self, demands, length, est):
        return super().earliest_slot(self.problem.platform_limits, demands,
                                     length, est)

    def __iter__(self):
        for machine in self.machines:
            yield machine

    def __len__(self):
        return len(self.machines)

    def cost(self):
        return sum(machine.cost() for machine in self)

    def span(self):
        return max(machine.close_time()
                   for machine in self) - min(machine.open_time()
                                              for machine in self)
