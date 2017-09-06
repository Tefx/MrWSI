from MrWSI.core.resource import MultiRes
from MrWSI.core.problem import Problem, COMM_INPUT, COMM_OUTPUT
from MrWSI.core.platform import Context, Machine, Platform
from MrWSI.core.schedule import Schedule
from math import ceil


def sort_by_rank_u(problem):
    ranks = {}

    def upward_rank(task):
        if task not in ranks:
            ranks[task] = max(
                [
                    upward_rank(comm.to_task) + comm.mean_runtime()
                    for comm in task.communications(COMM_OUTPUT)
                ],
                default=0) + task.mean_runtime()
        return ranks[task]

    return sorted(problem.tasks, key=upward_rank, reverse=True)


def candidate_types(problem, min_mr, max_mr):
    typs = [typ for typ in problem.types if typ.capacities >= min_mr]
    for typ in typs:
        flag = False
        for t in typs:
            if t != typ and max_mr <= t.capacities <= typ.capacities:
                flag = True
                break
        if not flag:
            yield typ


def heft(problem, limit=1000):
    context = Context(problem)
    platform = Platform(context)
    start_times = {}
    finish_times = {}
    machines = {}
    for task in sort_by_rank_u(problem):
        st_bst, ft_bst, ci_bst, machine_bst, type_bst = (float("inf"),
                                                         float("inf"),
                                                         float("inf"), None,
                                                         None)
        for machine in platform:
            for vm_type in candidate_types(
                    problem,
                    MultiRes.max(machine.peak_usage(), task.demands()),
                    machine.peak_usage() + task.demands()):

                est = 0
                for comm in task.communications(COMM_INPUT):
                    if comm.from_task in machine:
                        est = max(est, finish_times[comm.from_task])
                    else:
                        bandwidth = min(
                            machines[comm.from_task].vm_type.bandwidth,
                            vm_type.bandwidth)
                        est = max(est, finish_times[comm.from_task] +
                                  comm.runtime(bandwidth))

                st, _ = machine.earliest_slot_for_task(vm_type, task, est)
                runtime = task.runtime(vm_type)
                ft = st + runtime
                ci = machine.cost_increase(st, runtime, vm_type)
                if (ft, ci) < (ft_bst, ci_bst):
                    st_bst, ft_bst, ci_bst, machine_bst, type_bst = (st, ft,
                                                                     ci,
                                                                     machine,
                                                                     vm_type)

        if limit > len(platform.machines):
            for vm_type in candidate_types(problem,
                                           task.demands(), task.demands()):
                est = 0
                for comm in task.communications(COMM_INPUT):
                    bandwidth = min(machines[comm.from_task].vm_type.bandwidth,
                                    vm_type.bandwidth)
                    est = max(
                        est,
                        finish_times[comm.from_task] + comm.runtime(bandwidth))
                ft = est + task.runtime(vm_type)
                ci = vm_type.charge(task.runtime(vm_type))
                if (ft, ci) < (ft_bst, ci_bst):
                    st_bst, ft_bst, ci_bst, machine_bst, type_bst = (est, ft,
                                                                     ci, None,
                                                                     vm_type)
        if not machine_bst:
            machine_bst = Machine(type_bst, context)
        else:
            machine_bst.vm_type = type_bst
        machine_bst.place_task(task, st_bst)
        platform.update_machine(machine_bst)
        machines[task] = machine_bst
        start_times[task] = st_bst
        finish_times[task] = st_bst + task.runtime(type_bst)

    pls = {}
    typs = []
    for i, machine in enumerate(platform.machines):
        for task in machine.tasks:
            pls[task] = i
        typs.append(machine.vm_type)
        schedule = Schedule(
            problem, lambda x: pls[x], lambda x: typs[x],
            lambda x: start_times[x] if isinstance(x, Task) else finish_times[x.from_task],
            len(platform))
    cost = sum(
        machine_info(machine, schedule)[2] for machine in platform.machines)
    return schedule, cost


def machine_info(machine, schedule):
    def comm_time(comm):
        if schedule.PL(comm.from_task) == schedule.PL(comm.to_task):
            return 0
        else:
            return comm.runtime(
                min(
                    schedule.TYP_PL(comm.from_task).bandwidth,
                    schedule.TYP_PL(comm.to_task).bandwidth))

    open_time = min(
        min([schedule.FT(prev_task) for prev_task in task.prevs()],
            default=schedule.ST(task)) for task in machine.tasks)
    close_time = max(
        schedule.FT(task) + max(
            [comm_time(comm) for comm in task.communications(COMM_OUTPUT)], default=0)
        for task in machine.tasks)
    return open_time, close_time, machine.cost(close_time - open_time)
