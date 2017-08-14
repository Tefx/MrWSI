from MrWSI.core.resource import MultiRes
from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context, Machine, Platform
from MrWSI.core.schedule import Schedule
from math import ceil


def task_prioritizing(problem):
    mean_bandwidth = problem.type_mean_bandwidth()
    ranks = {}

    def upward_rank(task):
        if task.task_id not in ranks:
            ranks[task.task_id] = max(
                [
                    upward_rank(t) + ceil(
                        task.data_size_between(t) / mean_bandwidth)
                    for t in task.succs()
                ],
                default=0) + task.mean_runtime()
        return ranks[task.task_id]

    return sorted(problem.tasks, key=upward_rank, reverse=True)


def candidate_types(problem, min_mr, max_mr):
    typs = [typ for typ in problem.types if typ.capacities() >= min_mr]
    for typ in typs:
        if not any(
                t for t in typs
                if t != typ and max_mr <= t.capacities() <= typ.capacities()):
            yield typ


def heft(problem, limit=1000):
    context = Context(problem)
    platform = Platform(context)
    start_times = {}
    finish_times = {}
    machines = {}
    for task in task_prioritizing(problem):
        st_bst, ci_bst, machine_bst, type_bst = float("inf"), float(
            "inf", ), None, None
        for machine in platform:
            for vm_type in candidate_types(
                    problem,
                    MultiRes.max(machine.peak_usage(), task.demands()),
                    machine.peak_usage() + task.demands()):

                est = 0
                for prev_task in task.prevs():
                    if prev_task in machine.tasks:
                        est = max(est, finish_times[prev_task.task_id])
                    else:
                        bandwidth = min(
                            machines[prev_task.task_id].vm_type.bandwidth(),
                            vm_type.bandwidth())
                        est = max(est, finish_times[prev_task.task_id] + ceil(
                            prev_task.data_size_between(task) / bandwidth))

                st, _ = machine.earliest_slot(vm_type, task, est)
                ci = machine.cost_increase(st, task.runtime(vm_type))
                if (st, ci) < (st_bst, ci_bst):
                    st_bst, ci_bst, machine_bst, type_bst = st, ci, machine, vm_type

        if limit > len(platform.machines):
            for vm_type in candidate_types(problem, task.demands(),
                                           task.demands()):
                est = 0
                for prev_task in task.prevs():
                    bandwidth = min(
                        machines[prev_task.task_id].vm_type.bandwidth(),
                        vm_type.bandwidth())
                    est = max(est, finish_times[prev_task.task_id] + ceil(
                        prev_task.data_size_between(task) / bandwidth))

                ci = vm_type.charge(task.runtime(vm_type))
                if (est, ci) < (st_bst, ci_bst):
                    st_bst, ci_bst, machine_bst, type_bst = est, ci, None, vm_type

        if not machine_bst:
            machine_bst = Machine(type_bst, context)
        else:
            machine_bst.vm_type = type_bst
        machine_bst.place_task(task, st_bst)
        platform.update_machine(machine_bst)
        machines[task.task_id] = machine_bst
        start_times[task.task_id] = st_bst
        finish_times[task.task_id] = st_bst + task.runtime(type_bst)

    pls = {}
    typs = [0] * len(platform.machines)
    for i, machine in enumerate(platform.machines):
        for task in machine.tasks:
            pls[task.task_id] = i
        typs[i] = machine.vm_type
    return platform, Schedule.from_arrays(pls, typs, start_times)
