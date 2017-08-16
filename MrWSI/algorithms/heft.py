from MrWSI.core.resource import MultiRes
from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context, Machine, Platform
from MrWSI.core.schedule import Schedule
from math import ceil


def sort_by_rank_u(problem):
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
    for task in sort_by_rank_u(problem):
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
            for vm_type in candidate_types(problem,
                                           task.demands(), task.demands()):
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
    typs = []
    for i, machine in enumerate(platform.machines):
        for task in machine.tasks:
            pls[task.task_id] = i
        typs.append(machine.vm_type)
        schedule = Schedule.from_arrays(problem, pls, typs, start_times,
                                        lambda x, y: finish_times[x.task_id])
    cost = sum(
        machine_info(machine, schedule)[2] for machine in platform.machines)
    return schedule, cost


def machine_info(machine, schedule):
    def comm_time(task, succ_task):
        if schedule.PL(task) == schedule.PL(succ_task):
            return 0
        else:
            return ceil(
                task.data_size_between(succ_task) / min(
                    schedule.TYP_PL(task).bandwidth(),
                    schedule.TYP_PL(succ_task).bandwidth()))

    open_time = min(
        min([schedule.FT(prev_task) for prev_task in task.prevs()],
            default=schedule.ST(task)) for task in machine)
    close_time = max(
        schedule.FT(task) + max(
            [comm_time(task, succ_task) for succ_task in task.succs()],
            default=0) for task in machine)
    return open_time, close_time, machine.cost(close_time - open_time)
