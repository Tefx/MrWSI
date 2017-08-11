from MrWSI.core.resource import MultiRes
from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context, Machine, Platform


def task_prioritizing(problem):
    def upward_rank(task):
        if not hasattr(task, "rank_u"):
            task.rank_u = max(
                [upward_rank(t)
                 for t in task.succs()], default=0) + task.mean_runtime()
        return task.rank_u

    return sorted(problem.tasks, key=upward_rank, reverse=True)


def candidate_types(problem, min_mr, max_mr):
    typs = [typ for typ in problem.types if typ.capacities() >= min_mr]
    for typ in typs:
        if not any(
                t for t in typs
                if t != typ and max_mr <= t.capacities() <= typ.capacities()):
            yield typ


def heft(problem):
    context = Context(problem)
    platform = Platform(context)
    finish_times = {}
    for task in task_prioritizing(problem):
        est = max([finish_times[t.task_id] for t in task.prevs()], default=0)
        st_bst, ci_bst, machine_bst, type_bst = float("inf"), float(
            "inf", ), None, None
        for machine in platform:
            for vm_type in candidate_types(
                    problem,
                    MultiRes.max(machine.peak_usage(), task.demands()),
                    machine.peak_usage() + task.demands()):
                st, _ = machine.earliest_slot(vm_type, task, est)
                ci = machine.cost_increase(st, task.runtime(vm_type))
                if (st, ci) < (st_bst, ci_bst):
                    st_bst, ci_bst, machine_bst, type_bst = st, ci, machine, vm_type
        for vm_type in candidate_types(problem, task.demands(),
                                       task.demands()):
            st, _ = platform.earliest_slot(vm_type.demands(),
                                           task.runtime(vm_type), est)
            ci = vm_type.charge(task.runtime(vm_type))
            if (st, ci) < (st_bst, ci_bst):
                st_bst, ci_bst, machine_bst, type_bst = st, ci, None, vm_type

        if not machine_bst:
            machine_bst = Machine(type_bst, context)
        else:
            machine_bst.vm_type = type_bst
        task.start_time = st
        machine_bst.place_task(task, st_bst)
        platform.update_machine(machine_bst)
        finish_times[task.task_id] = st_bst + task.runtime(type_bst)
    return platform
