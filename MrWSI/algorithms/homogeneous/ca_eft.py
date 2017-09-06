from MrWSI.core.platform import Machine, COMM_INPUT, COMM_OUTPUT
from .eft import RankUSort
from .base import Heuristic

from math import ceil


class CA_EFT(Heuristic):
    def sorted_in_comms(self, task):
        return sorted(
            task.communications(COMM_INPUT),
            key=lambda comm: comm.runtime(self.bandwidth) + self.FT(comm.from_task),
            reverse=True)

    def find_slots_for_communication(self, comm, from_machine, to_machine):
        comm_runtime = comm.runtime(self.bandwidth)
        comm_st, _, _ = from_machine.earliest_slot_for_communication(
            to_machine, self.vm_type, self.vm_type, comm, self.bandwidth,
            self.FT(comm.from_task))
        comm_crs = [(comm_runtime, self.bandwidth)]
        return comm_st, comm_st + comm_runtime, comm_crs

    def plan_task_on(self, task, machine):
        task_est = 0
        comm_pls = {}
        machines = set(
            self.PL_m(comm.from_task)
            for comm in task.communications(COMM_INPUT))
        machines.add(machine)
        cost_orig = sum(m.cost() for m in machines)
        for comm in self.sorted_in_comms(task):
            if self.need_communication(comm, machine):
                prev_machine = self.PL_m(comm.from_task)
                comm_st, comm_ft, comm_crs = self.find_slots_for_communication(
                    comm, prev_machine, machine)
                self.place_communication(comm, prev_machine, machine, comm_st,
                                         comm_crs)
                comm_pls[comm] = comm_st, comm_crs
                task_est = max(task_est, comm_ft)
            else:
                task_est = max(task_est, self.FT(comm.from_task))
        task_st, _ = machine.earliest_slot_for_task(self.vm_type, task,
                                                    task_est)
        task_runtime = task.runtime(self.vm_type)
        task_ft = task_st + task_runtime
        cost_increase = sum(
            m.cost() for m in machines) - cost_orig + machine.cost_increase(
                task_st, task_runtime, self.vm_type)

        for comm in task.communications(COMM_INPUT):
            if self.need_communication(comm, machine):
                self.remove_communication(comm,
                                          self.PL_m(comm.from_task), machine)

        return (machine, comm_pls, task_st), (task_ft, cost_increase)
        # return (machine, comm_pls, task_st), task_ft

    def place_communication(self, comm, from_machine, to_machine, st, crs):
        from_machine.place_communication(comm, st, crs, COMM_OUTPUT)
        to_machine.place_communication(comm, st, crs, COMM_INPUT)
        self.start_times[comm] = st
        self.finish_times[comm] = st + sum(l for l, _ in crs)

    def remove_communication(self, comm, from_machine, to_machine):
        from_machine.remove_communication(comm, COMM_OUTPUT)
        to_machine.remove_communication(comm, COMM_INPUT)
        del self.start_times[comm]
        del self.finish_times[comm]

    def perform_placement(self, task, placement):
        machine, comm_pls, task_st = placement
        self.place_task(task, machine, task_st)
        for comm, comm_pls in comm_pls.items():
            self.place_communication(comm,
                                     self.PL_m(comm.from_task), machine,
                                     *comm_pls)


class CA_EFT_P(CA_EFT):
    def find_slots_for_communication(self, comm, from_machine, to_machine):
        remaining_data_size = comm.data_size
        st = self.FT(comm.from_task)
        crs = []
        runtime = 0
        bn_0 = bn_1 = None

        while remaining_data_size > 0:
            cr_0, len_0, bn_0 = from_machine.current_available_cr(
                st, self.vm_type, COMM_OUTPUT, bn_0)
            cr_1, len_1, bn_1 = to_machine.current_available_cr(
                st, self.vm_type, COMM_INPUT, bn_1)
            cr = min(cr_0, cr_1)
            length = min(len_0, len_1)
            remaining_data_size -= length * cr
            if remaining_data_size < 0:
                length += ceil(remaining_data_size / cr)
            if crs or cr:
                crs.append([length, cr])
                runtime += length
            st += length

        return st - runtime, st, crs


def memo(func):
    store = func.__name__ + "_store"

    def wrapped(self, task):
        d = getattr(self, store, None)
        if d is None:
            d = {}
            setattr(self, store, d)
        if task not in d:
            d[task] = func(self, task)
        return d[task]

    return wrapped


class RankSort(Heuristic):
    def tct(self, task, comm_type):
        return ceil(task.data_size(comm_type) / self.bandwidth)

    def sort_tasks(self):
        self.rem_in_deps = {t: t.in_degree for t in self.problem.tasks}
        self.ready_set = set(t for t in self.problem.tasks if not t.in_degree)
        while (self.ready_set):
            # print([(t, self.rank(t), self.rank_s(t)) for t in self.ready_set])
            task = max(self.ready_set, key=self.rank)
            self.ready_set.remove(task)
            for succ in task.succs():
                self.rem_in_deps[succ] -= 1
                if not self.rem_in_deps[succ]:
                    self.ready_set.add(succ)
                    del self.rem_in_deps[succ]
            yield task


class RankSSort(RankSort):
    @memo
    def rank(self, task):
        return sum(
            c.runtime(self.bandwidth)
            for c in task.communications(COMM_INPUT)) + task.runtime(
                self.vm_type) + max(
                    [self.rank(t) for t in task.succs()], default=0)


class RankWSort(RankSort):
    @memo
    def rank(self, task):
        return task.runtime(self.vm_type) + self.tct(task, COMM_OUTPUT) + max(
            [self.rank(t) for t in task.succs()], default=0)


class RankFSort(RankSort):
    @memo
    def rank_u(self, task):
        return task.runtime(self.vm_type) + max(
            [
                self.rank_u(c.to_task) + c.runtime(self.bandwidth)
                for c in task.communications(COMM_OUTPUT)
            ],
            default=0)

    @memo
    def rank_d(self, task):
        if task in self.placements:
            return self.ST(task)
        else:
            return max(
                [
                    self.rank_d(c.from_task) +
                    c.from_task.runtime(self.vm_type) +
                    c.runtime(self.bandwidth)
                    for c in task.communications(COMM_INPUT)
                ],
                default=0)

    def rank(self, task):
        return self.rank_u(task) + self.rank_d(task)

    def delete_rank_d(self, task):
        if task in self.rank_d_store:
            del self.rank_d_store[task]
            for st in task.succs():
                self.delete_rank_d(st)

    def next_task(self, tasks):
        task = max(tasks, key=self.rank)
        pts = [t for t in task.prevs() if t in self.rem_tasks]
        if pts:
            return self.next_task(pts)
        else:
            return task

    def sort_tasks(self):
        self.rem_tasks = set(self.problem.tasks)
        task = None
        while (self.rem_tasks):
            if task: self.delete_rank_d(task)
            # print([(t, self.rank(t), self.rank_d(t), self.rank_u(t))
                   # for t in self.rem_tasks])
            task = self.next_task(self.rem_tasks)
            self.rem_tasks.remove(task)
            yield task


class RankESort(RankFSort):
    @memo
    def rank_u(self, task):
        return task.runtime(self.vm_type) + self.tct(task, COMM_OUTPUT) + max(
            [self.rank_u(t) for t in task.succs()], default=0)

    @memo
    def rank_d(self, task):
        if task in self.placements:
            return self.FT(task)
        else:
            return max(
                [
                    self.rank_d(t) + self.tct(t, COMM_OUTPUT)
                    for t in task.prevs()
                ],
                default=0) + task.runtime(self.vm_type)

    def rank(self, task):
        return self.rank_u(task) + self.rank_d(task) - task.runtime(
            self.vm_type)


class R2Sort(Heuristic):
    @memo
    def rank(self, task):
        return max(
            [
                c.runtime(self.bandwidth)
                for c in task.communications(COMM_INPUT)
            ],
            default=0) + task.runtime(self.vm_type) + max(
                [self.rank(t) for t in task.succs()], default=0)

    def next_task(self, tasks):
        task = max(tasks, key=self.rank)
        pts = [t for t in task.prevs() if t in self.unscheduled_tasks]
        if pts:
            yield from self.next_task(pts)
        self.unscheduled_tasks.remove(task)
        yield task

    def sort_tasks(self):
        self.unscheduled_tasks = set(self.problem.tasks)
        while self.unscheduled_tasks:
            yield from self.next_task(self.unscheduled_tasks)


class CA_EFT_U(CA_EFT, RankUSort):
    pass


class CA_EFT_S(CA_EFT, RankSSort):
    pass


class CA_EFT_2(CA_EFT, R2Sort):
    pass


class CA_EFT_PU(CA_EFT_P, RankUSort):
    pass


class CA_EFT_PS(CA_EFT_P, RankSSort):
    pass


class CA_EFT_P2(CA_EFT_P, R2Sort):
    pass


class CA_EFT_PF(CA_EFT_P, RankFSort):
    pass


class CA_EFT_PE(CA_EFT_P, RankESort):
    pass


class CA_EFT_PW(CA_EFT_P, RankWSort):
    pass


def CA_EFT_MIX(problem):
    alg_e = CA_EFT_PE(problem)
    alg_w = CA_EFT_PW(problem)
    if alg_e.span < alg_w.span:
        return alg_e
    else:
        return alg_w
