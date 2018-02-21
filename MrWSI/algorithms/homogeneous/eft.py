from MrWSI.core.problem import COMM_INPUT, COMM_OUTPUT
from .base import Heuristic, Machine, memo
from .sorting import UpwardRanking
from statistics import mean
from math import inf


class EFT(UpwardRanking):
    alg_name = "EFT"
    allow_share = True

    def earliest_start_time(self, task, machine):
        est = 0
        for comm in task.communications(COMM_INPUT):
            if self.need_communication(comm, machine):
                est = max(est, self.FT(comm.from_task) + self.RT(comm))
            else:
                est = max(est, self.FT(comm.from_task))
        return est

    def placement_on(self, task, machine):
        est = self.earliest_start_time(task, machine)
        st, _ = machine.earliest_slot_for_task(self.vm_type, task, est)
        return machine, st

    def fitness(self, task, machine, st):
        return st + task.runtime(self.vm_type)

    def perform_placement(self, task, placement):
        machine, st = placement
        self.place_task(task, machine, st)
        for comm in task.communications(COMM_INPUT):
            if self.need_communication(comm, machine):
                self.start_times[comm] = self.FT(comm.from_task)
                self.finish_times[
                    comm] = self.FT(comm.from_task) + self.RT(comm)


class PEFT(Heuristic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m = min(self.problem.num_tasks, self.problem.platform_limits[0])
        self.machines = [Machine(self.problem.vm_type, self.context)
                         for _ in range(m)]

    def available_machines(self):
        return sorted(self.machines, key=lambda m:-len(m.tasks))

    @memo
    def OCT(self, task, machine):
        res = 0
        for t in task.succs():
            res = max(res,
                      min(self.OCT(t, m) + self.RT(t) + (0 if m == machine else self._CT[task.id, t.id]) for m in self.machines))
        return res

    def sort_tasks(self):
        return sorted(self.problem.tasks, key=lambda t: -mean(self.OCT(t, m) for m in self.machines))

    def default_fitness(self):
        return inf

    def fitness(self, task, machine, comm_pls, st):
        return st + self.RT(task) + self.OCT(task, machine)
