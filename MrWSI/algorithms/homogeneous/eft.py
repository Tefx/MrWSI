from MrWSI.core.problem import COMM_INPUT, COMM_OUTPUT
from .base import Heuristic


class RankUSort(Heuristic):
    def sort_tasks(self):
        self.ranks = {}
        return sorted(self.problem.tasks, key=self.upward_rank, reverse=True)

    def upward_rank(self, task):
        if task not in self.ranks:
            self.ranks[task] = max(
                [
                    self.upward_rank(comm.to_task) +
                    comm.runtime(self.problem.vm_type.bandwidth)
                    for comm in task.communications(COMM_OUTPUT)
                ],
                default=0) + task.runtime(self.problem.vm_type)
        return self.ranks[task]


class EFT(RankUSort):
    def earliest_start_time(self, task, machine):
        est = 0
        for comm in task.communications(COMM_INPUT):
            if self.need_communication(comm, machine):
                est = max(est,
                          self.FT(comm.from_task) +
                          comm.runtime(self.vm_type.bandwidth))
            else:
                est = max(est, self.FT(comm.from_task))
        return est

    def plan_task_on(self, task, machine):
        est = self.earliest_start_time(task, machine)
        st, _ = machine.earliest_slot_for_task(self.vm_type, task, est)
        runtime = task.runtime(self.vm_type)
        ci = machine.cost_increase(st, runtime, self.vm_type)
        return (machine, st), (st + runtime, ci)

    def perform_placement(self, task, placement):
        machine, st = placement
        self.place_task(task, machine, st)
        for comm in task.communications(COMM_INPUT):
            if self.need_communication(comm, machine):
                self.start_times[comm] = self.FT(comm.from_task)
                self.finish_times[comm] = self.FT(
                    comm.from_task) + comm.runtime(self.bandwidth)
