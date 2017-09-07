from MrWSI.core.problem import COMM_INPUT, COMM_OUTPUT
from .base import Heuristic
from .sorting import UpwardRanking


class EFT(UpwardRanking):
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
