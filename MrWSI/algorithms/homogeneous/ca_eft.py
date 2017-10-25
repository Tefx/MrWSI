from MrWSI.core.platform import Machine, COMM_INPUT, COMM_OUTPUT
from .base import Heuristic

from math import ceil, floor


def mkalg(name, *cls):
    class InnerClass(*cls):
        alg_name = name

    return InnerClass


class CAEFT(Heuristic):
    allow_share = False
    allow_preemptive = False

    def sorted_in_comms(self, task):
        return sorted(
            task.communications(COMM_INPUT),
            key=lambda comm: self.FT(comm.from_task))

    def find_slots_for_communication(self, comm, from_machine, to_machine):
        comm_runtime = comm.runtime(self.bandwidth)
        comm_st, _, _ = from_machine.earliest_slot_for_communication(
            to_machine, self.vm_type, self.vm_type, comm, self.bandwidth,
            self.FT(comm.from_task))
        comm_crs = [(comm_runtime, self.bandwidth)]
        return comm_st, comm_st + comm_runtime, comm_crs

    def prepare_machines(self, machines):
        pass

    def plan_task_on(self, task, machine):
        task_est = 0
        comm_pls = {}
        # machines = set(
            # self.PL_m(comm.from_task)
            # for comm in task.communications(COMM_INPUT))
        # self.prepare_machines(machines)
        # machines.add(machine)
        # cost_orig = sum(m.cost() for m in machines)
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
        # cost_increase = sum(
        # m.cost() for m in machines) - cost_orig + machine.cost_increase(
        # task_st, task_runtime, self.vm_type)

        for comm in task.communications(COMM_INPUT):
            if self.need_communication(comm, machine):
                self.remove_communication(comm,
                                          self.PL_m(comm.from_task), machine)

        # return (machine, comm_pls, task_st), (task_ft, cost_increase)
        return (machine, comm_pls, task_st), task_ft

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


class CAEFT_P(CAEFT):
    allow_share = False
    allow_preemptive = True

    def find_slots_for_communication(self, comm, from_machine, to_machine):
        return from_machine.find_idle_common_slots2(to_machine,
                                                    self.FT(comm.from_task),
                                                    comm.data_size,
                                                    2 + COMM_OUTPUT,
                                                    2 + COMM_INPUT,
                                                    self.bandwidth)
