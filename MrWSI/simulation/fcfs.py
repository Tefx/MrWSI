from MrWSI.simulation.common import *


class FCFSMachineSnap(MachineSnap):
    def __init__(self, vm_type):
        super().__init__(vm_type)
        self.in_link = self.out_link = None
        self.links = {"input": None, "output": None}

    def start_communication(self, env, current_time, comm_event, comm_type):
        self.links[comm_type] = comm_event

    def finish_communication(self, comm_event, comm_type):
        self.links[comm_type] = None

    def available_bandwidth(self, vm_type):
        return self.bandwidth


class FCFSEnvironment(SimulationEnvironment):
    machine_snap_cls = FCFSMachineSnap

    def communication_is_ready(self, event):
        return super().communication_is_ready(event) and not (
            event.from_machine(self).links["output"]
            or event.to_machine(self).links["input"])
