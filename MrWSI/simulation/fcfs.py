from MrWSI.simulation.common import *


class FCFSCommunicationStartEvent(CommunicationStartEvent):
    def possible_bandwidth(self):
        return min(self.from_machine.bandwidth, self.to_machine.bandwidth)


class FCFSMachineSnap(MachineSnap):
    def __init__(self, vm_type):
        super().__init__(vm_type)
        self.links = {"input": None, "output": None}

    def start_communication(self, current_time, event, comm_type):
        super().start_communication(current_time, event, comm_type)
        self.links[comm_type] = event

    def finish_communication(self, current_time, event, comm_type):
        super().finish_communication(current_time, event, comm_type)
        self.links[comm_type] = None

    def available_bandwidth(self, comm_type):
        return self.bandwidth if not self.links[comm_type] else 0


class FCFSEnvironment(SimulationEnvironment):
    machine_snap_cls = FCFSMachineSnap
    comm_start_event_cls = FCFSCommunicationStartEvent


class FCFS2CommunicationStartEvent(CommunicationStartEvent):
    def possible_bandwidth(self):
        return min(self.from_machine.available_bandwidth("output"),
                   self.to_machine.available_bandwidth("input"))


class FCFS2MachineSnap(MachineSnap):
    def __init__(self, vm_type):
        super().__init__(vm_type)
        self.remaining_bandwidth = {
            "input": self.bandwidth,
            "output": self.bandwidth
        }

    def start_communication(self, current_time, event, comm_type):
        super().start_communication(current_time, event, comm_type)
        self.remaining_bandwidth[comm_type] -= event.bandwidth

    def finish_communication(self, current_time, event, comm_type):
        super().finish_communication(current_time, event, comm_type)
        self.remaining_bandwidth[comm_type] += event.bandwidth

    def available_bandwidth(self, comm_type):
        return self.remaining_bandwidth[comm_type]


class FCFS2Environment(SimulationEnvironment):
    machine_snap_cls = FCFS2MachineSnap
    comm_start_event_cls = FCFS2CommunicationStartEvent
