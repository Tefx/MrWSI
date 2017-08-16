from MrWSI.simulation.common import *


class FairMachineSnap(MachineSnap):
    def __init__(self, vm_type):
        super().__init__(vm_type)
        self.remaining_bandwidths = {
            "input": self.bandwidth,
            "output": self.bandwidth
        }
        self.links = {"input": set(), "output": set()}

    def start_communication(self, current_time, event, comm_type):
        super().start_communication(current_time, event, comm_type)
        fair_bandwidth = self.available_bandwidth(comm_type)
        for other_event in [
                e for e in self.links[comm_type]
                if e.bandwidth > fair_bandwidth
        ]:
            other_event.adjust_bandwidth(
                fair_bandwidth - other_event.bandwidth, current_time)
        self.links[comm_type].add(event)
        self.remaining_bandwidths[comm_type] = self.bandwidth - sum(
            event.bandwidth for event in self.links[comm_type])

    def finish_communication(self, current_time, event, comm_type):
        super().finish_communication(current_time, event, comm_type)
        self.links[comm_type].remove(event)
        self.remaining_bandwidths[comm_type] += event.bandwidth

    def available_bandwidth(self, comm_type):
        return floor(self.bandwidth / (len(self.links[comm_type]) + 1))

    def assign_adjustable_bandwidths(self):
        for comm_type in ("input", "output"):
            if self.remaining_bandwidths[comm_type] > 0 and self.links[comm_type]:
                increase = floor(
                    self.remaining_bandwidths[comm_type] / len(self.links[comm_type]))
                for event in self.links[comm_type]:
                    event.adjustable_bandwidths[comm_type] = increase

    def adjust_bandwidths(self, current_time):
        has_adjustment = False
        for comm_type in ("input", "output"):
            for event in self.links[comm_type]:
                adjustable_bandwidth = min(event.adjustable_bandwidths.values())
                if adjustable_bandwidth > 0:
                    event.adjust_bandwidth(adjustable_bandwidth, current_time)
                    self.remaining_bandwidths[comm_type] -= adjustable_bandwidth
                    has_adjustment = True
        return has_adjustment


class FairCommunicationStartEvent(CommunicationStartEvent):
    def possible_bandwidth(self):
        return min(
            self.from_machine.available_bandwidth("output"),
            self.to_machine.available_bandwidth("input"))


class FairCommunicationFinishEvent(CommunicationFinishEvent):
    def __init__(self, env, task_pair, bandwidth, current_time, data_size):
        super().__init__(env, task_pair, bandwidth, current_time, data_size)
        self.adjustable_bandwidths = {"input": 0, "output": 0}
        self.cancelled = False

    def adjust_bandwidth(self, adjust_bandwidth, current_time):
        new_event = FairCommunicationFinishEvent(
            self.env, self.task_pair, self.bandwidth + adjust_bandwidth,
            current_time,
            self.data_size - self.bandwidth * (current_time - self.start_time))
        self.from_machine.links["output"].remove(self)
        self.from_machine.links["output"].add(new_event)
        self.to_machine.links["input"].remove(self)
        self.to_machine.links["input"].add(new_event)
        self.cancelled = True
        self.env.push_event(new_event.finish_time(), new_event)

    def finish(self, current_time):
        if not self.cancelled:
            super().finish(current_time)


class FairEnvironment(SimulationEnvironment):
    machine_snap_cls = FairMachineSnap
    comm_start_event_cls = FairCommunicationStartEvent
    comm_finish_event_cls = FairCommunicationFinishEvent

    def on_finish_communication(self, event, current_time):
        if not event.cancelled:
            super().on_finish_communication(event, current_time)

    def adjust(self, current_time):
        has_adjustment = True
        while has_adjustment:
            for machine in self.machines:
                machine.assign_adjustable_bandwidths()
            for machine in self.machines:
                has_adjustment = machine.adjust_bandwidths(current_time)
