from MrWSI.simulation.common import *


class FairMachineSnap(MachineSnap):
    def __init__(self, vm_type):
        super().__init__(vm_type)
        self.in_remaining_bandwidth = self.bandwidth
        self.out_remaining_bandwidth = self.bandwidth
        self.in_links = set()
        self.out_links = set()

    def start_communication(self, env, current_time, comm_event, comm_type):
        fair_bandwidth = self.available_bandwidth(comm_type)
        links = self.in_links if comm_type == "input" else self.out_links
        for event in [
                event for event in links if event.bandwidth > fair_bandwidth
        ]:
            event.adjust_bandwidth(env, fair_bandwidth - event.bandwidth,
                                   current_time)
        if comm_type == "input":
            self.in_links.add(comm_event)
            self.in_remaining_bandwidth = self.bandwidth - sum(
                event.bandwidth for event in self.in_links)
        elif comm_type == "output":
            self.out_links.add(comm_event)
            self.out_remaining_bandwidth = self.bandwidth - sum(
                event.bandwidth for event in self.out_links)

    def finish_communication(self, comm_event, comm_type):
        if comm_type == "input":
            self.in_links.remove(comm_event)
            self.in_remaining_bandwidth += comm_event.bandwidth
        elif comm_type == "output":
            self.out_links.remove(comm_event)
            self.out_remaining_bandwidth += comm_event.bandwidth

    def available_bandwidth(self, vm_type):
        num_links = len(self.in_links
                        if vm_type == "input" else self.out_links) + 1
        return floor(self.bandwidth / num_links)

    def assign_adjustable_bandwidths(self):
        if self.in_remaining_bandwidth > 0 and self.in_links:
            in_increase = floor(
                self.in_remaining_bandwidth / len(self.in_links))
            for event in self.in_links:
                event.in_adjustable_bandwidth = in_increase
        if self.out_remaining_bandwidth > 0 and self.out_links:
            out_increase = floor(
                self.out_remaining_bandwidth / len(self.out_links))
            for event in self.out_links:
                event.out_adjustable_bandwidth = out_increase

    def adjust_bandwidths(self, env, current_time):
        has_adjustment = False
        for event in self.in_links:
            adjustable_bandwidth = min(event.in_adjustable_bandwidth,
                                       event.out_adjustable_bandwidth)
            if adjustable_bandwidth > 0:
                event.adjust_bandwidth(env, adjustable_bandwidth, current_time)
                self.in_remaining_bandwidth -= adjustable_bandwidth
                has_adjustment = True
        for event in self.out_links:
            adjustable_bandwidth = min(event.in_adjustable_bandwidth,
                                       event.out_adjustable_bandwidth)
            if adjustable_bandwidth > 0:
                event.adjust_bandwidth(env, adjustable_bandwidth, current_time)
                self.out_remaining_bandwidth -= adjustable_bandwidth
                has_adjustment = True
        return has_adjustment


class FairCommunicationFinishEvent(CommunicationFinishEvent):
    def __init__(self, task_pair, bandwidth, current_time, data_size):
        super().__init__(task_pair, bandwidth, current_time, data_size)
        self.in_adjustable_bandwidth = 0
        self.out_adjustable_bandwidth = 0
        self.cancelled = False

    def adjust_bandwidth(self, env, adjust_bandwidth, current_time):
        new_event = FairCommunicationFinishEvent(
            self.task_pair, self.bandwidth + adjust_bandwidth, current_time,
            self.data_size - self.bandwidth * (current_time - self.start_time))
        from_machine = self.from_machine(env)
        from_machine.out_links.remove(self)
        from_machine.out_links.add(new_event)
        to_machine = self.to_machine(env)
        to_machine.in_links.remove(self)
        to_machine.in_links.add(new_event)
        self.cancelled = True
        env.push_event(new_event.finish_time(), new_event)


class FairEnvironment(SimulationEnvironment):
    machine_snap_cls = FairMachineSnap
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
                has_adjustment = machine.adjust_bandwidths(self, current_time)
