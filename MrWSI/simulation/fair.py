from MrWSI.simulation.base import *


class FairCommFinishEvent(CommFinishEvent):
    def __init__(self, env, comm, start_time, bandwidth, data_size):
        super().__init__(env, comm, start_time, bandwidth, data_size)
        self.adjustable_bandwidths = [0, 0]
        self.cancelled = False

    def adjust_bandwidth(self, bandwidth_delta, current_time):
        self.cancel()

        new_event = FairCommFinishEvent(self.env, self.comm, current_time,
                                        self.bandwidth + bandwidth_delta,
                                        self.data_size - self.bandwidth *
                                        (current_time - self.start_time))
        self.env.push_event(new_event.finish_time(), new_event)
        self.from_machine.links[COMM_OUTPUT].remove(self)
        self.from_machine.links[COMM_OUTPUT].add(new_event)
        self.from_machine.remaining_resources -= bandwidth2capacities(
            bandwidth_delta, RES_DIM, COMM_OUTPUT)
        if self.from_machine.remaining_bandwidth(COMM_OUTPUT) < 0:
            print("ERROR!", self, self.from_machine.remaining_resources,
                  bandwidth_delta)
            for event in self.from_machine.links[COMM_OUTPUT]:
                print(event, event.bandwidth)
        assert self.from_machine.remaining_bandwidth(COMM_OUTPUT) >= 0

        self.to_machine.links[COMM_INPUT].remove(self)
        self.to_machine.links[COMM_INPUT].add(new_event)
        self.to_machine.remaining_resources -= bandwidth2capacities(
            bandwidth_delta, RES_DIM, COMM_INPUT)
        assert self.from_machine.remaining_bandwidth(COMM_INPUT) >= 0


class FairMachine(SimMachine):
    def add_comm(self, event, current_time, comm_type):
        fair_bandwidth = self.available_bandwidth(comm_type)
        for e in list(self.links[comm_type]):
            if e.bandwidth > fair_bandwidth:
                e.adjust_bandwidth(fair_bandwidth - e.bandwidth, current_time)
        super().add_comm(event, current_time, comm_type)

    def available_bandwidth(self, comm_type):
        return floor(self.bandwidth / (len(self.links[comm_type]) + 1))

    def mark_adjustable_comms(self):
        for comm_type in (COMM_OUTPUT, COMM_INPUT):
            if self.remaining_bandwidth(comm_type) > 0 \
               and self.links[comm_type]:
                fair_share = floor(
                    self.remaining_bandwidth(comm_type) /
                    len(self.links[comm_type]))
                for e in self.links[comm_type]:
                    e.adjustable_bandwidths[comm_type] = fair_share
            else:
                for e in self.links[comm_type]:
                    e.adjustable_bandwidths[comm_type] = 0

    def adjust_task_bandwidths(self, current_time):
        has_adjustment = False
        for e in self.links[COMM_OUTPUT]:
            bandwidth_delta = min(e.adjustable_bandwidths)
            if bandwidth_delta > 0:
                e.adjust_bandwidth(bandwidth_delta, current_time)
                has_adjustment = True
        if has_adjustment:
            self.record(current_time)
        return has_adjustment


class FairEnv(SimEnv):
    machine_cls = FairMachine
    comm_finish_event_cls = FairCommFinishEvent
    env_name = "fair"
    allow_share = True

    def after_events(self, current_time):
        while True:
            for machine in self.machines:
                machine.mark_adjustable_comms()
            if not any(
                    machine.adjust_task_bandwidths(current_time)
                    for machine in self.machines):
                break
