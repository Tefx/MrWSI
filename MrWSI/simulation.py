import heapq
from math import floor, ceil


class MachineSnap(object):
    def __init__(self, capacities):
        self.remaining_resources = capacities
        self.in_remaining_bandwidth = capacities[2]
        self.out_remaining_bandwidth = capacities[2]
        self.bandwidth = capacities[2]
        self.in_links = set()
        self.out_links = set()

    def resources_enough_for(self, task_event):
        return self.remaining_resources >= task_event.task.demands()

    def start_task(self, task_event):
        self.remaining_resources -= task_event.task.demands()

    def finish_task(self, task_event):
        self.remaining_resources += task_event.task.demands()

    def start_in_link(self, env, current_time, comm_event):
        fair_bandwidth = self.spare_in_bandwidth()
        for event in [
                event for event in self.in_links
                if event.bandwidth > fair_bandwidth
        ]:
            finish_event = event.adjust_bandwidth(
                env, fair_bandwidth - event.bandwidth, current_time)
            env.push_event(finish_event.predicted_finish_time(), finish_event)
        self.in_links.add(comm_event)
        self.in_remaining_bandwidth = self.bandwidth - sum(
            event.bandwidth for event in self.in_links)

    def start_out_link(self, env, current_time, comm_event):
        fair_bandwidth = self.spare_out_bandwidth()
        for event in [
                event for event in self.out_links
                if event.bandwidth > fair_bandwidth
        ]:
            finish_event = event.adjust_bandwidth(
                env, fair_bandwidth - event.bandwidth, current_time)
            env.push_event(finish_event.predicted_finish_time(), finish_event)
        self.out_links.add(comm_event)
        self.out_remaining_bandwidth = self.bandwidth - sum(
            event.bandwidth for event in self.out_links)

    def finish_in_link(self, comm_event):
        self.in_links.remove(comm_event)
        self.in_remaining_bandwidth += comm_event.bandwidth

    def finish_out_link(self, comm_event):
        self.out_links.remove(comm_event)
        self.out_remaining_bandwidth += comm_event.bandwidth

    def spare_in_bandwidth(self):
        return floor(self.bandwidth / (len(self.in_links) + 1))

    def spare_out_bandwidth(self):
        return floor(self.bandwidth / (len(self.out_links) + 1))

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
            if adjustable_bandwidth:
                new_event = event.adjust_bandwidth(env, adjustable_bandwidth,
                                                   current_time)
                self.in_remaining_bandwidth -= adjustable_bandwidth
                has_adjustment = True
                env.push_event(new_event.predicted_finish_time(), new_event)
        for event in self.out_links:
            adjustable_bandwidth = min(event.in_adjustable_bandwidth,
                                       event.out_adjustable_bandwidth)
            if adjustable_bandwidth:
                new_event = event.adjust_bandwidth(env, adjustable_bandwidth,
                                                   current_time)
                self.out_remaining_bandwidth -= adjustable_bandwidth
                has_adjustment = True
                env.push_event(new_event.predicted_finish_time(), new_event)
        return has_adjustment


class Event(object):
    def __lt__(self, other):
        return True


class StartEvent(Event):
    pass


class FinishEvent(Event):
    pass


class TaskEvent(Event):
    def __init__(self, task, machine):
        self.task = task
        self.machine = machine


class TaskFinishEvent(TaskEvent, FinishEvent):
    def finish_time(self, start_time, env):
        return start_time + self.task.runtime(
            env.schedule.TYP(env.schedule.PL(self.task)))

    def __repr__(self):
        return "TASK_STOP({})".format(self.task.task_id)


class TaskStartEvent(TaskEvent, StartEvent):
    def finish_event(self):
        return TaskFinishEvent(self.task, self.machine)

    def __repr__(self):
        return "TASK_START({})".format(self.task.task_id)


class CommunicationEvent(Event):
    def __init__(self, from_task, to_task):
        self.task_pair = (from_task, to_task)

    def from_machine(self, env):
        return env.task2machine(self.task_pair[0])

    def to_machine(self, env):
        return env.task2machine(self.task_pair[1])


class CommunicationFinishEvent(CommunicationEvent, FinishEvent):
    def __init__(self, task_pair, bandwidth, current_time, data_size):
        super().__init__(*task_pair)
        self.data_size = data_size
        self.bandwidth = bandwidth
        self.start_time = current_time
        self.cancelled = False
        self.in_adjustable_bandwidth = 0
        self.out_adjustable_bandwidth = 0

    def predicted_finish_time(self):
        return self.start_time + ceil(float(self.data_size) / self.bandwidth)

    def adjust_bandwidth(self, env, adjust_bandwidth, current_time):
        self.cancelled = True
        new_event = CommunicationFinishEvent(
            self.task_pair, self.bandwidth + adjust_bandwidth, current_time,
            self.data_size - self.bandwidth * (current_time - self.start_time))
        from_machine = self.from_machine(env)
        to_machine = self.to_machine(env)
        from_machine.out_links.remove(self)
        from_machine.out_links.add(new_event)
        to_machine.in_links.remove(self)
        to_machine.in_links.add(new_event)
        return new_event

    def __repr__(self):
        return "COMM_STOP[{}]({}, {}|{})".format(self.cancelled,
                                                 self.task_pair[0].task_id,
                                                 self.task_pair[1].task_id,
                                                 self.predicted_finish_time())


class CommunicationStartEvent(CommunicationEvent, StartEvent):
    def finish_event(self, bandwidth, current_time):
        return CommunicationFinishEvent(
            self.task_pair, bandwidth, current_time,
            self.task_pair[0].data_size_between(self.task_pair[1]))

    def __repr__(self):
        return "COMM_START({}, {})".format(self.task_pair[0].task_id,
                                           self.task_pair[1].task_id)


class SimulationEnvironment(object):
    def __init__(self, problem, schedule):
        self.last_stop_time = 0
        self.event_queue = []
        self.delayed_events = []
        self.problem = problem
        self.schedule = schedule
        self.prepare_machines()
        self.prepare_events()
        self.finished_tasks = set()
        self.finished_communications = set()

    def prepare_machines(self):
        self.machines = []
        for i in range(self.schedule.num_vms):
            self.machines.append(
                MachineSnap(self.schedule.TYP(i).capacities()))

    def prepare_events(self):
        for task in self.problem.tasks:
            st = self.schedule.ST(task)
            pl = self.schedule.PL(task)
            self.push_event(st, TaskStartEvent(task, self.machines[pl]))
            for succ_task in task.succs():
                if self.schedule.PL(task) != self.schedule.PL(succ_task):
                    self.push_event(st + task.runtime(self.schedule.TYP(pl)),
                                    CommunicationStartEvent(task, succ_task))

    def pop_event(self):
        return heapq.heappop(self.event_queue)

    def push_event(self, time, event):
        heapq.heappush(self.event_queue, (time, event))

    def task2machine(self, task):
        return self.machines[self.schedule.PL(task)]

    def on_start_event(self, event, current_time):
        # print(current_time, event)
        if isinstance(event, TaskStartEvent):
            event.machine.start_task(event)
            finish_event = event.finish_event()
            self.push_event(
                finish_event.finish_time(current_time, self), finish_event)
        elif isinstance(event, CommunicationStartEvent):
            from_machine = event.from_machine(self)
            to_machine = event.to_machine(self)
            bandwidth = min(from_machine.spare_out_bandwidth(),
                            to_machine.spare_in_bandwidth())
            finish_event = event.finish_event(bandwidth, current_time)
            from_machine.start_out_link(self, current_time, finish_event)
            to_machine.start_in_link(self, current_time, finish_event)
            self.push_event(finish_event.predicted_finish_time(), finish_event)

    def on_finish_event(self, event, current_time):
        # if not (isinstance(event, CommunicationFinishEvent) and event.cancelled):
            # print(current_time, event)
        if isinstance(event, TaskFinishEvent):
            event.machine.finish_task(event)
            self.finished_tasks.add(event.task.task_id)
        elif isinstance(event,
                        CommunicationFinishEvent) and not event.cancelled:
            event.from_machine(self).finish_out_link(event)
            event.to_machine(self).finish_in_link(event)
            self.finished_communications.add((event.task_pair[0].task_id,
                                              event.task_pair[1].task_id))

    def event_is_ready(self, event):
        if isinstance(event, TaskStartEvent):
            for prev_task in event.task.prevs():
                if self.schedule.PL(prev_task) != self.schedule.PL(event.task):
                    if (prev_task.task_id, event.task.task_id
                        ) not in self.finished_communications:
                        return False
            return event.machine.resources_enough_for(event)
        elif isinstance(event, CommunicationStartEvent):
            return event.task_pair[0].task_id in self.finished_tasks

    def adjust_bandwidths(self, current_time):
        has_adjustment = True
        while has_adjustment:
            for machine in self.machines:
                machine.assign_adjustable_bandwidths()
            for machine in self.machines:
                has_adjustment = machine.adjust_bandwidths(self, current_time)

    def run(self):
        while self.event_queue:
            current_time, event = self.pop_event()
            events = [event]
            while self.event_queue and current_time == self.event_queue[0][0]:
                events.append(self.pop_event()[1])
            for e in events:
                if isinstance(e, FinishEvent):
                    self.on_finish_event(e, current_time)
            for t, e in sorted(self.delayed_events):
                if self.event_is_ready(e):
                    self.delayed_events.remove((t, e))
                    self.on_start_event(e, current_time)
            for e in events:
                if isinstance(e, StartEvent):
                    if self.event_is_ready(e):
                        self.on_start_event(e, current_time)
                    else:
                        self.delayed_events.append((current_time, e))
            self.adjust_bandwidths(current_time)
        return current_time
