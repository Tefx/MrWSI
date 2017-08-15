from math import floor, ceil
import heapq


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


class TaskStartEvent(TaskEvent, StartEvent):
    def __repr__(self):
        return "TASK_START({})".format(self.task.task_id)


class TaskFinishEvent(TaskEvent, FinishEvent):
    def finish_time(self, start_time, env):
        return start_time + self.task.runtime(
            env.schedule.TYP(env.schedule.PL(self.task)))

    @classmethod
    def from_start_event(cls, start_event):
        return cls(start_event.task, start_event.machine)

    def __repr__(self):
        return "TASK_STOP({})".format(self.task.task_id)


class CommunicationEvent(Event):
    def __init__(self, from_task, to_task):
        self.task_pair = (from_task, to_task)

    def task_id_pair(self):
        return self.task_pair[0].task_id, self.task_pair[1].task_id

    def from_machine(self, env):
        return env.task2machine(self.task_pair[0])

    def to_machine(self, env):
        return env.task2machine(self.task_pair[1])


class CommunicationStartEvent(CommunicationEvent, StartEvent):
    def data_size(self):
        return self.task_pair[0].data_size_between(self.task_pair[1])

    def __repr__(self):
        return "COMM_START({}, {})".format(self.task_pair[0].task_id,
                                           self.task_pair[1].task_id)


class CommunicationFinishEvent(CommunicationEvent, FinishEvent):
    def __init__(self, task_pair, bandwidth, current_time, data_size):
        super().__init__(*task_pair)
        self.data_size = data_size
        self.bandwidth = bandwidth
        self.start_time = current_time

    @classmethod
    def from_start_event(cls, start_event, bandwidth, current_time):
        return cls(start_event.task_pair, bandwidth, current_time,
                   start_event.data_size())

    def finish_time(self):
        return self.start_time + ceil(float(self.data_size) / self.bandwidth)

    def __repr__(self):
        return "COMM_STOP[{}]({}, {}|{})".format(self.cancelled,
                                                 self.task_pair[0].task_id,
                                                 self.task_pair[1].task_id,
                                                 self.finish_time())


class MachineSnap(object):
    def __init__(self, vm_type):
        self.remaining_resources = vm_type.capacities()
        self.bandwidth = self.remaining_resources[2]

    def resources_enough_for(self, task_event):
        return self.remaining_resources >= task_event.task.demands()

    def start_task(self, task_event):
        self.remaining_resources -= task_event.task.demands()

    def finish_task(self, task_event):
        self.remaining_resources += task_event.task.demands()

    def start_communication(self, env, current_time, comm_event, comm_type):
        raise NotImplementedError

    def finish_communication(self, comm_event, comm_type):
        raise NotImplementedError

    def available_bandwidth(self, vm_type):
        raise NotImplementedError


class SimulationEnvironment(object):
    machine_snap_cls = MachineSnap
    task_start_event_cls = TaskStartEvent
    task_finish_event_cls = TaskFinishEvent
    comm_start_event_cls = CommunicationStartEvent
    comm_finish_event_cls = CommunicationFinishEvent

    def __init__(self, problem, schedule):
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
            self.machines.append(self.machine_snap_cls(self.schedule.TYP(i)))

    def prepare_events(self):
        for task in self.problem.tasks:
            st = self.schedule.ST(task)
            pl = self.schedule.PL(task)
            self.push_event(st,
                            self.task_start_event_cls(task, self.machines[pl]))
            for succ_task in task.succs():
                if self.schedule.PL(task) != self.schedule.PL(succ_task):
                    self.push_event(st + task.runtime(self.schedule.TYP(pl)),
                                    self.comm_start_event_cls(task, succ_task))

    def pop_event(self):
        return heapq.heappop(self.event_queue)

    def push_event(self, time, event):
        heapq.heappush(self.event_queue, (time, event))

    def task2machine(self, task):
        return self.machines[self.schedule.PL(task)]

    def task_is_ready(self, event):
        for prev_task in event.task.prevs():
            if self.schedule.PL(prev_task) != self.schedule.PL(event.task):
                if (prev_task.task_id, event.task.task_id
                    ) not in self.finished_communications:
                    return False
        return event.machine.resources_enough_for(event)

    def communication_is_ready(self, event):
        return event.task_id_pair()[0] in self.finished_tasks

    def event_is_ready(self, event):
        if isinstance(event, TaskEvent):
            return self.task_is_ready(event)
        elif isinstance(event, CommunicationEvent):
            return self.communication_is_ready(event)

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
                    self.on_start_event(e, current_time)
                    self.delayed_events.remove((t, e))
            for e in events:
                if isinstance(e, StartEvent):
                    if self.event_is_ready(e):
                        self.on_start_event(e, current_time)
                    else:
                        self.delayed_events.append((current_time, e))
            self.adjust(current_time)
        return current_time

    def on_start_event(self, event, current_time):
        if isinstance(event, TaskEvent):
            self.on_start_task(event, current_time)
        elif isinstance(event, CommunicationEvent):
            self.on_start_communication(event, current_time)

    def on_finish_event(self, event, current_time):
        if isinstance(event, TaskEvent):
            self.on_finish_task(event, current_time)
        elif isinstance(event, CommunicationEvent):
            self.on_finish_communication(event, current_time)

    def on_start_task(self, event, current_time):
        event.machine.start_task(event)
        finish_event = self.task_finish_event_cls.from_start_event(event)
        self.push_event(
            finish_event.finish_time(current_time, self), finish_event)

    def on_start_communication(self, event, current_time):
        from_machine = event.from_machine(self)
        to_machine = event.to_machine(self)
        bandwidth = min(
            from_machine.available_bandwidth("output"),
            to_machine.available_bandwidth("input"))
        finish_event = self.comm_finish_event_cls.from_start_event(
            event, bandwidth, current_time)
        from_machine.start_communication(self, current_time, finish_event,
                                         "output")
        to_machine.start_communication(self, current_time, finish_event,
                                       "input")
        self.push_event(finish_event.finish_time(), finish_event)

    def on_finish_task(self, event, current_time):
        event.machine.finish_task(event)
        self.finished_tasks.add(event.task.task_id)

    def on_finish_communication(self, event, current_time):
        event.from_machine(self).finish_communication(event, "output")
        event.to_machine(self).finish_communication(event, "input")
        self.finished_communications.add(event.task_id_pair())

    def adjust(self, current_time):
        pass
