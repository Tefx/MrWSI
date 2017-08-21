from math import floor, ceil
import heapq


class Event(object):
    def __lt__(self, other):
        return (self.rank, repr(self)) <= (other.rank, repr(other))
        # return (self.rank, id(self)) <= (other.rank, id(self))


class StartEvent(Event):
    rank = 1
    pass


class FinishEvent(Event):
    rank = 0
    pass


class TaskEvent(Event):
    def __init__(self, env, task):
        self.env = env
        self.task = task
        self.machine = env.task2machine(task)


class TaskStartEvent(TaskEvent, StartEvent):
    def start(self, current_time):
        self.machine.start_task(current_time, self)
        return self.env.task_finish_event_cls(self.env, self.task,
                                              current_time)

    def __repr__(self):
        return "TASK_START({})".format(self.task.task_id)


class TaskFinishEvent(TaskEvent, FinishEvent):
    def __init__(self, env, task, current_time):
        super().__init__(env, task)
        self.start_time = current_time

    def finish(self, current_time):
        self.machine.finish_task(current_time, self)

    def finish_time(self):
        return self.start_time + self.task.runtime(
            self.env.schedule.TYP(self.env.schedule.PL(self.task)))

    def __repr__(self):
        return "TASK_STOP({})".format(self.task.task_id)


class CommunicationEvent(Event):
    def __init__(self, env, communication):
        self.env = env
        self.communication = communication
        self.data_size = communication.data_size
        self.from_machine = env.task2machine(communication.from_task)
        self.to_machine = env.task2machine(communication.to_task)

    def from_task(self):
        return self.communication.from_task

    def to_task(self):
        return self.communication.to_task

    def task_id_pair(self):
        return self.communication.from_task_id, self.communication.to_task_id


class CommunicationStartEvent(CommunicationEvent, StartEvent):
    def possible_bandwidth(self):
        raise NotImplementedError

    def start(self, current_time):
        finish_event = self.env.comm_finish_event_cls(
            self.env, self.communication,
            self.possible_bandwidth(), current_time, self.data_size)
        self.from_machine.start_communication(current_time, finish_event,
                                              "output")
        self.to_machine.start_communication(current_time, finish_event,
                                            "input")
        return finish_event

    def bandwidth_is_ready(self):
        bandwidth_demands = self.possible_bandwidth()
        return self.from_machine.available_bandwidth(
            "output"
        ) >= bandwidth_demands and self.to_machine.available_bandwidth(
            "input") >= bandwidth_demands

    def __repr__(self):
        return "COMM_START({}, {})".format(*self.task_id_pair())


class CommunicationFinishEvent(CommunicationEvent, FinishEvent):
    def __init__(self, env, communication, bandwidth, current_time, data_size):
        super().__init__(env, communication)
        self.data_size = data_size
        self.bandwidth = bandwidth
        self.start_time = current_time

    def finish(self, current_time):
        self.from_machine.finish_communication(current_time, self, "output")
        self.to_machine.finish_communication(current_time, self, "input")

    def finish_time(self):
        return self.start_time + ceil(float(self.data_size) / self.bandwidth)

    def __repr__(self):
        return "COMM_STOP({}, {})[{}]".format(self.communication.from_task_id,
                                              self.communication.to_task_id,
                                              self.finish_time())


class MachineSnap(object):
    def __init__(self, vm_type):
        self.vm_type = vm_type
        self.remaining_resources = vm_type.capacities
        self.bandwidth = vm_type.bandwidth
        self.launched = False
        self.open_time = self.close_time = 0

    def cost(self):
        return self.vm_type.charge(self.close_time - self.open_time)

    def resources_enough_for(self, task_event):
        return self.remaining_resources >= task_event.task.demands()

    def start_task(self, current_time, event):
        if not self.launched:
            self.open_time = current_time
            self.launched = True
        self.remaining_resources -= event.task.demands()

    def finish_task(self, current_time, event):
        self.close_time = current_time
        self.remaining_resources += event.task.demands()

    def start_communication(self, current_time, event, comm_type):
        if not self.launched:
            self.open_time = current_time
            self.launched = True

    def finish_communication(self, current_time, event, comm_type):
        self.close_time = current_time

    def available_bandwidth(self, comm_type):
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
            self.push_event(
                self.schedule.ST(task), self.task_start_event_cls(self, task))
            for comm in task.out_communications:
                if self.schedule.PL(comm.from_task) != self.schedule.PL(
                        comm.to_task):
                    self.push_event(
                        self.schedule.CST(comm.from_task, comm.to_task),
                        self.comm_start_event_cls(self, comm))

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
            elif prev_task.task_id not in self.finished_tasks:
                return False
        return event.machine.resources_enough_for(event)

    def communication_is_ready(self, event):
        return event.task_id_pair(
        )[0] in self.finished_tasks and event.possible_bandwidth(
        ) > 0 and event.bandwidth_is_ready()

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
        span = max(machine.close_time for machine in self.machines) - min(
            machine.open_time for machine in self.machines)
        cost = sum(machine.cost() for machine in self.machines)
        return span, cost

    def on_start_event(self, event, current_time):
        finish_event = event.start(current_time)
        self.push_event(finish_event.finish_time(), finish_event)

    def on_finish_event(self, event, current_time):
        event.finish(current_time)
        if isinstance(event, TaskEvent):
            self.finished_tasks.add(event.task.task_id)
        elif isinstance(event, CommunicationEvent):
            self.finished_communications.add(event.task_id_pair())

    def adjust(self, current_time):
        pass
