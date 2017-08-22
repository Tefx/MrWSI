from MrWSI.core.platform import bandwidth2capacities, COMM_INPUT, COMM_OUTPUT
from MrWSI.core.resource import MultiRes

from math import floor, ceil
import heapq
from copy import copy

RES_DIM = 4


class Event(object):
    class_rank = NotImplemented

    def __init__(self, env, *args, **kwargs):
        self.env = env

    def object_rank(self):
        return NotImplemented

    def __lt__(self, other):
        return (self.class_rank, self.object_rank()) < (other.class_rank,
                                                        other.object_rank())


class StartEvent(Event):
    class_rank = 1

    def start(self, current_time):
        raise NotImplementedError

    def is_ready():
        raise NotImplementedError


class FinishEvent(Event):
    class_rank = 0

    def __init__(self, env, start_time, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.start_time = start_time
        self.cancelled = False

    def finish(self, current_time):
        raise NotImplementedError

    def finish_time(self):
        raise NotImplementedError

    def cancel(self):
        self.cancelled = True


class TaskEvent(Event):
    def __init__(self, env, task, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.task = task
        self.machine = env.machine_for_task(task)

    def object_rank(self):
        return self.env.schedule.ST(self.task)


class CommEvent(Event):
    def __init__(self, env, comm, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.comm = comm
        self.from_machine = env.machine_for_task(self.comm.from_task)
        self.to_machine = env.machine_for_task(self.comm.to_task)

    def object_rank(self):
        return self.env.schedule.CST(self.comm)


class TaskStartEvent(TaskEvent, StartEvent):
    def start(self, current_time):
        finish_event = self.env.task_finish_event_cls(self.env, self.task,
                                                      current_time)
        self.machine.add_task(finish_event, current_time)
        return finish_event

    def is_ready(self):
        for comm in self.task.in_communications:
            if self.env.schedule.is_inter_comm(comm):
                if not self.env.is_finished(comm):
                    return False
            elif not self.env.is_finished(comm.from_task):
                return False
        return self.machine.remaining_resources >= self.task.demands()

    def __repr__(self):
        return "[TaskStart:{}]".format(self.task)


class TaskFinishEvent(TaskEvent, FinishEvent):
    def __init__(self, env, task, start_time):
        super().__init__(env=env, task=task, start_time=start_time)

    def finish(self, current_time):
        if not self.cancelled:
            self.machine.remove_task(self, current_time)
            self.env.mark_finished(self.task)

    def finish_time(self):
        return self.start_time + self.task.runtime(self.machine.vm_type)

    def __repr__(self):
        return "[TaskFinish:{}]".format(self.task)


class CommStartEvent(CommEvent, StartEvent):
    def start(self, current_time):
        finish_event = self.env.comm_finish_event_cls(
            self.env, self.comm, current_time,
            self.possible_bandwidth(), self.comm.data_size)
        self.from_machine.add_comm(finish_event, current_time, COMM_OUTPUT)
        self.to_machine.add_comm(finish_event, current_time, COMM_INPUT)
        return finish_event

    def is_ready(self):
        return self.env.is_finished(
            self.comm.from_task) and self.possible_bandwidth() > 0

    def possible_bandwidth(self):
        return min(
            self.from_machine.available_bandwidth(COMM_OUTPUT),
            self.to_machine.available_bandwidth(COMM_INPUT))

    def __repr__(self):
        return "[CommStart:{}]".format(self.comm)


class CommFinishEvent(CommEvent, FinishEvent):
    def __init__(self, env, comm, start_time, bandwidth, data_size):
        super().__init__(env=env, comm=comm, start_time=start_time)
        self.bandwidth = bandwidth
        self.data_size = data_size

    def finish(self, current_time):
        if not self.cancelled:
            self.from_machine.remove_comm(self, current_time, COMM_OUTPUT)
            self.to_machine.remove_comm(self, current_time, COMM_INPUT)
            self.env.mark_finished(self.comm)

    def finish_time(self):
        return self.start_time + ceil(self.data_size / self.bandwidth)

    def __repr__(self):
        return "[CommFinish:{}]<{}|{}>".format(self.comm, self.bandwidth, self.cancelled)


class SimMachine(object):
    def __init__(self, mtype):
        self.vm_type = mtype
        self.remaining_resources = copy(mtype.capacities)
        self.bandwidth = mtype.capacities[3]
        self.links = [set(), set()]
        self.histroy = []

    def record(self, current_time):
        if self.histroy and self.histroy[-1][0] == current_time:
            self.histroy[-1][1] = copy(self.remaining_resources)
        else:
            self.histroy.append([current_time, copy(self.remaining_resources)])

    @property
    def open_time(self):
        return self.histroy[0][0]

    @property
    def close_time(self):
        return self.histroy[-1][0]

    def cost(self):
        return self.vm_type.charge(self.close_time - self.open_time)

    def add_task(self, event, current_time):
        self.remaining_resources -= event.task.demands()
        self.record(current_time)

    def remove_task(self, event, current_time):
        self.remaining_resources += event.task.demands()
        self.record(current_time)

    def add_comm(self, event, current_time, comm_type):
        self.links[comm_type].add(event)
        assert self.remaining_bandwidth(comm_type) >= 0
        self.remaining_resources -= bandwidth2capacities(
            event.bandwidth, RES_DIM, comm_type)
        assert self.remaining_bandwidth(comm_type) >= 0
        self.record(current_time)

    def remove_comm(self, event, current_time, comm_type):
        self.links[comm_type].remove(event)
        assert self.remaining_bandwidth(comm_type) >= 0
        self.remaining_resources += bandwidth2capacities(
            event.bandwidth, RES_DIM, comm_type)
        assert self.remaining_bandwidth(comm_type) >= 0
        self.record(current_time)

    def remaining_bandwidth(self, comm_type):
        return self.remaining_resources[2 + comm_type]

    def available_bandwidth(self, comm_type):
        return self.remaining_bandwidth(comm_type)

    def fix_remaining_bandwidth(self, comm_type):
        self.remaining_resources[2 + comm_type] = self.bandwidth - sum(
            e.bandwidth for e in self.links[comm_type])


class SimEnv(object):
    machine_cls = SimMachine
    task_start_event_cls = TaskStartEvent
    task_finish_event_cls = TaskFinishEvent
    comm_start_event_cls = CommStartEvent
    comm_finish_event_cls = CommFinishEvent

    def __init__(self, problem, schedule):
        self.problem = problem
        self.schedule = schedule
        self.machines = []
        self.event_queue = []
        self.delayed_events = []
        self.finished_set = set()
        self.prepare_machines()
        self.prepare_events()

    def prepare_machines(self):
        self.machines = [
            self.machine_cls(self.schedule.TYP(i))
            for i in range(self.schedule.num_vms)
        ]

    def prepare_events(self):
        for task in self.problem.tasks:
            self.push_event(
                self.schedule.ST(task), self.task_start_event_cls(self, task))
            for comm in task.out_communications:
                if self.schedule.is_inter_comm(comm):
                    self.push_event(
                        self.schedule.CST(comm),
                        self.comm_start_event_cls(self, comm))

    def machine_for_task(self, task):
        return self.machines[self.schedule.PL(task)]

    def is_finished(self, obj):
        return obj.id_for_set in self.finished_set

    def mark_finished(self, obj):
        self.finished_set.add(obj.id_for_set)

    def pop_event(self):
        return heapq.heappop(self.event_queue)

    def push_event(self, time, event):
        return heapq.heappush(self.event_queue, (time, event))

    def pop_events_at_next_time(self):
        current_time, event = self.pop_event()
        events = [event]
        while self.event_queue and current_time == self.event_queue[0][0]:
            events.append(self.pop_event()[1])
        return current_time, events

    def on_start_event(self, event, current_time):
        finish_event = event.start(current_time)
        self.push_event(finish_event.finish_time(), finish_event)

    def on_finish_event(self, event, current_time):
        event.finish(current_time)

    def after_events(self, current_time):
        pass

    def run(self):
        while self.event_queue:
            current_time, events = self.pop_events_at_next_time()
            for e in events:
                if isinstance(e, FinishEvent):
                    self.on_finish_event(e, current_time)
            for t, e in sorted(self.delayed_events):
                if e.is_ready():
                    self.on_start_event(e, current_time)
                    self.delayed_events.remove((t, e))
            for e in events:
                if isinstance(e, StartEvent):
                    if e.is_ready():
                        self.on_start_event(e, current_time)
                    else:
                        self.delayed_events.append((current_time, e))
            self.after_events(current_time)
        for machine in self.machines:
            for t, u in machine.histroy:
                if not MultiRes.zero(
                        RES_DIM) <= u <= machine.vm_type.capacities:
                    print(t, u)
        span = max(machine.close_time for machine in self.machines) - min(
            machine.open_time for machine in self.machines)
        cost = sum(machine.cost() for machine in self.machines)
        return span, cost


class FCFSEnv(SimEnv):
    pass
