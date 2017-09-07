from MrWSI.core.platform import COMM_INPUT, COMM_OUTPUT
from .base import Heuristic
from math import ceil


def memo(func):
    store = func.__name__ + "_store"

    def wrapped(self, task):
        d = getattr(self, store, None)
        if d is None:
            d = {}
            setattr(self, store, d)
        if task not in d:
            d[task] = func(self, task)
        return d[task]

    return wrapped


class StaticSort(Heuristic):
    def rank(self, task):
        raise NotImplementedError

    def sort_tasks(self):
        return sorted(self.problem.tasks, key=self.rank, reverse=True)


class ReadySort(Heuristic):
    def rank(self, task):
        raise NotImplementedError

    def sort_tasks(self):
        self.rem_in_deps = {t: t.in_degree for t in self.problem.tasks}
        self.ready_set = set(t for t in self.problem.tasks if not t.in_degree)
        while (self.ready_set):
            task = max(self.ready_set, key=self.rank)
            self.ready_set.remove(task)
            for succ in task.succs():
                self.rem_in_deps[succ] -= 1
                if not self.rem_in_deps[succ]:
                    self.ready_set.add(succ)
                    del self.rem_in_deps[succ]
            yield task


class GlobalSort(Heuristic):
    def rank(self, task):
        raise NotImplementedError

    def rank_clean(self, task):
        pass

    def next_task(self, tasks):
        task = max(tasks, key=self.rank)
        pts = [t for t in task.prevs() if t in self.rem_tasks]
        if pts:
            return self.next_task(pts)
        else:
            return task

    def sort_tasks(self):
        self.rem_tasks = set(self.problem.tasks)
        task = None
        while (self.rem_tasks):
            if task: self.rank_clean(task)
            task = self.next_task(self.rem_tasks)
            self.rem_tasks.remove(task)
            yield task


class UpwardRanking(StaticSort):
    @memo
    def rank(self, task):
        return max(
            [
                self.rank(comm.to_task) +
                comm.runtime(self.problem.vm_type.bandwidth)
                for comm in task.communications(COMM_OUTPUT)
            ],
            default=0) + task.runtime(self.problem.vm_type)


class TCTRanking(ReadySort):
    def tct(self, task, comm_type):
        return ceil(task.data_size(comm_type) / self.bandwidth)

    @memo
    def rank(self, task):
        return task.runtime(self.vm_type) + self.tct(task, COMM_OUTPUT) + max(
            [self.rank(t) for t in task.succs()], default=0)


class SRanking(ReadySort):
    @memo
    def rank(self, task):
        return sum(
            c.runtime(self.bandwidth)
            for c in task.communications(COMM_INPUT)) + task.runtime(
                self.vm_type) + max(
                    [self.rank(t) for t in task.succs()], default=0)


class MRanking(GlobalSort):
    @memo
    def rank(self, task):
        return max(
            [
                c.runtime(self.bandwidth)
                for c in task.communications(COMM_INPUT)
            ],
            default=0) + task.runtime(self.vm_type) + max(
                [self.rank(t) for t in task.succs()], default=0)

class LLTRanking(GlobalSort):
    @memo
    def rank_u(self, task):
        return task.runtime(self.vm_type) + max(
            [
                self.rank_u(c.to_task) + c.runtime(self.bandwidth)
                for c in task.communications(COMM_OUTPUT)
            ],
            default=0)

    @memo
    def rank_d(self, task):
        if task in self.placements:
            return self.ST(task)
        else:
            return max(
                [
                    self.rank_d(c.from_task) + c.runtime(self.bandwidth)
                    for c in task.communications(COMM_INPUT)
                ],
                default=0) + task.runtime(self.vm_type)

    def rank(self, task):
        return self.rank_u(task) + self.rank_d(task) - task.runtime(
            self.vm_type)

    def delete_rank_d(self, task):
        if task in self.rank_d_store:
            del self.rank_d_store[task]
            for st in task.succs():
                self.delete_rank_d(st)

    def rank_clean(self, task):
        self.delete_rank_d(task)
        delattr(self, "rank_u_store")


class LLT2Ranking(LLTRanking):
    def tct(self, task, comm_type):
        return ceil(task.data_size(comm_type) / self.bandwidth)

    @memo
    def rank_u(self, task):
        return task.runtime(self.vm_type) + self.tct(task, COMM_OUTPUT) + max(
            [self.rank_u(t) for t in task.succs()], default=0)

    @memo
    def rank_d(self, task):
        if task in self.placements:
            return self.FT(task)
        else:
            lst = 0
            for pt in task.prevs():
                st = sum(
                    c.runtime(self.bandwidth)
                    for c in pt.communications(COMM_OUTPUT)
                    if self.need_communication(c)) + self.rank_d(pt)
                lst = max(lst, st)
            return lst + task.runtime(self.vm_type)


class LLT3Ranking(LLTRanking):
    @memo
    def rank_u(self, task):
        comms = sorted(
            [c for c in task.communications(COMM_OUTPUT)],
            key=lambda c: self.rank(c.to_task),
            reverse=True)
        if not comms:
            return task.runtime(self.vm_type)
        else:
            tft = task.runtime(self.vm_type)
            cft = tft
            for c in comms:
                cft += c.runtime(self.bandwidth)
                tft = max(tft, cft + self.rank_u(c.to_task))
            return tft

    @memo
    def rank_d(self, task):
        if task in self.placements:
            return self.FT(task)
        else:
            lst = 0
            for comm in task.communications(COMM_INPUT):
                pt = comm.from_task
                st = self.rank_d(pt) + comm.runtime(self.bandwidth)
                for c in pt.communications(COMM_OUTPUT):
                    if c.to_task in self.placements and \
                       self.need_communication(c):
                        st += c.runtime(self.bandwidth)
                lst = max(lst, st)
            return lst + task.runtime(self.vm_type)
