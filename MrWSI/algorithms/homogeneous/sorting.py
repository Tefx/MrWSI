from MrWSI.core.platform import COMM_INPUT, COMM_OUTPUT, Machine
from .base import Heuristic
from math import ceil
from functools import wraps


def memo(func):
    store = func.__name__ + "_store"

    @wraps(func)
    def wrapped(self, task):
        d = getattr(self, store, None)
        if d is None:
            d = {}
            setattr(self, store, d)
        if task not in d:
            d[task] = func(self, task)
        return d[task]

    return wrapped


def memo_clean(func):
    if hasattr(func.__self__, func.__name__ + "_store"):
        delattr(func.__self__, func.__name__ + "_store")


def tryFT(func):
    @wraps(func)
    def wrapped(self, task):
        if task in self.placements:
            return self.FT(task)
        else:
            return func(self, task)

    return wrapped


class StaticSort(Heuristic):
    def rank(self, task):
        raise NotImplementedError

    def sort_tasks(self):
        for task in sorted(self.problem.tasks, key=self.rank, reverse=True):
            if "sort" in self.log: self.log_sort(task)
            yield task

    def log_sort(self, task):
        print("Chosen", task, self.rank(task))
        print([(task, self.rank(task)) for task in self.problem.tasks
               if task not in self.placements])


class ReadySort(Heuristic):
    def rank(self, task):
        raise NotImplementedError

    def rank_clean(self, task):
        pass

    def sort_tasks(self):
        self.rem_in_deps = {t: t.in_degree for t in self.problem.tasks}
        self.ready_set = set(t for t in self.problem.tasks if not t.in_degree)
        while (self.ready_set):
            task = max(self.ready_set, key=self.rank)
            if "sort" in self.log: self.log_sort(task)
            self.ready_set.remove(task)
            self.rank_clean(task)
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

    def all_prevs(self, task):
        for t in task.prevs():
            if t in self.rem_tasks:
                yield from self.all_prevs(t)
                yield t

    def task_is_ready(self, task):
        return all(pt in self.placements for pt in task.prevs())

    def next_task(self, tasks):
        highest_rank = max(self.rank(t) for t in tasks)
        co_tasks = set()
        pts = set()
        for t in tasks:
            if self.rank(t) == highest_rank:
                if self.task_is_ready(t):
                    return t
                else:
                    pts |= set(pt for pt in t.prevs() if pt in self.rem_tasks)
        return self.next_task(pts)

    def log_sort(self, task):
        pass

    def sort_tasks(self):
        self.rem_tasks = set(self.problem.tasks)
        while (self.rem_tasks):
            task = self.next_task(self.rem_tasks)
            if "sort" in self.log: self.log_sort(task)
            self.rem_tasks.remove(task)
            self.rank_clean(task)
            yield task


class UpwardRanking(StaticSort):
    @memo
    def rank(self, task):
        return max(
            [
                self.rank(comm.to_task) + self.RT(comm)
                for comm in task.communications(COMM_OUTPUT)
            ],
            default=0) + self.RT(task)


class M1Ranking(StaticSort):
    def is_free_task(self, task):
        return not any(t in self.placements for t in task.prevs())

    @memo
    def rank(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank(c.to_task),
            reverse=True)
        rr = 0
        cst = 0
        lcst = 0
        for c in comms:
            if self.is_free_task(
                    c.to_task) and lcst < cst + c.runtime(self.bandwidth):
                lcst += self.rank(c.to_task)
                rr = max(rr, lcst)
            else:
                cst += c.runtime(self.bandwidth)
                rr = max(rr, cst + self.rank(c.to_task))
        return rr + task.runtime(self.vm_type)


class M1_1Ranking(M1Ranking):
    def is_critial_comm(self, comm):
        from_task = comm.from_task
        to_task = comm.to_task
        max_f = max(
            c.runtime(self.bandwidth)
            for c in from_task.communications(COMM_OUTPUT))
        max_t = max(
            c.runtime(self.bandwidth)
            for c in from_task.communications(COMM_OUTPUT))
        return max_f == max_t == comm.runtime(self.bandwidth)

    @memo
    def rank(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank(c.to_task),
            reverse=True)
        rr = 0
        cst = 0
        lcst = 0
        for c in comms:
            if self.is_critial_comm(
                    c) and lcst < cst + c.runtime(self.bandwidth):
                lcst += self.rank(c.to_task)
                rr = max(rr, lcst)
            else:
                cst += c.runtime(self.bandwidth)
                rr = max(rr, cst + self.rank(c.to_task))
        return rr + task.runtime(self.vm_type)


class M2Ranking(M1Ranking):
    @memo
    def rank(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank(c.to_task),
            reverse=True)
        rr = 0
        cst = 0
        lcst = 0
        for c in comms:
            if self.is_free_task(
                    c.to_task) and lcst < cst + c.runtime(self.bandwidth):
                rr = max(rr, lcst + self.rank(c.to_task))
                lcst += c.to_task.runtime(self.vm_type)
            else:
                cst += c.runtime(self.bandwidth)
                rr = max(rr, cst + self.rank(c.to_task))
        return rr + task.runtime(self.vm_type)


class M2_1Ranking(M2Ranking):
    def is_critial_comm(self, comm):
        from_task = comm.from_task
        to_task = comm.to_task
        max_f = max(
            c.runtime(self.bandwidth)
            for c in from_task.communications(COMM_OUTPUT))
        max_t = max(
            c.runtime(self.bandwidth)
            for c in from_task.communications(COMM_OUTPUT))
        return max_f == max_t == comm.runtime(self.bandwidth)

    @memo
    def rank(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank(c.to_task),
            reverse=True)
        rr = 0
        cst = 0
        lcst = 0
        for c in comms:
            if self.is_critial_comm(
                    c) and lcst < cst + c.runtime(self.bandwidth):
                rr = max(rr, lcst + self.rank(c.to_task))
                lcst += c.to_task.runtime(self.vm_type)
            else:
                cst += c.runtime(self.bandwidth)
                rr = max(rr, cst + self.rank(c.to_task))
        return rr + task.runtime(self.vm_type)


class M3Ranking(StaticSort):
    @memo
    def rank(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank(c.to_task),
            reverse=True)
        r = 0
        cst = 0
        for c in comms:
            cst += self.RT(c)
            r = max(r, cst + self.rank(c.to_task))
        return r + self.RT(task)


class M3_1Ranking(StaticSort):
    @memo
    def rank(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank(c.to_task),
            reverse=True)
        if comms:
            lc = len(comms)
            X = [0 for _ in range(lc)]
            Hu = [0 for _ in range(lc)]
            Hd = [0 for _ in range(lc)]
            for i, c in enumerate(comms):
                if i == 0:
                    X[i] = self.RT(c)
                    Hu[i] = self.RT(c) + self.rank(c.to_task)
                else:
                    X[i] = X[i - 1] + self.RT(c)
                    Hu[i] = max(Hu[i - 1], X[i] + self.rank(c.to_task))
            for i, c in enumerate(reversed(comms)):
                if i == 0:
                    Hd[lc - i - 1] = self.RT(c) + self.rank(c.to_task)
                else:
                    Hd[lc - i - 1] = self.RT(c) + max(Hd[lc - i],
                                                      self.rank(c.to_task))
            r = Hu[-1]
            for i, c in enumerate(comms):
                hu = 0 if i == 0 else Hu[i - 1]
                x = 0 if i == 0 else X[i - 1]
                hd = 0 if i == len(comms) - 1 else Hd[i + 1]
                r1 = max(hu, self.rank(c.to_task), x + hd)
                r = min(r, r1)
        return r + self.RT(task) if comms else self.RT(task)


class M4Ranking(StaticSort):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        self.rank_c = {t: 0 for t in self.problem.tasks}

    @memo
    def rank_s(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank_s(c.to_task),
            reverse=True)
        if comms:
            lc = len(comms)
            X = [0 for _ in range(lc)]
            Hu = [0 for _ in range(lc)]
            Hd = [0 for _ in range(lc)]
            for i, c in enumerate(comms):
                if i == 0:
                    X[i] = self.RT(c)
                    Hu[i] = self.RT(c) + self.rank_s(c.to_task)
                else:
                    X[i] = X[i - 1] + self.RT(c)
                    Hu[i] = max(Hu[i - 1], X[i] + self.rank_s(c.to_task))
            for i, c in enumerate(reversed(comms)):
                if i == 0:
                    Hd[lc - i - 1] = self.RT(c) + self.rank_s(c.to_task)
                else:
                    Hd[lc - i - 1] = self.RT(c) + max(Hd[lc - i],
                                                      self.rank_s(c.to_task))
            r = Hu[-1]
            rt = None
            for i, c in enumerate(comms):
                hu = 0 if i == 0 else Hu[i - 1]
                x = 0 if i == 0 else X[i - 1]
                hd = 0 if i == len(comms) - 1 else Hd[i + 1]
                r1 = max(hu, self.rank_s(c.to_task), x + hd)
                if r1 < r:
                    r = r1
                    rt = c
            if rt: self.rank_c[rt.to_task] += 1
        return r + self.RT(task) if comms else self.RT(task)

    @memo
    def rank(self, task):
        for t in task.prevs():
            self.rank(t)
        return self.rank_s(task), self.rank_c[task]


class M5Ranking(StaticSort):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        self.critical_parent = {}
        for task in self.problem.tasks:
            self.rank_ft(task)
        # print(self.critical_parent)

    @memo
    @tryFT
    def rank_ft(self, task):
        self.critical_parent[task] = None
        ft_max = None
        comms = sorted(
            task.communications(COMM_INPUT),
            key=lambda c: self.rank_ft(c.from_task))
        for c in comms:
            ft = self.ft_without(comms, c)
            if ft_max is None or ft < ft_max:
                ft_max = ft
                self.critical_parent[task] = c.from_task
        return (ft_max or 0) + self.RT(task)

    def ft_without(self, comms, comm):
        ft = 0
        for c in comms:
            if c != comm:
                ft = max(ft, self.rank_ft(c.from_task)) + self.RT(c)
        return max(ft, self.rank_ft(comm.from_task))

    @memo
    def rank(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank(c.to_task),
            reverse=True)
        if comms:
            lc = len(comms)
            X = [0 for _ in range(lc)]
            Hu = [0 for _ in range(lc)]
            Hd = [0 for _ in range(lc)]
            for i, c in enumerate(comms):
                if i == 0:
                    X[i] = self.RT(c)
                    Hu[i] = self.RT(c) + self.rank(c.to_task)
                else:
                    X[i] = X[i - 1] + self.RT(c)
                    Hu[i] = max(Hu[i - 1], X[i] + self.rank(c.to_task))
            for i, c in enumerate(reversed(comms)):
                if i == 0:
                    Hd[lc - i - 1] = self.RT(c) + self.rank(c.to_task)
                else:
                    Hd[lc - i - 1] = self.RT(c) + max(Hd[lc - i],
                                                      self.rank(c.to_task))
            r = Hu[-1]
            for i, c in enumerate(comms):
                if task == self.critical_parent[c.to_task]:
                    hu = 0 if i == 0 else Hu[i - 1]
                    x = 0 if i == 0 else X[i - 1]
                    hd = 0 if i == len(comms) - 1 else Hd[i + 1]
                    r1 = max(hu, self.rank(c.to_task), x + hd)
                    if r1 < r:
                        r = r1
        return r + self.RT(task) if comms else self.RT(task)


class LLT4Ranking(GlobalSort):
    def existing_comm_time(self, task):
        if task not in self.placements:
            return 0
        else:
            machine = self.PL_m(task)
            return sum(
                c.runtime(self.bandwidth)
                for c in task.communications(COMM_OUTPUT)
                if c.to_task in self.placements
                and self.PL_m(c.to_task) != machine)

    @memo
    @tryFT
    def rank_d(self, task):
        comms = sorted(
            [(self.rank_d(c.from_task) + self.existing_comm_time(c.from_task),
              c.runtime(self.bandwidth))
             for c in task.communications(COMM_INPUT)])
        cft = 0
        for st, rt in comms:
            if st > cft:
                cft = st + rt
            else:
                cft += rt
        return cft + task.runtime(self.vm_type)

    @memo
    def rank_u(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank(c.to_task),
            reverse=True)
        rr = 0
        cst = 0
        for c in comms:
            cst += c.runtime(self.bandwidth)
            rr = max(rr, cst + self.rank_u(c.to_task))
        return rr + task.runtime(self.vm_type)

    @memo
    def rank(self, task):
        return self.rank_d(task) + self.rank_u(task) - task.runtime(
            self.vm_type)

    def rank_clean(self, task):
        memo_clean(self.rank_u)
        memo_clean(self.rank_d)
        memo_clean(self.rank)

    def log_sort(self, task):
        print("Chosen", task, self.rank(task))
        print("  ", [(t, self.rank(t), self.rank_d(t), self.rank_u(t))
                     for t in self.problem.tasks if t not in self.placements])


class LLT4_1Ranking(LLT4Ranking):
    def is_free_task(self, task):
        return not any(t in self.placements for t in task.prevs())

    @memo
    def rank_u(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank(c.to_task),
            reverse=True)
        rr = 0
        cst = 0
        lcst = 0
        for c in comms:
            if self.is_free_task(
                    c.to_task) and lcst < cst + c.runtime(self.bandwidth):
                lcst += self.rank_u(c.to_task)
                rr = max(rr, lcst)
            else:
                cst += c.runtime(self.bandwidth)
                rr = max(rr, cst + self.rank_u(c.to_task))
        return rr + task.runtime(self.vm_type)


class LLT4_2Ranking(LLT4_1Ranking):
    def comm_est(self, comm):
        from_task = comm.from_task
        return self.rank_d(from_task) + self.existing_comm_time(from_task)

    @memo
    @tryFT
    def rank_d(self, task):
        comms = sorted(task.communications(COMM_INPUT), key=self.comm_est)
        cft = 0
        for c in comms:
            cft = max(cft, self.comm_est(c)) + c.runtime(self.bandwidth)
        if comms and comms[-1].from_task not in self.placements:
            cft -= comms[-1].runtime(self.bandwidth)
        return cft + task.runtime(self.vm_type)


class LLT4_3Ranking(LLT4_2Ranking):
    @memo
    @tryFT
    def rank_d(self, task):
        comms = sorted(
            task.communications(COMM_INPUT), key=self.comm_est, reverse=True)
        lc = None
        for c in comms:
            if c.from_task not in self.placements:
                lc = c
                break
        comms.reverse()
        cft = 0
        for c in comms:
            if not lc or c != lc:
                cft = max(cft, self.comm_est(c)) + c.runtime(self.bandwidth)
        if lc: cft = max(cft, self.rank_d(c.from_task))
        return cft + task.runtime(self.vm_type)


class LLT4_4Ranking(LLT4_3Ranking):
    def is_critial_comm(self, comm):
        from_task = comm.from_task
        to_task = comm.to_task
        max_f = max(
            c.runtime(self.bandwidth)
            for c in from_task.communications(COMM_OUTPUT))
        max_t = max(
            c.runtime(self.bandwidth)
            for c in from_task.communications(COMM_OUTPUT))
        return max_f == max_t == comm.runtime(self.bandwidth)

    @memo
    @tryFT
    def rank_d(self, task):
        comms = sorted(
            task.communications(COMM_INPUT), key=self.comm_est, reverse=True)
        lc = None
        for c in comms:
            if c.from_task not in self.placements and self.is_critial_comm(c):
                lc = c
                break
        comms.reverse()
        cft = 0
        for c in comms:
            if not lc or c != lc:
                cft = max(cft, self.comm_est(c)) + c.runtime(self.bandwidth)
        if lc: cft = max(cft, self.rank_d(c.from_task))
        return cft + task.runtime(self.vm_type)


class LLT4_5Ranking(LLT4_4Ranking):
    @memo
    @tryFT
    def rank_d(self, task):
        comms = sorted(task.communications(COMM_INPUT), key=self.comm_est)
        critial_pre_comm = max(
            [
                c for c in task.communications(COMM_INPUT)
                if c.from_task not in self.placements
            ],
            key=lambda c: self.rank_d(c.from_task) + c.runtime(self.bandwidth),
            default=None)
        critial_pre_task = critial_pre_comm.from_task if critial_pre_comm else None
        cft = 0
        for c in comms:
            if c is not critial_pre_task:
                cft = max(cft, self.comm_est(c)) + c.runtime(self.bandwidth)
        if critial_pre_comm: cft = max(cft, self.rank_d(c.from_task))
        return cft + task.runtime(self.vm_type)


class LLT4_6Ranking(LLT4_5Ranking):
    def is_critial_comm(self, comm):
        from_task = comm.from_task
        to_task = comm.to_task
        return self.rank_d(from_task) + comm.runtime(self.bandwidth) == \
                max(self.rank_d(c.from_task) + c.runtime(self.bandwidth) for c in to_task.communications(COMM_INPUT))

    @memo
    def rank_u(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank(c.to_task),
            reverse=True)
        rr = 0
        cst = 0
        lcst = 0
        for c in comms:
            if self.is_critial_comm(
                    c) and lcst < cst + c.runtime(self.bandwidth):
                lcst += self.rank_u(c.to_task)
                rr = max(rr, lcst)
            else:
                cst += c.runtime(self.bandwidth)
                rr = max(rr, cst + self.rank_u(c.to_task))
        return rr + task.runtime(self.vm_type)


class LLT5Ranking(LLT4_3Ranking):
    @memo
    @tryFT
    def rank_d(self, task):
        comms = sorted(
            task.communications(COMM_INPUT),
            key=lambda c: self.rank_d(c.from_task),
            reverse=True)
        lc = None
        for c in comms:
            if c.from_task not in self.placements:
                lc = c
                break
        comms.reverse()
        cft = 0
        for c in comms:
            if not lc or c != lc:
                from_task = c.from_task
                st = max(cft, self.rank_d(from_task))
                if from_task in self.placements:
                    cft = self.PL_m(from_task).comm_finish_time_est(
                        c, st, COMM_OUTPUT)
                else:
                    cft = st + c.runtime(self.bandwidth)
        if lc:
            cft = max(cft, self.rank_d(lc.from_task))
        return cft + task.runtime(self.vm_type)


class CPRanking(GlobalSort):
    @memo
    def rank(self, task):
        return self.rank_d(task) + self.rank_u(task) - self.RT(task)

    def rank_clean(self, task):
        memo_clean(self.rank_u)
        memo_clean(self.rank_d)
        memo_clean(self.rank)

    def log_sort(self, task):
        print("Chosen", task, self.rank(task))
        print("  ", [(t, self.rank(t), self.rank_d(t), self.rank_u(t))
                     for t in self.problem.tasks if t not in self.placements])

    def comm_est(self, comm):
        est = self.rank_d(comm.from_task) + sum(
            c.runtime(self.bandwidth)
            for c in comm.from_task.communications(COMM_OUTPUT)
            if c in self.start_times)
        return est

    @memo
    @tryFT
    def rank_d(self, task):
        comms = sorted(task.communications(COMM_INPUT), key=self.comm_est)
        cft = 0
        for c in comms:
            cft = max(cft, self.comm_est(c)) + self.RT(c)
        return cft + self.RT(task)

    @memo
    def rank_u(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank_u(c.to_task),
            reverse=True)
        cst = self.RT(task)
        cft = cst
        for c in comms:
            cst += self.RT(c)
            cft = max(cft, cst + self.rank_u(c.to_task))
        return cft


class CP2Ranking(CPRanking):
    @memo
    @tryFT
    def rank_d(self, task):
        comms = sorted(task.communications(COMM_INPUT), key=self.comm_est)
        critial_pre_comm = max(
            [
                c for c in task.communications(COMM_INPUT)
                if c.from_task in self.placements
            ],
            key=lambda c: self.comm_est(c) + self.RT(c),
            default=None)
        cft = 0
        for c in comms:
            if c is not critial_pre_comm:
                cft = max(cft, self.comm_est(c)) + self.RT(c)
        if critial_pre_comm:
            cft = max(cft, self.rank_d(critial_pre_comm.from_task))
        return cft + self.RT(task)

    @memo
    def rank_u(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank_u(c.to_task),
            reverse=True)
        critial_post_comm = max(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.RT(c) + self.rank_u(c.to_task),
            default=None)
        cst = self.RT(task)
        if critial_post_comm:
            cft = cst + self.rank_u(critial_post_comm.to_task)
        else:
            cft = cst
        for c in comms:
            if c is not critial_post_comm:
                cst += self.RT(c)
                cft = max(cft, cst + self.rank_u(c.to_task))
        return cft


class CP3Ranking(CP2Ranking):
    @memo
    @tryFT
    def rank_d(self, task):
        comms = sorted(task.communications(COMM_INPUT), key=self.comm_est)
        cft = 0
        critical_comm = None
        for c in comms:
            if self.comm_est(c) > cft:
                critical_comm = c
                cft = self.comm_est(c) + self.RT(c)
            else:
                cft += self.RT(c)
                if c.from_task not in self.placements:
                    if critical_comm or self.RT(critical_comm) <= self.RT(c):
                        critical_comm = c
        if critical_comm:
            for c in comms:
                if c is not critical_comm:
                    cft = max(cft, self.comm_est(c) + self.RT(c))
            cft = max(cft, self.rank_d(critical_comm.from_task))
        return cft + self.RT(task)

    @memo
    def rank_u(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank_u(c.to_task),
            reverse=True)
        if comms:
            lc = len(comms)
            X = [0 for _ in range(lc)]
            Hu = [0 for _ in range(lc)]
            Hd = [0 for _ in range(lc)]
            for i, c in enumerate(comms):
                if i == 0:
                    X[i] = self.RT(c)
                    Hu[i] = self.RT(c) + self.rank_u(c.to_task)
                else:
                    X[i] = X[i - 1] + self.RT(c)
                    Hu[i] = max(Hu[i - 1], X[i] + self.rank_u(c.to_task))
            for i, c in enumerate(reversed(comms)):
                if i == 0:
                    Hd[lc - i - 1] = self.RT(c) + self.rank_u(c.to_task)
                else:
                    Hd[lc - i - 1] = self.RT(c) + max(Hd[lc - i],
                                                      self.rank_u(c.to_task))
            r = Hu[-1]
            bst = None
            for i, c in enumerate(comms):
                if self.is_free_task(c.to_task):
                    hu = 0 if i == 0 else Hu[i - 1]
                    x = 0 if i == 0 else X[i - 1]
                    hd = 0 if i == len(comms) - 1 else Hd[i + 1]
                    r1 = max(hu, self.rank_u(c.to_task), x + hd)
                    if r1 < r: bst = i, c, r1
                    r = min(r, r1)
        return r + self.RT(task) if comms else self.RT(task)

    def is_free_task(self, task):
        return not any(t in self.placements for t in task.prevs())
