from MrWSI.core.platform import COMM_INPUT, COMM_OUTPUT
from .base import Heuristic
from .sorting import memo, memo_clean, tryFT
from copy import copy


class PSort(Heuristic):
    def prepare_selection(self):
        memo_clean(self.before_rt)
        memo_clean(self.before_rt_alt)
        memo_clean(self.after_rt)
        memo_clean(self.after_rt_alt)
        memo_clean(self.rank_before)
        memo_clean(self.rank_after)
        for task in self.problem.tasks:
            self.rank_before(task)

    def estimated_span(self, tx, ty):
        t0 = self.before_rt(tx)
        x0 = self.RT(tx)
        y0 = self.after_rt(tx)
        y0a = self.after_rt_alt(tx)
        t1 = self.before_rt(ty)
        t1a = self.before_rt_alt(ty)
        x1 = self.RT(ty)
        y1 = self.after_rt(ty)
        y1a = self.after_rt_alt(ty)
        # print(tx, t0, "-", x0, y0, y0a)
        # print(ty, t1, t1a, x1, y1, y1a)
        return min(
            max(t0 + x0 + y0, t1a + x1 + y1),
            max(t0 + x0 + y0, max(t1, t0 + x0) + x1 + y1a),
            max(t0 + x0 + y0a, max(t1, t0 + x0) + x1 + y1),
            max(t1, t0 + x0 + y0) + x1 + y1)

    def better_place_first(self, tx, ty):
        if self.critical_parent[tx] & self.critical_parent[ty]:
            sx = self.estimated_span(tx, ty)
            sy = self.estimated_span(ty, tx)
            # print(tx, ty, sx, sy)
            return sx < sy
        else:
            return False

    def default_rank(self, task):
        return self.rank_after(task)

    @memo
    def after_rt(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank_after(c.to_task),
            reverse=True)
        if comms:
            lc = len(comms)
            X = [0 for _ in range(lc)]
            Hu = [0 for _ in range(lc)]
            Hd = [0 for _ in range(lc)]
            for i, c in enumerate(comms):
                if i == 0:
                    X[i] = self.RT(c)
                    Hu[i] = self.RT(c) + self.rank_after(c.to_task)
                else:
                    X[i] = X[i - 1] + self.RT(c)
                    Hu[i] = max(Hu[i - 1], X[i] + self.rank_after(c.to_task))
            for i, c in enumerate(reversed(comms)):
                if i == 0:
                    Hd[lc - i - 1] = self.RT(c) + self.rank_after(c.to_task)
                else:
                    Hd[lc
                       - i - 1] = self.RT(c) + max(Hd[lc - i],
                                                   self.rank_after(c.to_task))
            r = Hu[-1]
            for i, c in enumerate(comms):
                if task in self.critical_parent[c.to_task]:
                    hu = 0 if i == 0 else Hu[i - 1]
                    x = 0 if i == 0 else X[i - 1]
                    hd = 0 if i == len(comms) - 1 else Hd[i + 1]
                    r1 = max(hu, self.rank_after(c.to_task), x + hd)
                    if r1 < r:
                        r = r1
        return r if comms else 0

    @memo
    def after_rt_alt(self, task):
        comms = sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.rank_after(c.to_task),
            reverse=True)
        r = 0
        cst = 0
        for c in comms:
            cst += self.RT(c)
            r = max(r, cst + self.rank_after(c.to_task))
        return r

    def ft_without(self, comms, comm):
        ft = 0
        for c in comms:
            if not (c == comm or
                    (comm in self.placements and
                     self.placements.get(c, None) is self.placements[comm])):
                ft = max(ft, self.rank_before(c.from_task)) + self.RT(c)
        return max(ft, self.rank_before(comm.from_task))

    @memo
    def before_rt(self, task):
        comms = sorted(
            task.communications(COMM_INPUT),
            key=lambda c: self.rank_before(c.from_task))
        self.critical_parent[task] = set()
        self.best_pl_gauss[task] = set()
        ft_max = None
        for c in comms:
            ft = self.ft_without(comms, c)
            if ft_max is None or ft < ft_max:
                ft_max = ft
                self.critical_parent[task] = set([c.from_task])
                if c.from_task in self.placements:
                    self.best_pl_gauss[task] = set([self.PL_m(c.from_task)])
            elif ft == ft_max:
                self.critical_parent[task].add(c.from_task)
                if c.from_task in self.placements:
                    self.best_pl_gauss[task].add(self.PL_m(c.from_task))
        return ft_max or 0

    @memo
    def before_rt_alt(self, task):
        comms = sorted(
            task.communications(COMM_INPUT),
            key=lambda c: self.rank_before(c.from_task))
        ft = 0
        for c in comms:
            ft = max(ft, self.rank_before(c.from_task)) + self.RT(c)
        return ft

    @tryFT
    @memo
    def rank_before(self, task):
        return self.before_rt(task) + self.RT(task)

    @memo
    def rank_after(self, task):
        return self.after_rt(task) + self.RT(task)

    def select_task(self):
        dominated_count = {t: 0 for t in self.ready_tasks}
        for task in self.ready_tasks:
            for t in self.ready_tasks:
                if t is not task and self.better_place_first(t, task):
                    dominated_count[task] += 1
        min_dc = min(dominated_count.values())
        return max(
            [t for t, c in dominated_count.items() if c == min_dc],
            key=self.default_rank)

    def sort_tasks(self):
        self.critical_parent = {}
        self.best_pl_gauss = {}
        self.rem_in_deps = {t: t.in_degree for t in self.problem.tasks}
        self.ready_tasks = set(t for t in self.problem.tasks
                               if not t.in_degree)
        while self.ready_tasks:
            self.prepare_selection()
            task = self.select_task()
            self.ready_tasks.remove(task)
            for succ in task.succs():
                self.rem_in_deps[succ] -= 1
                if not self.rem_in_deps[succ]:
                    self.ready_tasks.add(succ)
            yield task
