from MrWSI.core.platform import Machine, COMM_INPUT, COMM_OUTPUT
from .base import Heuristic

from math import inf


class CASort(Heuristic):
    def _topsort(self):
        toporder = [t for t in self.problem.tasks if not t.in_degree]
        rids = [t.in_degree for t in self.problem.tasks]
        i = 0
        while i < self.problem.num_tasks:
            for t in toporder[i].succs():
                rids[t.id] -= 1
                if not rids[t.id]:
                    toporder.append(t)
            i += 1
        return toporder

    def _prepare_arrays(self):
        self.AT = [None] * self.problem.num_tasks
        self.ATO = [None] * self.problem.num_tasks
        self.RA = [None] * self.problem.num_tasks
        self.PT = [None] * self.problem.num_tasks
        self.PTO = [None] * self.problem.num_tasks
        self.RP = [None] * self.problem.num_tasks
        self.BM = [None] * self.problem.num_tasks

    def sort_tasks(self):
        self.toporder = self._topsort()
        self._prepare_arrays()
        self.ready_tasks = set(
            t for t in self.problem.tasks if not t.in_degree)
        self.rids = [t.in_degree for t in self.problem.tasks]
        while self.ready_tasks:
            task = self.select_task()
            yield task
            self.ready_tasks.remove(task)
            for t in task.succs():
                self.rids[t.id] -= 1
                if not self.rids[t.id]:
                    self.ready_tasks.add(t)

    def select_task(self):
        self.order_for_at = []
        for t in self.toporder:
            if self.is_placed(t):
                self.RA[t.id] = self.FT(t) + \
                    sum(self.RT(c) for c in t.communications(COMM_OUTPUT)
                        if c in self.start_times)
            else:
                self.calculate_AT(t)
                self.RA[t.id] = self.AT[t.id] + self.RT(t)
        for t in reversed(self.toporder):
            if not self.is_placed(t):
                self.calculate_PT(t)
                self.RP[t.id] = self.PT[t.id] + self.RT(t)

        dcs = [0] * self.problem.num_tasks
        for tx in self.ready_tasks:
            for ty in self.ready_tasks:
                if tx.id < ty.id and self.has_contention(tx, ty):
                    ftx = self.est_ft(tx, ty)
                    fty = self.est_ft(ty, tx)
                    if ftx < fty:
                        dcs[ty.id] -= 1
                    elif ftx > fty:
                        dcs[tx.id] -= 1
        task = max(self.ready_tasks, key=lambda t: (dcs[t.id], self.RP[t.id]))
        return task

    def has_contention(self, tx, ty):
        p0 = self.BM[tx.id]
        p1 = self.BM[ty.id]
        return (len(p0) == 1 or len(p1) == 1) and (p0 & p1) and \
            -self.RT(ty) < self.AT[ty.id] - self.AT[tx.id] < self.RT(tx)

    def est_ft(self, ti, tj):
        return min(
            self.AT[ti.id] + self.RT(ti) + self.RT(tj) +
            self.PT[ti.id] + self.PT[tj.id],
            self.AT[ti.id] + self.RT(ti) + self.RT(tj) +
            max(self.PT[ti.id], self.PTO[tj.id]),
            self.AT[ti.id] + self.RT(ti) + max(self.PTO[ti.id],
                                               self.RT(tj) + self.PT[tj.id]),
            max(self.AT[ti.id] + self.RT(ti) + self.PT[ti.id],
                self.ATO[tj.id] + self.RT(tj) + self.PT[tj.id])
        )

    def _at_divide_preds(self, comms):
        sm = {}
        st = []
        for i, c in enumerate(comms):
            if self.is_placed(c.from_task):
                m = self.PL_m(c.from_task)
                if m not in sm:
                    sm[m] = [i]
                else:
                    sm[m].append(i)
            else:
                st.append(i)
        return sm, st

    def _min2(self, at, ato, bmt, x, m):
        if x < at:
            return x, at, [m]
        elif x == at:
            bmt.append(m)
            return x, at, bmt
        elif x < ato:
            return at, x, bmt
        else:
            return at, ato, bmt

    def calculate_AT(self, task):
        comms = sorted(task.communications(COMM_INPUT),
                       key=lambda c: self.RA[c.from_task.id])
        A = [0] * len(comms)
        B = [0] * (len(comms) + 1)
        M = [0] * (len(comms) + 1)
        at_none = 0
        for i, c in enumerate(comms):
            ra = self.RA[c.from_task.id]
            tmp = max(at_none, ra)
            A[i] = tmp + self.RT(c) - at_none
            B[i] = tmp - ra
            at_none += A[i]
        B[-1] = inf
        k = inf
        for i in range(len(comms), -1, -1):
            if k >= B[i]:
                k = B[i]
                M[i] = i
            else:
                M[i] = M[i + 1]
        SM, ST = self._at_divide_preds(comms)
        if self.L > len(self.platform) or len(SM) < len(self.platform):
            at = at_none
            bmt = [-id(task)]
        else:
            at = inf
            bmt = []
        ato = inf
        for m, cs in SM.items():
            d = 0
            ma = 0
            t0 = 0
            for i in cs:
                t = comms[i].from_task
                if self.is_placed(t):
                    t0 = max(t0, self.FT(t))
                else:
                    t0 = max(t0, self.RA[t.id])
                if i < ma:
                    continue
                ma = M[i + 1]
                d += min(A[i], B[ma] - d)
            t0 = max(t0, at_none - d)
            t0, _ = m.earliest_slot_for_task(self.vm_type, task, t0)
            at, ato, bmt = self._min2(at, ato, bmt, t0, id(m))
        for i in ST:
            t = comms[i].from_task
            d = min(A[i], B[M[i + 1]])
            t0 = max(self.RA[t.id], at_none - d)
            at, ato, bmt = self._min2(at, ato, bmt, t0, id(t))
        self.AT[task.id] = at
        self.ATO[task.id] = ato
        self.BM[task.id] = set(bmt)

    def calculate_PT(self, task):
        comms = sorted(task.communications(COMM_OUTPUT),
                       key=lambda c: -self.RP[c.to_task.id])
        pto = 0
        tc = 0
        for c in comms:
            tc += self.RT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])

        pt = 0
        tc = 0
        tt = 0
        for c in comms:
            if id(task) in self.BM[c.to_task.id] and tt < tc + self.RT(c):
                tt += self.RP[c.to_task.id]
                pt = max(pt, tt)
            else:
                tc += self.RT(c)
                pt = max(pt, tc + self.RP[c.to_task.id])
        self.PT[task.id] = pt
        self.PTO[task.id] = pto

class CA2Sort(CASort):
    def sorted_in_comms(self, task):
        return sorted(task.communications(COMM_INPUT), key=lambda c: self.RA[c.from_task.id])

