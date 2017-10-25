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
        self._A = [None] * self.problem.num_tasks
        self._B = [None] * self.problem.num_tasks
        self._M = [None] * self.problem.num_tasks
        self._placed = [False] * self.problem.num_tasks

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
            self._placed[task.id] = True
            for t in task.succs():
                self.rids[t.id] -= 1
                if not self.rids[t.id]:
                    self.ready_tasks.add(t)

    def select_task(self):
        self.update_AT_and_PT()
        dcs = [0] * self.problem.num_tasks
        for tx in self.ready_tasks:
            for ty in self.ready_tasks:
                if tx.id < ty.id and self.has_contention(tx.id, ty.id):
                    ftx = self.est_ft(tx.id, ty.id)
                    fty = self.est_ft(ty.id, tx.id)
                    if ftx < fty:
                        dcs[ty.id] -= 1
                    elif ftx > fty:
                        dcs[tx.id] -= 1
        task = max(self.ready_tasks, key=lambda t: (dcs[t.id], self.RP[t.id]))
        return task

    def update_AT_and_PT(self):
        for t in self.toporder:
            if self._placed[t.id]:
                self.RA[t.id] = self.FT(t)
                for c in t.communications(COMM_OUTPUT):
                    if c in self.start_times:
                        self.RA[t.id] += self.CT(c)
            else:
                self.calculate_AT(t)
                self.RA[t.id] = self.AT[t.id] + self._RT[t.id]
        for t in reversed(self.toporder):
            if not self._placed[t.id]:
                self.calculate_PT(t)
                self.RP[t.id] = self.PT[t.id] + self._RT[t.id]

    def has_contention(self, tx, ty):
        p0 = self.BM[tx]
        p1 = self.BM[ty]
        return (len(p0) == 1 or len(p1) == 1) and (p0 & p1) and \
            -self._RT[ty] < self.AT[ty] - \
            self.AT[tx] < self._RT[tx]

    def est_ft(self, ti, tj):
        at_i = self.AT[ti]
        ato_i = self.ATO[ti]
        rt_i = self._RT[ti]
        pt_i = self.PT[ti]
        pto_i = self.PTO[ti]
        at_j = self.AT[tj]
        ato_j = self.ATO[tj]
        rt_j = self._RT[tj]
        pt_j = self.PT[tj]
        pto_j = self.PTO[tj]
        return min(
            at_i + rt_i + rt_j + pt_i + pt_j,
            at_i + rt_i + rt_j + max(pt_i, pto_j),
            at_i + rt_i + max(pto_i, rt_j + pt_j),
            max(at_i + rt_i + pt_i, ato_j + rt_j + pt_j)
        )

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
        A = self._A
        B = self._B

        at_none = 0
        for c in comms:
            t = c.from_task.id
            d = self.RA[t] - at_none
            if d > 0:
                A[t] = d + self.CT(c)
                B[t] = 0
            else:
                A[t] = self.CT(c)
                B[t] = -d
            at_none += A[t]

        Sm = []
        St = []
        k = inf
        for c in reversed(comms):
            t = c.from_task
            not_placed = not self._placed[t.id]
            mt = not_placed or self.PL_m(t)
            mt_in_sm = False
            for m, info in Sm:
                if m is mt:
                    mt_in_sm = True
                    d = min(info[2], A[t.id])
                    info[0] = max(info[0], self.FT(t))
                    info[1] += d
                    info[2] -= d
                else:
                    info[2] = min(info[2], B[t.id])
            if not_placed:
                d = min(k, A[t.id])
                St.append((t, d))
            elif not mt_in_sm:
                d = min(k, A[t.id])
                Sm.append((mt, [self.FT(t), d, k - d]))
            k = min(k, B[t.id])

        at = inf
        ato = inf
        bmt = []
        for m, (ft, d, _) in Sm:
            st = max(ft, at_none - d)
            st, _ = m.earliest_slot_for_task(self.vm_type, task, st)
            at, ato, bmt = self._min2(at, ato, bmt, st, id(m))
        for t, d in St:
            st = max(self.RA[t.id], at_none - d)
            at, ato, bmt = self._min2(at, ato, bmt, st, id(t))
        if self.L > len(self.platform) or len(Sm) < len(self.platform):
            at, ato, bmt = self._min2(at, ato, bmt, at_none, -id(task))

        self.AT[task.id] = at
        self.ATO[task.id] = ato
        self.BM[task.id] = set(bmt)

    def calculate_PT(self, task):
        comms = sorted(task.communications(COMM_OUTPUT),
                       key=lambda c: -self.RP[c.to_task.id])
        pto = 0
        tc = 0
        for c in comms:
            tc += self.CT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])

        pt = 0
        tc = 0
        tt = 0
        for c in comms:
            if id(task) in self.BM[c.to_task.id] and tt < tc + self.CT(c):
                tt += self.RP[c.to_task.id]
                pt = max(pt, tt)
            else:
                tc += self.CT(c)
                pt = max(pt, tc + self.RP[c.to_task.id])
        self.PT[task.id] = pt
        self.PTO[task.id] = pto


class CASimpleAT(CASort):
    def calculate_AT(self, task):
        comms = sorted(task.communications(COMM_INPUT),
                       key=lambda c: self.RA[c.from_task.id])
        A = self._A
        B = self._B
        M = self._M

        at_none = 0
        k = inf
        for c in comms:
            t = c.from_task.id
            d = self.RA[t] - at_none
            if d > 0:
                A[t] = d + self.CT(c)
                B[t] = 0
            else:
                A[t] = self.CT(c)
                B[t] = -d
            at_none += A[t]
        for c in reversed(comms):
            t = c.from_task.id
            M[t] = k
            k = min(B[t], k)

        Sm = {}
        St = []
        for c in comms:
            t = c.from_task
            if self._placed[t.id]:
                m = self.PL_m(t)
                if m not in Sm:
                    Sm[m] = [M[t.id], A[t.id], self.FT(t)]
                else:
                    Sm[m][0] = M[t.id]
                    Sm[m][1] += A[t.id]
                    Sm[m][2] = max(Sm[m][2], self.FT(t))
            else:
                St.append(t)

        if self.L > len(self.platform) or len(Sm) < len(self.platform):
            at = at_none
            bmt = [-id(task)]
        else:
            at = inf
            bmt = []
        ato = inf

        for m, (mt, sa, ft) in Sm.items():
            st = max(ft, at_none - min(mt, sa))
            st, _ = m.earliest_slot_for_task(self.vm_type, task, st)
            at, ato, bmt = self._min2(at, ato, bmt, st, id(m))

        for t in St:
            st = max(self.RA[t.id], at_none - min(A[t.id], M[t.id]))
            at, ato, bmt = self._min2(at, ato, bmt, st, id(t))

        self.AT[task.id] = at
        self.ATO[task.id] = ato
        self.BM[task.id] = set(bmt)


class CAMoreCompare(Heuristic):
    def select_task(self):
        self.update_AT_and_PT()
        dcs = [0] * self.problem.num_tasks
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and\
                        not (self.rids[tx] and self.rids[ty]) and\
                        self.has_contention(tx, ty):
                    ftx = self.est_ft(tx, ty)
                    fty = self.est_ft(ty, tx)
                    if ftx < fty:
                        dcs[ty] -= 1
                    elif ftx > fty:
                        dcs[tx] -= 1
        task = max(self.ready_tasks, key=lambda t: (dcs[t.id], self.RP[t.id]))
        return task


class CAFit(Heuristic):
    def default_fitness(self):
        return inf, inf

    def fitness(self, task, machine, comm_pls, st):
        rt_x = self._RT[task.id]
        ft_x = st + rt_x
        pt_x = self.PT[task.id]
        pto_x = self.PTO[task.id]
        wft = ft_x + pt_x
        for t in self.ready_tasks:
            rt_y = self._RT[t.id]
            at_y = self.AT[t.id]
            ato_y = self.ATO[t.id]
            pt_y = self.PT[t.id]
            pto_y = self.PTO[t.id]

            if t != task:
                pt = self.BM[t.id]
                if len(pt) == 1 and id(machine) in pt and\
                        -rt_x < st - at_y < rt_y:
                    wft = max(wft, min(
                        ft_x + rt_y + pt_x + pt_y,
                        ft_x + rt_y + max(pt_x, pto_y),
                        ft_x + max(pto_x, rt_y + pt_y),
                        max(ft_x + pt_x, ato_y + rt_y + pt_y)))
                else:
                    wft = max(wft, at_y + rt_y + pt_y)
        return wft, ft_x


class CAFit2(Heuristic):
    def default_fitness(self):
        return inf, inf

    def fitness(self, task, machine, comm_pls, st):
        rt_x = self._RT[task.id]
        ft_x = st + rt_x
        pt_x = self.PT[task.id]
        pto_x = self.PTO[task.id]
        wft = ft_x + pt_x
        for t in range(self.problem.num_tasks):
            if self._placed[t]:
                continue
            rt_y = self._RT[t]
            at_y = self.AT[t]
            ato_y = self.ATO[t]
            pt_y = self.PT[t]
            pto_y = self.PTO[t]

            if t != task.id:
                pt = self.BM[t]
                if len(pt) == 1 and id(machine) in pt and\
                        -rt_x < st - at_y < rt_y:
                    wft = max(wft, min(
                        ft_x + rt_y + pt_x + pt_y,
                        ft_x + rt_y + max(pt_x, pto_y),
                        ft_x + max(pto_x, rt_y + pt_y),
                        max(ft_x + pt_x, ato_y + rt_y + pt_y)))
                else:
                    wft = max(wft, at_y + rt_y + pt_y)
        return wft, ft_x
