from MrWSI.core.platform import Machine, COMM_INPUT, COMM_OUTPUT
from .base import Heuristic
from .ca import *


from math import inf
from heapq import heapify, heappush, heappop


class EFT_RankU(Heuristic):
    def sorted_in_comms(self, task):
        return task.communications(COMM_INPUT)

    def sort_tasks(self):
        self._ranks = [0] * self.problem.num_tasks
        for task in reversed(self._topsort()):
            self._ranks[task.id] = max([self._ranks[c.to_task.id] + self.CT(c)
                                        for c in task.communications(COMM_OUTPUT)],
                                       default=0) + (self._RT[task.id] or 0.1)
        return sorted(self.problem.tasks, key=lambda t: self._ranks[t.id], reverse=True)

    def default_fitness(self):
        return inf

    def fitness(self, task, machine, comm_pls, st):
        return st + self._RT[task.id]


class CA_Simple(Heuristic):
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
        self._dcs = [0] * self.problem.num_tasks
        self.toporder = self._topsort()

    def sort_tasks(self):
        self._prepare_arrays()
        self.ready_tasks = set(
            t for t in self.problem.tasks if not t.in_degree)
        self.rids = [t.in_degree for t in self.problem.tasks]
        for task in reversed(self.toporder):
            self.initialise_PT(task)
        while self.ready_tasks:
            for t in self.ready_tasks:
                self.calculate_AT(t)
            task = self.select_task()
            yield task
            self.ready_tasks.remove(task)
            self._placed[task.id] = True
            self.RA[task.id] = self.FT(task)
            for c in task.communications(COMM_INPUT):
                if c in self.start_times:
                    self.RA[c.from_task.id] += self.CT(c)
            for t in task.succs():
                self.rids[t.id] -= 1
                if not self.rids[t.id]:
                    self.ready_tasks.add(t)

    def select_task(self):
        for i in range(self.problem.num_tasks):
            self._dcs[i] = 0
        for tx in self.ready_tasks:
            for ty in self.ready_tasks:
                if tx.id < ty.id and self.has_contention(tx.id, ty.id):
                    ftx = self.est_ft(tx.id, ty.id, self.AT[tx.id])
                    fty = self.est_ft(ty.id, tx.id, self.AT[ty.id])
                    if ftx < fty:
                        self._dcs[ty.id] -= 1
                    elif ftx > fty:
                        self._dcs[tx.id] -= 1
        task = max(self.ready_tasks, key=lambda t: (
            self._dcs[t.id], self.RP[t.id]))
        return task

    def has_contention(self, tx, ty):
        p0 = self.BM[tx]
        p1 = self.BM[ty]
        return (len(p0) == 1 and len(p1) == 1) and (p0 & p1) and \
            -self._RT[ty] < self.AT[ty] - self.AT[tx] < self._RT[tx]

    def est_ft(self, ti, tj, st_i):
        rt_i = self._RT[ti]
        pt_i = self.PT[ti]
        pto_i = self.PTO[ti]
        ato_j = self.ATO[tj]
        rt_j = self._RT[tj]
        pt_j = self.PT[tj]
        pto_j = self.PTO[tj]
        return min(
            st_i + rt_i + rt_j + pt_i + pt_j,
            st_i + rt_i + rt_j + max(pt_i, pto_j),
            st_i + rt_i + max(pto_i, rt_j + pt_j),
            max(st_i + rt_i + pt_i, ato_j + rt_j + pt_j)
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
        k = inf
        for c in reversed(comms):
            t = c.from_task
            mt = self.PL_m(t)
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
            if not mt_in_sm:
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
        if self.L > len(self.platform) or len(Sm) < len(self.platform):
            at, ato, bmt = self._min2(at, ato, bmt, at_none, -id(task))

        self.AT[task.id] = at
        self.ATO[task.id] = ato
        self.BM[task.id] = set(bmt)

    def initialise_PT(self, task):
        comms = sorted(task.communications(COMM_OUTPUT),
                       key=lambda c: self.RP[c.to_task.id], reverse=True)
        pto, tc = 0, 0
        for c in comms:
            tc += self.CT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])
        pt, tc, tt = 0, 0, 0
        for c in comms:
            if tt < tc + self.CT(c):
                tt += self.RP[c.to_task.id]
                pt = max(pt, tt)
            else:
                tc += self.CT(c)
                pt = max(pt, tc + self.RP[c.to_task.id])
        self.PT[task.id] = pt
        self.PTO[task.id] = pto
        self.RP[task.id] = pt + self._RT[task.id]

    def default_fitness(self):
        return [inf]

    def fitness(self, task, machine, comm_pls, st):
        taskid = task.id
        rt_x = self._RT[taskid]
        pt_x = self.PT[taskid]
        wft = [st + rt_x + pt_x]
        for t in self.ready_tasks:
            tid = t.id
            at_y = self.AT[tid]
            rt_y = self._RT[tid]
            if tid != taskid:
                bm_t = self.BM[tid]
                if len(bm_t) == 1 and id(machine) in bm_t and -rt_x < st - at_y < rt_y:
                    wft.append(self.est_ft(taskid, tid, st))
                else:
                    wft.append(at_y + rt_y + self.PT[tid])
        return sorted(wft, reverse=True)


class CA_Simple2(CA_Simple):
    def _prepare_arrays(self):
        self.AT = [None] * self.problem.num_tasks
        self.ATO = [None] * self.problem.num_tasks
        self.RA = [None] * self.problem.num_tasks
        self.PT = [None] * self.problem.num_tasks
        self.PT_l = [None] * self.problem.num_tasks
        self.PT_r = [None] * self.problem.num_tasks
        self.PT_c = [None] * self.problem.num_tasks
        self.LS = [None] * self.problem.num_tasks
        self.PTO = [None] * self.problem.num_tasks
        self.PTO_c = [None] * self.problem.num_tasks
        self.RP = [None] * self.problem.num_tasks
        self.BM = [None] * self.problem.num_tasks
        self._A = [None] * self.problem.num_tasks
        self._B = [None] * self.problem.num_tasks
        self._M = [None] * self.problem.num_tasks
        self._placed = [False] * self.problem.num_tasks
        self._dcs = [0] * self.problem.num_tasks
        self.toporder = self._topsort()

    def initialise_PT(self, task):
        comms = sorted(task.communications(COMM_OUTPUT),
                       key=lambda c: self.RP[c.to_task.id], reverse=True)
        pto, tc = 0, 0
        for c in comms:
            tc += self.CT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])
        self.PTO[task.id] = pto
        self.PTO_c[task.id] = tc

        tc, tt, tr = 0, 0, 0
        ls = set([task.id])
        for c in comms:
            tid = c.to_task.id
            if tt < tc + self.CT(c):
                tt += self._RT[tid]
                tr = max(tr, tt + self.PT_r[tid])
                tc = max(tc, tt + self.PT_c[tid])
                tt += self.PT_l[tid]
                # for tl in self.LS[tid]:
                # if tl in ls:
                # tt -= self._RT[tl]
                # else:
                # ls.add(tl)
            else:
                tc += self.CT(c)
                tr = max(tr, tc + self.RP[tid])

        self.PT[task.id] = max(tt, tr)
        self.PT_l[task.id] = tt
        self.PT_r[task.id] = tr
        self.PT_c[task.id] = tc
        self.RP[task.id] = self.PT[task.id] + self._RT[task.id]
        self.LS[task.id] = ls

    def est_ft(self, ti, tj, st_i):
        rt_i = self._RT[ti]
        pt_i = self.PT[ti]
        ptl_i = self.PT_l[ti]
        ptr_i = self.PT_r[ti]
        ptc_i = self.PT_c[ti]
        pto_i = self.PTO[ti]
        ptoc_i = self.PTO_c[ti]
        ato_j = self.ATO[tj]
        rt_j = self._RT[tj]
        pt_j = self.PT[tj]
        ptl_j = self.PT_l[tj]
        ptr_j = self.PT_r[tj]
        ptc_j = self.PT_c[tj]
        pto_j = self.PTO[tj]
        ptoc_j = self.PTO_c[tj]
        lc = sum(self._RT[t] for t in (self.LS[ti] & self.LS[tj]))
        # lc = 0
        ret = inf
        if rt_j < ptc_i:
            ret = min(ret, st_i + rt_i + max(rt_j + ptl_i + ptl_j - lc,
                                             min(max(ptr_i, ptc_i + ptr_j),
                                                 max(ptr_i + ptc_j, rt_j + ptr_j))))
            ret = min(ret, st_i + rt_i + max(rt_j + ptl_j,
                                             min(max(ptr_i, ptc_i + pto_j),
                                                 max(rt_j + pto_j, ptr_i + ptoc_j))))
        else:
            ret = min(ret,
                      st_i + rt_i + max(ptr_i,
                                        rt_j + ptr_j,
                                        rt_j + ptl_i + ptl_j - lc))
            ret = min(ret,
                      st_i + rt_i + rt_j + max(rt_j + ptl_i,
                                               ptr_i,
                                               rt_j + pto_j))
        if ptoc_i > rt_j:
            ret = min(ret,
                      st_i + rt_i + max(rt_j + ptl_j,
                                        min(max(pto_i, ptoc_i + ptr_j),
                                            max(ptoc_i + ptc_j, rt_j + ptr_j))))
        else:
            ret = min(ret,
                      st_i + rt_i + max(rt_j + ptl_j,
                                        pto_i,
                                        rt_j + ptr_j))
        ret = min(ret, max(st_i + rt_i + pt_i, ato_j + rt_j + pt_j))
        return ret


class CA2(CAMoreCompare):
    def _prepare_arrays(self):
        self.AT = [None] * self.problem.num_tasks
        self.ATO = [None] * self.problem.num_tasks
        self.RA = [None] * self.problem.num_tasks
        self.PT = [None] * self.problem.num_tasks
        self.PT_l = [None] * self.problem.num_tasks
        self.PT_r = [None] * self.problem.num_tasks
        self.PT_c = [None] * self.problem.num_tasks
        self.LS = [None] * self.problem.num_tasks
        self.PTO = [None] * self.problem.num_tasks
        self.PTO_c = [None] * self.problem.num_tasks
        self.RP = [None] * self.problem.num_tasks
        self.BM = [None] * self.problem.num_tasks
        self._A = [None] * self.problem.num_tasks
        self._B = [None] * self.problem.num_tasks
        self._M = [None] * self.problem.num_tasks
        self._placed = [False] * self.problem.num_tasks
        self._dcs = [0] * self.problem.num_tasks
        self.toporder = self._topsort()

    def has_contention(self, tx, ty):
        p0 = self.BM[tx]
        p1 = self.BM[ty]
        return (len(p0) == 1 and len(p1) == 1) and (p0 & p1) and \
            -self._RT[ty] < self.AT[ty] - self.AT[tx] < self._RT[tx]

    def default_fitness(self):
        return [inf]

    def fitness(self, task, machine, comm_pls, st):
        taskid = task.id
        rt_x = self._RT[taskid]
        pt_x = self.PT[taskid]
        wft = [st + rt_x + pt_x]
        for t in self.ready_tasks:
            tid = t.id
            at_y = self.AT[tid]
            rt_y = self._RT[tid]
            if tid != taskid:
                bm_t = self.BM[tid]
                if len(bm_t) == 1 and id(machine) in bm_t and -rt_x < st - at_y < rt_y:
                    wft.append(self.est_ft(taskid, tid, st))
                else:
                    wft.append(at_y + rt_y + self.PT[tid])
        return sorted(wft, reverse=True)

    def est_ft(self, ti, tj, st_i=None):
        if st_i is None:
            st_i = self.AT[ti]
        rt_i = self._RT[ti]
        pt_i = self.PT[ti]
        ptl_i = self.PT_l[ti]
        ptr_i = self.PT_r[ti]
        ptc_i = self.PT_c[ti]
        pto_i = self.PTO[ti]
        ptoc_i = self.PTO_c[ti]
        ato_j = self.ATO[tj]
        rt_j = self._RT[tj]
        pt_j = self.PT[tj]
        ptl_j = self.PT_l[tj]
        ptr_j = self.PT_r[tj]
        ptc_j = self.PT_c[tj]
        pto_j = self.PTO[tj]
        ptoc_j = self.PTO_c[tj]
        lc = sum(self._RT[t] for t in (self.LS[ti] & self.LS[tj]))
        # lc = 0
        ret = inf
        if rt_j < ptc_i:
            ret = min(ret, st_i + rt_i + max(rt_j + ptl_i + ptl_j - lc,
                                             min(max(ptr_i, ptc_i + ptr_j),
                                                 max(ptr_i + ptc_j, rt_j + ptr_j))))
            ret = min(ret, st_i + rt_i + max(rt_j + ptl_j,
                                             min(max(ptr_i, ptc_i + pto_j),
                                                 max(rt_j + pto_j, ptr_i + ptoc_j))))
        else:
            ret = min(ret,
                      st_i + rt_i + max(ptr_i,
                                        rt_j + ptr_j,
                                        rt_j + ptl_i + ptl_j - lc))
            ret = min(ret,
                      st_i + rt_i + rt_j + max(rt_j + ptl_i,
                                               ptr_i,
                                               rt_j + pto_j))
        if ptoc_i > rt_j:
            ret = min(ret,
                      st_i + rt_i + max(rt_j + ptl_j,
                                        min(max(pto_i, ptoc_i + ptr_j),
                                            max(ptoc_i + ptc_j, rt_j + ptr_j))))
        else:
            ret = min(ret,
                      st_i + rt_i + max(rt_j + ptl_j,
                                        pto_i,
                                        rt_j + ptr_j))
        ret = min(ret, max(st_i + rt_i + pt_i, ato_j + rt_j + pt_j))
        return ret

    def calculate_PT(self, task):
        comms = sorted(task.communications(COMM_OUTPUT),
                       key=lambda c: self.RP[c.to_task.id], reverse=True)
        pto, tc = 0, 0
        for c in comms:
            tc += self.CT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])
        self.PTO[task.id] = pto
        self.PTO_c[task.id] = tc

        tc, tt, tr = 0, 0, 0
        ls = set([task.id])
        for c in comms:
            tid = c.to_task.id
            if id(task) in self.BM[tid] and tt < tc + self.CT(c):
                tt += self._RT[tid]
                tr = max(tr, tt + self.PT_r[tid])
                tc = max(tc, tt + self.PT_c[tid])
                tt += self.PT_l[tid]
                # for tl in self.LS[tid]:
                # if tl in ls:
                # tt -= self._RT[tl]
                # else:
                # ls.add(tl)
            else:
                tc += self.CT(c)
                tr = max(tr, tc + self.RP[tid])
        self.PT[task.id] = max(tt, tr)
        self.PT_l[task.id] = tt
        self.PT_r[task.id] = tr
        self.PT_c[task.id] = tc
        self.RP[task.id] = self.PT[task.id] + self._RT[task.id]
        self.LS[task.id] = ls


class CASelectBase(CAFit3, CAMoreCompare):
    def select_task(self):
        raise NotImplementedError


class CASelect2(CASelectBase):
    def select_task(self):
        self.update_AT_and_PT()
        self._dcs = [0] * self.problem.num_tasks
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and\
                        not (self.rids[tx] and self.rids[ty]) and\
                        self.has_contention(tx, ty):
                    ftx = self.est_ft(tx, ty)
                    fty = self.est_ft(ty, tx)
                    if ftx < fty:
                        self._dcs[tx] += 1
                    elif ftx > fty:
                        self._dcs[ty] += 1
        task = max(self.ready_tasks, key=lambda t: (
            self._dcs[t.id], self.RP[t.id]))
        return task


class CASelect3(CASelectBase):
    def select_task(self):
        self.update_AT_and_PT()
        self._dcs = [0] * self.problem.num_tasks
        self._m2 = [0] * self.problem.num_tasks
        for tx in self.ready_tasks:
            self._m2[tx.id] = self.AT[tx.id] + self._RT[tx.id] + self.PT[tx.id]
            for ty in self.problem.tasks:
                if not self._placed[ty.id]:
                    if self.has_contention(tx.id, ty.id):
                        self._m2[tx.id] = max(
                            self._m2[tx.id], self.est_ft(tx.id, ty.id))
                    else:
                        self._m2[tx.id] = max(self._m2[tx.id],
                                              self.AT[ty.id] + self._RT[ty.id] + self.PT[ty.id])
        task = max(self.ready_tasks, key=lambda t: (
            -self._m2[t.id], self.RP[t.id]))
        return task


class CASelect4(CASelectBase):
    def select_task(self):
        self.update_AT_and_PT()
        pool = []
        for tx in self.ready_tasks:
            for ty in self.ready_tasks:
                if tx != ty:
                    if self.has_contention(tx.id, ty.id):
                        pool.append((tx, ty, self.est_ft(tx.id, ty.id)))
        pool.sort(key=lambda x: x[-1], reverse=True)
        removed = [False] * self.problem.num_tasks
        for tx, ty, ft in pool:
            if removed[tx.id] or removed[ty.id]:
                continue
            else:
                removed[tx.id] = True
        ts = [t for t in self.ready_tasks if not removed[t.id]]
        return max(ts, key=lambda t: self.RP[t.id])


class CASelect5(CASelectBase):
    def select_task(self):
        self.update_AT_and_PT()
        m0 = [0] * self.problem.num_tasks
        m1 = [0] * self.problem.num_tasks
        for tx in self.ready_tasks:
            for ty in self.ready_tasks:
                if tx != ty:
                    if self.has_contention(tx.id, ty.id):
                        ft = self.est_ft(tx.id, ty.id)
                        m0[ty.id] = max(m0[ty.id], ft)
                        m1[tx.id] = max(m1[tx.id], ft)
        return max(self.ready_tasks,
                   key=lambda t: (-m1[t.id], self.RP[t.id]))


class CASelectU2(CASelectBase):
    def select_task(self):
        self.update_AT_and_PT()
        self._dcs = [0] * self.problem.num_tasks
        self._m2 = [0] * self.problem.num_tasks
        task = max(self.ready_tasks, key=lambda t: self.RP[t.id])
        return task


class CARankU(CASelectBase):
    def sort_tasks(self):
        self._prepare_arrays()
        self._ranks = [0] * self.problem.num_tasks
        self.ready_tasks = set(
            t for t in self.problem.tasks if not t.in_degree)
        self.rids = [t.in_degree for t in self.problem.tasks]
        for task in reversed(self._topsort()):
            self._ranks[task.id] = max([self._ranks[c.to_task.id] + self.CT(c)
                                        for c in task.communications(COMM_OUTPUT)],
                                       default=0) + (self._RT[task.id] or 0.1)
        for task in sorted(self.problem.tasks, key=lambda t: self._ranks[t.id], reverse=True):
            self.update_AT_and_PT()
            yield task
            self.ready_tasks.remove(task)
            self._placed[task.id] = True
            for t in task.succs():
                self.rids[t.id] -= 1
                if not self.rids[t.id]:
                    self.ready_tasks.add(t)


class CA3(CANewRA, CAMoreCompare):
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
        self._dcs = [0] * self.problem.num_tasks
        self.toporder = self._topsort()
        self._ctasks = [set() for _ in self.problem.tasks]
        self._cdeps = [set() for _ in self.problem.tasks]

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
        self._cdeps[task.id] = set()
        self._ctasks[task.id] = set()
        for c in comms:
            if tt < tc + self.CT(c):
                # if id(task) in self.BM[c.to_task.id] and tt < tc + self.CT(c):
                tt += self.RP[c.to_task.id]
                pt = max(pt, tt)
                self._cdeps[task.id].update(self._cdeps[c.to_task.id])
                self._cdeps[task.id].add(c)
                self._ctasks[task.id].update(self._ctasks[c.to_task.id])
                self._ctasks[task.id].add(c.to_task)
            else:
                tc += self.CT(c)
                pt = max(pt, tc + self.RP[c.to_task.id])
        self.PT[task.id] = pt
        self.PTO[task.id] = pto

    def est_ft(self, ti, tj, at_i=None):
        # print("=>", ti, tj, at_i)
        if not at_i:
            at_i = self.AT[ti]
        rt_i = self._RT[ti]
        pt_i = self.PT[ti]
        pto_i = self.PTO[ti]
        at_j = self.AT[tj]
        ato_j = self.ATO[tj]
        rt_j = self._RT[tj]
        pt_j = self.PT[tj]
        pto_j = self.PTO[tj]
        ft = min(
            at_i + rt_i + rt_j + pt_i + pt_j,
            at_i + rt_i + rt_j + max(pt_i, pto_j),
            at_i + rt_i + max(pto_i, rt_j + pt_j),
        )
        cdeps_i = [d for d in self._cdeps[ti]
                   if d not in self._cdeps[tj] and d.to_task in self._ctasks[tj]]
        cdeps_j = [d for d in self._cdeps[tj]
                   if d not in self._cdeps[ti] and d.to_task in self._ctasks[ti]]
        # print(ti, tj, self._cdeps[ti], self._cdeps[tj], self._ctasks[ti], self._ctasks[tj], cdeps_i, cdeps_j)
        ctasks = {}
        for d in cdeps_i:
            if d.to_task.id not in ctasks:
                ctasks[d.to_task.id] = [0, 0]
            ctasks[d.to_task.id][0] += self.CT(d)
        for d in cdeps_j:
            if d.to_task.id not in ctasks:
                ctasks[d.to_task.id] = [0, 0]
            ctasks[d.to_task.id][1] += self.CT(d)
        delta_i = sum(x for x, y in ctasks.values() if x <= y)
        delta_j = sum(y for x, y in ctasks.values() if y <= x)
        return min(ft,
                   max(at_i + rt_i + pt_i + delta_i,
                       ato_j + rt_j + pt_j + delta_j))

    def default_fitness(self):
        return inf, inf, [inf]

    def fitness(self, task, machine, comm_pls, st):
        taskid = task.id
        rt_x = self._RT[taskid]
        pt_x = self.PT[taskid]
        wft = [st - self.AT[task.id]]
        ftm = st + rt_x + pt_x
        for t in self.ready_tasks:
            tid = t.id
            at_y = self.AT[tid]
            rt_y = self._RT[tid]
            if tid != taskid:
                bm_t = self.BM[tid]
                fto = at_y + rt_y + self.PT[tid]
                if len(bm_t) == 1 and id(machine) in bm_t and -rt_x < st - at_y < rt_y:
                    wft.append(self.est_ft(taskid, tid, st) - fto)
                    ftm = max(ftm, wft[-1] + fto)
                else:
                    ftm = max(ftm, fto)

        for m in self.platform.machines:
            if m == machine:
                continue
            for t in m.tasks:
                cdi = [d for d in self._cdeps[task.id] if d not in self._cdeps[t.id]
                       and d.to_task in self._ctasks[t.id]]
                cdj = [d for d in self._cdeps[t.id] if d not in self._cdeps[task.id]
                       and d.to_task in self._ctasks[task.id]]
                ctasks = {}
                for d in cdi:
                    if d.to_task not in ctasks:
                        ctasks[d.to_task] = [0, 0]
                    ctasks[d.to_task][0] += self.CT(d)
                for d in cdj:
                    if d.to_task not in ctasks:
                        ctasks[d.to_task] = [0, 0]
                    ctasks[d.to_task][1] += self.CT(d)
                # print(m, t, cdi, cdj, ctasks)
                delta_i = sum(x for x, y in ctasks.values() if x <= y)
                delta_j = sum(y for x, y in ctasks.values() if y <= x)
                ftm = max(ftm, max(
                    st + rt_x + pt_x + delta_i,
                    self.FT(t) + self.PT[t.id] + delta_j
                ))

        return ftm, st + rt_x + pt_x, sorted(wft, reverse=True)


def memo_reset(func):
    func, _memo = {}


def memo(func):
    memo_reset(func)

    def wrapped(*args):
        if args not in func._memo:
            func._memo[args] = func(*args)
        return func._memo
    return wrapped


class CA3_1(CA3):
    def comm_succ_len(self, ti, tj):
        cdeps_i = [d for d in self._cdeps[ti]
                   if d not in self._cdeps[tj] and d.to_task in self._ctasks[tj]]
        cdeps_j = [d for d in self._cdeps[tj]
                   if d not in self._cdeps[ti] and d.to_task in self._ctasks[ti]]
        ctasks = {}
        for d in cdeps_i:
            if d.to_task.id not in ctasks:
                ctasks[d.to_task.id] = [0, 0]
            ctasks[d.to_task.id][0] += self.CT(d)
        for d in cdeps_j:
            if d.to_task.id not in ctasks:
                ctasks[d.to_task.id] = [0, 0]
            ctasks[d.to_task.id][1] += self.CT(d)
        delta_i = sum(x for x, y in ctasks.values() if x <= y)
        delta_j = sum(y for x, y in ctasks.values() if y <= x)
        return delta_i, delta_j

    def est_ft(self, ti, tj, at_i=None):
        if not at_i:
            at_i = self.AT[ti]
        rt_i = self._RT[ti]
        pt_i = self.PT[ti]
        pto_i = self.PTO[ti]
        at_j = self.AT[tj]
        ato_j = self.ATO[tj]
        rt_j = self._RT[tj]
        pt_j = self.PT[tj]
        pto_j = self.PTO[tj]
        di, dj = self.comm_succ_len(ti, tj)
        return min(
            at_i + rt_i + rt_j + pt_i + pt_j,
            at_i + rt_i + rt_j + max(pt_i, pto_j),
            at_i + rt_i + max(pto_i, rt_j + pt_j),
            max(at_i + rt_i + pt_i + di, ato_j + rt_j + pt_j + dj)
        )

    def fitness(self, task, machine, comm_pls, st):
        taskid = task.id
        rt_x = self._RT[taskid]
        pt_x = self.PT[taskid]
        wft = [st - self.AT[taskid]]
        ftt = st + rt_x + pt_x
        ftm = ftt
        for t in self.ready_tasks:
            tid = t.id
            at_y = self.AT[tid]
            rt_y = self._RT[tid]
            if tid != taskid:
                bm_t = self.BM[tid]
                fto = at_y + rt_y + self.PT[tid]
                if len(bm_t) == 1 and id(machine) in bm_t and -rt_x < st - at_y < rt_y:
                    wft.append(self.est_ft(taskid, tid, st) - fto)
                    ftm = max(ftm, wft[-1] + fto)
                else:
                    di, dj = self.comm_succ_len(taskid, tid)
                    wft[0] = max(wft[0], di + (st - self.AT[taskid]))
                    wft.append(dj)
                    # ftm = max(ftm, ftt + di, fto + dj)
                    ftm = max(ftm, fto)

        for m in self.platform.machines:
            if m == machine:
                continue
            for t in m.tasks:
                di, dj = self.comm_succ_len(taskid, t.id)
                wft[0] = max(wft[0], di + (st - self.AT[taskid]))
                wft.append(dj)
                ftm = max(ftm, ftt + di, self.FT(t) + self.PT[t.id] + dj)

        return ftm, ftt, sorted(wft, reverse=True)
