from collections import Counter
from copy import copy
from math import inf
from functools import wraps

from MrWSI.core.platform import Machine, COMM_INPUT, COMM_OUTPUT
from .ca import *


def memo_reset(func):
    func._memo = {}


def memo(func):
    memo_reset(func)

    @wraps(func)
    def wrapped(*args):
        if args not in func._memo:
            func._memo[args] = func(*args)
        return func._memo[args]
    return wrapped


class CAN(CANewRA, CAMoreCompare):
    def _prepare_arrays(self):
        self.AT = [None] * self.problem.num_tasks
        self.ATO = [None] * self.problem.num_tasks
        self.RA = [None] * self.problem.num_tasks
        self.PT = [None] * self.problem.num_tasks
        self.PT_l = [None] * self.problem.num_tasks
        self.PT_c = [None] * self.problem.num_tasks
        self.PT_r = [None] * self.problem.num_tasks
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

    @memo
    def calculate_PT(self, task):
        comms = sorted(task.communications(COMM_OUTPUT),
                       key=lambda c: -self.RP[c.to_task.id])
        pto = 0
        tc = 0
        for c in comms:
            tc += self.CT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])

        local_ft = 0
        local_st = 0
        comm_ft = 0
        remote_ft = 0
        self._cdeps[task.id] = Counter()
        self._ctasks[task.id] = set()
        # print("CPT", task)
        for c in comms:
            lts = (self._ctasks[c.to_task.id] | {
                   c.to_task}) - self._ctasks[task.id]
            local_delta = sum(self._RT[t.id] for t in lts)
            t_local = max(local_ft + local_delta,
                          local_st + self._RT[c.to_task.id] + self.PT_r[c.to_task.id])
            t_remote = comm_ft + self.CT(c) + self.RP[c.to_task.id]
            # print(c, t_local, t_remote)
            if t_local < t_remote:
                # print("local")
                local_ft += local_delta
                local_st += self._RT[c.to_task.id]
                if self.PT_r[c.to_task.id] > 0:
                    remote_ft = max(remote_ft, local_st +
                                    self.PT_r[c.to_task.id])
                self._cdeps[task.id][c] += 1
                self._cdeps[task.id] += self._cdeps[c.to_task.id]
                self._ctasks[task.id].add(c.to_task)
                self._ctasks[task.id].update(self._ctasks[c.to_task.id])
            else:
                # print("remote")
                comm_ft += self.CT(c)
                remote_ft = max(remote_ft, t_remote)
        self.PT[task.id] = max(local_ft, remote_ft)
        self.PT_l[task.id] = local_ft
        self.PT_c[task.id] = comm_ft
        self.PT_r[task.id] = remote_ft
        self.PTO[task.id] = pto
        # print(task, self.PT[task.id], self.PT_l[task.id], self.PT_r[task.id], self._ctasks[task.id], self._cdeps[task.id])

    @memo
    def comm_succ_len(self, ti, tj):
        cts = self._ctasks[ti] & self._ctasks[tj]
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        pt_i = self.PT[ti]
        pt_j = self.PT[tj]
        rft_i = self.PT_r[ti]
        rft_j = self.PT_r[tj]

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i = max(pt_i - self.PT[c.from_task.id] +
                       self.CT(c) for c in ds_i)
            dt_j = max(pt_j - self.PT[c.from_task.id] +
                       self.CT(c) for c in ds_j)
            if dt_i <= dt_j:
                rft_i = max(rft_i, dt_i + self.RP[t.id])
                deps_i -= self._cdeps[t.id]
            else:
                rft_j = max(rft_j, dt_j + self.RP[t.id])
                deps_j -= self._cdeps[t.id]
        di = max(rft_i - pt_i, 0)
        dj = max(rft_j - pt_j, 0)
        return di, dj

    def est_ft_together(self, ti, tj, at_i=None, at_j=None):
        at_i = at_i or self.AT[ti]
        rt_i = self._RT[ti]
        pt_i = self.PT[ti]
        pto_i = self.PTO[ti]
        ptl_i = self.PT_l[ti]
        ptr_i = self.PT_r[ti]
        ptc_i = self.PT_c[ti]

        at_j = at_j or self.AT[tj]
        rt_j = self._RT[tj]
        pt_j = self.PT[tj]
        ptl_j = self.PT_l[tj]
        ptr_j = self.PT_r[tj]
        pto_j = self.PTO[tj]

        lc = sum(self._RT[t.id] for t in (self._ctasks[ti] & self._ctasks[tj]))
        ft_i = at_i + rt_i
        ft_j = max(at_i + rt_i, at_j) + rt_j
        return min(
            max(ft_i + ptr_i, ft_j + ptr_j, ft_j + ptl_i + ptl_j - lc),
            ft_j + max(pt_i, pto_j),
            max(ft_i + pto_i, ft_j + pt_j)
        )

    def est_ft_devided(self, ti, tj, at_i=None):
        if not at_i:
            at_i = self.AT[ti]
        rt_i = self._RT[ti]
        pt_i = self.PT[ti]
        ato_j = self.ATO[tj]
        rt_j = self._RT[tj]
        pt_j = self.PT[tj]
        di, dj = self.comm_succ_len(ti, tj)
        # di, dj = 0, 0
        return max(at_i + rt_i + pt_i + di, ato_j + rt_j + pt_j + dj), di, dj

    def est_ft_devided_free(self, ti, tj, at_i=None):
        if not at_i:
            at_i = self.AT[ti]
        rt_i = self._RT[ti]
        pt_i = self.PT[ti]
        at_j = self.AT[tj]
        rt_j = self._RT[tj]
        pt_j = self.PT[tj]
        di, dj = self.comm_succ_len(ti, tj)
        # di, dj = 0, 0
        return max(at_i + rt_i + pt_i + di, at_j + rt_j + pt_j + dj), di, dj

    def est_ft(self, ti, tj, at_i=None):
        return min(self.est_ft_together(ti, tj, at_i),
                   self.est_ft_devided(ti, tj, at_i)[0])

    def default_fitness(self):
        return inf, inf, [inf]

    def fitness(self, task, machine, comm_pls, st):
        taskid = task.id
        rt_x = self._RT[taskid]
        pt_x = self.PT[taskid]
        wft = [st - self.AT[taskid]]
        ftt = st + rt_x + pt_x
        ftm = ftt

        frontier_0 = set()
        frontier_1 = set()
        for t in self.problem.tasks:
            if self._placed[t.id] and \
                    any(not self._placed[c.to_task.id]
                        for c in t.communications(COMM_OUTPUT)):
                if task not in t.succs():
                    if self.PL_m(t) == machine:
                        frontier_1.update(st for st in t.succs()
                                          if st in self.ready_tasks)
                        if any(st not in self.ready_tasks for st in t.succs()):
                            frontier_0.add(t)
                    else:
                        frontier_0.add(t)
                else:
                    frontier_1.update(st for st in t.succs()
                                      if st in self.ready_tasks and st != task)
                    if any(st not in self.ready_tasks for st in t.succs()):
                        frontier_0.add(t)

        for t in frontier_0:
            if self.PL_m(t) != machine:
                di, dj = self.comm_succ_len(taskid, t.id)
                wft[0] = max(wft[0], di + (st - self.AT[taskid]))
                wft.append(dj)
                ftm = max(ftm, ftt + di, self.FT(t) + self.PT[t.id] + dj)
            else:
                ftm = max(ftm, self.FT(t) + self.PT[t.id])

        for t in frontier_1:
            tid = t.id
            at_y = self.AT[tid]
            rt_y = self._RT[tid]
            bm_t = self.BM[tid]
            fto = at_y + rt_y + self.PT[tid]
            if len(bm_t) == 1 and id(machine) in bm_t and -rt_x < st - at_y < rt_y:
                wft.append(self.est_ft(taskid, tid, st) - fto)
                ftm = max(ftm, wft[-1] + fto)
            elif any(m < 0 or m == id(machine) for m in bm_t):
                ftm = max(ftm, fto)
            else:
                ft_t, di, dj = self.est_ft_devided_free(task.id, t.id, st)
                wft.append(dj)
                wft[0] = max(wft[0], di + st - self.AT[t.id])
                ftm = max(ftm, ft_t)

        return ftm, ftt, sorted(wft, reverse=True)


class CAN2(CAN):
    def est_ft_together_fixed(self, ti, tj, at_i, at_j):
        rt_i = self._RT[ti]
        pt_i = self.PT[ti]
        pto_i = self.PTO[ti]
        ptl_i = self.PT_l[ti]
        ptr_i = self.PT_r[ti]
        ptc_i = self.PT_c[ti]

        rt_j = self._RT[tj]
        pt_j = self.PT[tj]
        ptl_j = self.PT_l[tj]
        ptr_j = self.PT_r[tj]
        pto_j = self.PTO[tj]

        lc = sum(self._RT[t.id] for t in (self._ctasks[ti] & self._ctasks[tj]))
        ft_i = at_i + rt_i
        ft_j = at_j + rt_j
        return max(ft_i + max(ptr_i, ptl_i - lc),
                   ft_j + max(ptr_j, ptl_j - lc))

    def fitness(self, task, machine, comm_pls, st):
        taskid = task.id
        task_ft = st + self._RT[task.id] + self.PT[taskid]
        all_ft = []
        wrk_ft = task_ft
        cur_ft = st - self.AT[taskid]

        for t in self.problem.tasks:
            if self._placed[t.id] and \
                    any(not self._placed[c.to_task.id]
                        for c in t.communications(COMM_OUTPUT)):
                if self.PL_m(t) != machine:
                    di, dj = self.comm_succ_len(taskid, t.id)
                    all_ft.append(dj)
                    cur_ft = max(cur_ft, di + st - self.AT[taskid])
                    wrk_ft = max(wrk_ft, task_ft + di,
                                 self.FT(t) + self.PT[t.id] + dj)
                else:
                    ft = self.est_ft_together_fixed(
                        t.id, taskid, self.ST(t), st)
                    all_ft.append(0)
                    cur_ft = max(cur_ft, ft - task_ft)
                    wrk_ft = max(wrk_ft, ft)
        all_ft.append(cur_ft)

        return wrk_ft, task_ft, sorted(all_ft, reverse=True)


class CAN3(CAN2):
    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        cts = ctasks_i & ctasks_j
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        pt_i = self.PT[ti]
        pt_j = self.PT[tj]
        rft_i = self.PT_r[ti]
        rft_j = self.PT_r[tj]
        # print(pt_i, rft_i, ctasks_i, pt_j, rft_j, ctasks_j, cts)

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i = max(self.RA[c.from_task.id] + self.CT(c) for c in ds_i)
            dt_j = max(self.RA[c.from_task.id] + self.CT(c) for c in ds_j)
            if dt_i <= dt_j:
                rft_i = max(rft_i,
                            dt_i + self.RP[t.id] - self.RA[ti])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                rft_j = max(rft_j,
                            dt_j + self.RP[t.id] - self.RA[tj])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        di = max(rft_i, sum(self._RT[t.id] for t in ctasks_i)) - pt_i
        dj = max(rft_j, sum(self._RT[t.id] for t in ctasks_j)) - pt_j
        # print("COMM", self.problem.tasks[ti], self.problem.tasks[tj], di, dj)
        return di, dj

    def ef_XYAB(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = max(ft_i, st_j) + self._RT[tj]
        lc = sum(self._RT[t.id] for t in (self._ctasks[ti] & self._ctasks[tj]))
        # print("XYAB", ft_i, ft_j, lc)
        if ft_i + self.PT_l[ti] <= st_j:
            # print(">1", self.PT_l[ti], self.PT_r[ti], self.PT_l[tj], self.PT_r[tj], lc)
            ft_ai = ft_i + max(self.PT_r[ti], self.PT_l[ti])
            ft_aj = ft_j + max(self.PT_r[tj], self.PT_l[tj] - lc)
        else:
            # print(">2", self.PT_l[ti], self.PT_r[ti], self.PT_l[tj], self.PT_r[tj], lc)
            ft_ai = ft_i + max(self.PT_r[ti], self.PT_l[ti] + self._RT[tj])
            ft_aj = max(ft_j + self.PT_r[tj], ft_ai + self.PT_l[tj] - lc)
        # print(self.problem.tasks[ti], self.problem.tasks[tj], ft_i, ft_j, self.PT[ti], self.PT[tj], ft_ai, ft_aj)
        return max(ft_ai, ft_aj), ft_ai, ft_aj

    def ef_XYA_b(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = max(ft_i, st_j) + self._RT[tj]
        ft_ai = ft_j + self.PT[ti]
        ft_aj = ft_j + self.PTO[tj]
        return max(ft_ai, ft_aj), ft_ai, ft_aj

    def ef_XYB_a(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = max(ft_i, st_j) + self._RT[tj]
        ft_ai = ft_i + self.PTO[ti]
        ft_aj = ft_j + self.PT[tj]
        return max(ft_ai, ft_aj), ft_ai, ft_aj

    def ef_XA_YB(self, ti, tj, st_i, st_j):
        di, dj = self.comm_succ_len(ti, tj)
        ft_ai = st_i + self._RT[ti] + self.PT[ti] + di
        ft_aj = st_j + self._RT[tj] + self.PT[tj] + dj
        # print("XA_YB", self.problem.tasks[ti], self.problem.tasks[tj], st_i, st_j, di, dj, ft_ai, ft_aj, self.PT[ti], self.PT[tj])
        return max(ft_ai, ft_aj), ft_ai, ft_aj

    def est_ft(self, ti, tj):
        ft0 = self.ef_XYAB(ti, tj, self.AT[ti], self.AT[tj])[0]
        ft1 = self.ef_XYA_b(ti, tj, self.AT[ti], self.AT[tj])[0]
        ft2 = self.ef_XYB_a(ti, tj, self.AT[ti], self.AT[tj])[0]
        ft3 = self.ef_XA_YB(ti, tj, self.AT[ti], self.ATO[tj])[0]
        # print(self.problem.tasks[ti], self.problem.tasks[tj], ft0, ft1, ft2, ft3)
        return min(ft0, ft1, ft2, ft3)

    @memo
    def is_successor(self, task_i, task_j):
        return task_j in task_i.succs() or \
            any(self.is_successor(t, task_j) for t in task_i.succs())

    def dft(self, task):
        ft_a = self.FT(task) + self.PT[task.id]
        for t in task.succs():
            if self._placed[t.id]:
                ft_a = max(ft_a, self.dft(t))
        return ft_a

    def fitness(self, task, machine, comm_pls, st):
        all_ft = []
        task_ft = st + self._RT[task.id] + self.PT[task.id]
        wrk_ft = task_ft
        cur_ft = task_ft

        # for t in self.problem.tasks:
        # print(">>>>", t, self.PT[t.id], self._ctasks[t.id])

        for t in self.problem.tasks:
            if self._placed[t.id] and \
                    any(not self._placed[c.to_task.id]
                        for c in t.communications(COMM_OUTPUT)
                        if c.to_task != task):
                st_t = self.ST(t)
                if self.PL_m(t) != machine:
                    ft_a, ft_i, ft_j = self.ef_XA_YB(t.id, task.id, st_t, st)
                elif self.is_successor(t, task):
                    # print("CASE 2", st, self.PT[task.id], self.PT[t.id])
                    ft_j = st + self._RT[task.id] + self.PT[task.id]
                    ft_a = ft_i = max(ft_j, self.FT(t) + self.PT[t.id])
                else:
                    ft_a0, ft_i0, ft_j0 = self.ef_XYAB(
                        t.id, task.id, st_t, st)
                    ft_a1, ft_i1, ft_j1 = self.ef_XYB_a(
                        t.id, task.id, st_t, st)
                    ft_a2, ft_i2, ft_j2 = self.ef_XYA_b(
                        t.id, task.id, st_t, st)
                    # print("CASE 3", ft_a0, ft_a1, ft_a2)
                    ft_a = min(ft_a0, ft_a1, ft_a2)
                    if ft_a == ft_a0:
                        ft_i, ft_j = ft_i0, ft_j0
                    elif ft_a == ft_a1:
                        ft_i, ft_j = ft_i1, ft_j1
                    else:
                        ft_i, ft_j = ft_i2, ft_j2
                # print(t, ft_a, ft_i, ft_j)
                all_ft.append(ft_i)
                cur_ft = max(cur_ft, ft_j)
                wrk_ft = max(wrk_ft, ft_a)
        all_ft.append(cur_ft)
        return wrk_ft, task_ft, sorted(all_ft, reverse=True)


class CAN3_1(CAN3):
    @memo
    def calculate_PT(self, task):
        comms = sorted(task.communications(COMM_OUTPUT),
                       key=lambda c: -self.RP[c.to_task.id])
        pto = 0
        tc = 0
        for c in comms:
            tc += self.CT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])

        local_ft = 0
        local_st = 0
        comm_ft = 0
        remote_ft = 0
        self._cdeps[task.id] = Counter()
        self._ctasks[task.id] = set()
        # print("CPT", task)
        for c in comms:
            lts = (self._ctasks[c.to_task.id] | {
                   c.to_task}) - self._ctasks[task.id]
            local_delta = sum(self._RT[t.id] for t in lts)
            t_local = max(local_ft + local_delta,
                          local_st + self._RT[c.to_task.id] + self.PT_r[c.to_task.id])
            t_remote = comm_ft + self.CT(c) + self.RP[c.to_task.id]
            # print(c, t_local, t_remote)
            if t_local <= t_remote:
                # print("local")
                local_ft += local_delta
                local_st += self._RT[c.to_task.id]
                if self.PT_r[c.to_task.id] > 0:
                    remote_ft = max(remote_ft, local_st +
                                    self.PT_r[c.to_task.id])
                self._cdeps[task.id][c] += 1
                self._cdeps[task.id] += self._cdeps[c.to_task.id]
                self._ctasks[task.id].add(c.to_task)
                self._ctasks[task.id].update(self._ctasks[c.to_task.id])
            else:
                # print("remote")
                comm_ft += self.CT(c)
                remote_ft = max(remote_ft, t_remote)
        self.PT[task.id] = max(local_ft, remote_ft)
        self.PT_l[task.id] = local_ft
        self.PT_c[task.id] = comm_ft
        self.PT_r[task.id] = remote_ft
        self.PTO[task.id] = pto


class CAN3_2(CAN3):
    @memo
    def calculate_PT(self, task):
        comms = sorted(task.communications(COMM_OUTPUT),
                       key=lambda c: -self.RP[c.to_task.id])
        pto = 0
        tc = 0
        for c in comms:
            tc += self.CT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])

        local_ft = 0
        local_st = 0
        comm_ft = 0
        remote_ft = 0
        self._cdeps[task.id] = Counter()
        self._ctasks[task.id] = set()
        for c in comms:
            lts = (self._ctasks[c.to_task.id] | {
                   c.to_task}) - self._ctasks[task.id]
            local_delta = sum(self._RT[t.id] for t in lts)
            t_local = max(remote_ft, local_ft + local_delta,
                          local_st + self._RT[c.to_task.id] + self.PT_r[c.to_task.id])
            t_remote = max(remote_ft, local_ft, comm_ft +
                           self.CT(c) + self.RP[c.to_task.id])
            if (t_local, local_st) <= (t_remote, comm_ft + self.CT(c)):
                local_ft += local_delta
                local_st += self._RT[c.to_task.id]
                if self.PT_r[c.to_task.id] > 0:
                    remote_ft = max(remote_ft, local_st +
                                    self.PT_r[c.to_task.id])
                self._cdeps[task.id][c] += 1
                self._cdeps[task.id] += self._cdeps[c.to_task.id]
                self._ctasks[task.id].add(c.to_task)
                self._ctasks[task.id].update(self._ctasks[c.to_task.id])
            else:
                comm_ft += self.CT(c)
                remote_ft = max(remote_ft, t_remote)
        self.PT[task.id] = max(local_ft, remote_ft)
        self.PT_l[task.id] = local_ft
        self.PT_c[task.id] = comm_ft
        self.PT_r[task.id] = remote_ft
        self.PTO[task.id] = pto
        # print(task, self._ctasks[task.id])


class CAN3_2_1(CAN3_2):
    def fitness(self, task, machine, comm_pls, st):
        all_ft = []
        task_ft = st + self._RT[task.id] + self.PT[task.id]
        wrk_ft = task_ft
        cur_ft = task_ft

        # for t in self.problem.tasks:
        # print(">>>>", t, self.PT[t.id], self._ctasks[t.id])

        for t in self.problem.tasks:
            if self._placed[t.id] and \
                    any(not self._placed[c.to_task.id]
                        for c in t.communications(COMM_OUTPUT)
                        if c.to_task != task):
                st_t = self.ST(t)
                if self.PL_m(t) != machine:
                    # print("CASE 1", st_t, st)
                    ft_a, ft_i, ft_j = self.ef_XA_YB(t.id, task.id, st_t, st)
                elif self.is_successor(t, task):
                    # print("CASE 2", st, self.PT[task.id], self.PT[t.id])
                    ft_j = st + self._RT[task.id] + self.PT[task.id]
                    ft_a = ft_i = max(ft_j, self.FT(t) + self.PT[t.id])
                else:
                    ft_a0, ft_i0, ft_j0 = self.ef_XYAB(
                        t.id, task.id, st_t, st)
                    ft_a1, ft_i1, ft_j1 = self.ef_XYB_a(
                        t.id, task.id, st_t, st)
                    ft_a2, ft_i2, ft_j2 = self.ef_XYA_b(
                        t.id, task.id, st_t, st)
                    # print("CASE 3", ft_a0, ft_a1, ft_a2)
                    ft_a = min(ft_a0, ft_a1, ft_a2)
                    if ft_a == ft_a0:
                        ft_i, ft_j = ft_i0, ft_j0
                    elif ft_a == ft_a1:
                        ft_i, ft_j = ft_i1, ft_j1
                    else:
                        ft_i, ft_j = ft_i2, ft_j2
                # print(t, ft_a, ft_i, ft_j)
                all_ft.append(ft_i)
                cur_ft = max(cur_ft, ft_j)
                wrk_ft = max(wrk_ft, ft_a)
        all_ft.append(cur_ft)
        return wrk_ft, sorted(all_ft, reverse=True), task_ft


class CAN4(CAN3):
    @memo
    def comm_succ_len(self, ti, tj, st_i, st_j):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        cts = ctasks_i & ctasks_j
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        pt_i = self.PT[ti]
        pt_j = self.PT[tj]
        rft_i = self.PT_r[ti]
        rft_j = self.PT_r[tj]

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i = max(self.RA[c.from_task.id] + self.CT(c) for c in ds_i)
            dt_j = max(self.RA[c.from_task.id] + self.CT(c) for c in ds_j)
            if dt_i - self.AT[ti] + st_i <= dt_j - self.AT[tj] + st_j:
                rft_i = max(rft_i,
                            dt_i + self.RP[t.id] - self.RA[ti])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                rft_j = max(rft_j,
                            dt_j + self.RP[t.id] - self.RA[tj])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        di = max(rft_i, sum(self._RT[t.id] for t in ctasks_i)) - pt_i
        dj = max(rft_j, sum(self._RT[t.id] for t in ctasks_j)) - pt_j
        return di, dj

    def ef_XA_YB(self, ti, tj, st_i, st_j):
        di, dj = self.comm_succ_len(ti, tj, st_i, st_j)
        ft_ai = st_i + self._RT[ti] + self.PT[ti] + di
        ft_aj = st_j + self._RT[tj] + self.PT[tj] + dj
        return max(ft_ai, ft_aj), ft_ai, ft_aj


class CAN5(CAN3_2):
    def _prepare_arrays(self):
        self.AT = [None] * self.problem.num_tasks
        self.ATO = [None] * self.problem.num_tasks
        self.RA = [None] * self.problem.num_tasks
        self.PT = [None] * self.problem.num_tasks
        self.PT_l = [None] * self.problem.num_tasks
        self.PT_c = [None] * self.problem.num_tasks
        self.PT_r = [None] * self.problem.num_tasks
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
        self._rsuccs = [[] for _ in self.problem.tasks]
        self._cdeps = [set() for _ in self.problem.tasks]

    @memo
    def calculate_PT(self, task):
        comms = sorted(task.communications(COMM_OUTPUT),
                       key=lambda c: -self.RP[c.to_task.id])
        pto = 0
        tc = 0
        for c in comms:
            tc += self.CT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])

        local_ft = 0
        local_st = 0
        comm_ft = 0
        remote_ft = 0
        self._cdeps[task.id] = Counter()
        self._ctasks[task.id] = set()
        self._rsuccs[task.id] = []
        for c in comms:
            lts = (self._ctasks[c.to_task.id] | {
                   c.to_task}) - self._ctasks[task.id]
            local_delta = sum(self._RT[t.id] for t in lts)
            t_local = max(remote_ft, local_ft + local_delta,
                          local_st + self._RT[c.to_task.id] + self.PT_r[c.to_task.id])
            t_remote = max(remote_ft, local_ft, comm_ft +
                           self.CT(c) + self.RP[c.to_task.id])
            if (t_local, local_st) <= (t_remote, comm_ft + self.CT(c)):
                local_ft += local_delta
                local_st += self._RT[c.to_task.id]
                if self.PT_r[c.to_task.id] > 0:
                    remote_ft = max(remote_ft, local_st +
                                    self.PT_r[c.to_task.id])
                self._cdeps[task.id][c] += 1
                self._cdeps[task.id] += self._cdeps[c.to_task.id]
                self._ctasks[task.id].add(c.to_task)
                self._ctasks[task.id].update(self._ctasks[c.to_task.id])
            else:
                comm_ft += self.CT(c)
                remote_ft = max(remote_ft, t_remote)
                self._rsuccs[task.id].append(c)
        self.PT[task.id] = max(local_ft, remote_ft)
        self.PT_l[task.id] = local_ft
        self.PT_c[task.id] = comm_ft
        self.PT_r[task.id] = remote_ft
        self.PTO[task.id] = pto
        # print(task, self._ctasks[task.id], self._rsuccs[task.id])

    def comm_merge_length(self, ft_delta, cs_i, cs_j):
        if ft_delta < 0:
            return self.comm_merge_length(-ft_delta, cs_j, cs_i)
        i, j, cst, l, l_i, l_j = 0, 0, 0, 0, 0, 0
        while i < len(cs_i) and cst + self.CT(cs_i[i]) < ft_delta:
            cst += self.CT(cs_i[i])
            l = max(l, cst + self.RP[cs_i[i].to_task.id])
            # print("1", cst, l)
            l_i = max(l_i, l)
            i += 1
        if j < len(cs_j):
            if i < len(cs_i):
                A, B = self.CT(cs_i[i]), self.CT(cs_j[j])
                x, y = self.RP[cs_i[i].to_task.id], self.RP[cs_j[j].to_task.id]
                d = ft_delta - cst
                cst += A + B
                if max(A + x, A + B + y) <= max(B + d + y, A + B + x):
                    l = max(l, A + x, A + B + y)
                    l_i = max(l_i, A + x)
                    l_j = max(l_j, A + B + y)
                else:
                    l = max(l, B + d + y, A + B + x)
                    l_i = max(l_i, A + B + x)
                    l_j = max(l_j, B + d + y)
                i += 1
                j += 1
            else:
                cst = ft_delta
        while i < len(cs_i) and j < len(cs_j):
            if self.RP[cs_i[i].to_task.id] >= self.RP[cs_j[j].to_task.id]:
                cst += self.CT(cs_i[i])
                l = max(l, cst + self.RP[cs_i[i].to_task.id])
                # print("3", cst, l)
                l_i = max(l_i, l)
                i += 1
            else:
                cst += self.CT(cs_j[j])
                l = max(l, cst + self.RP[cs_j[j].to_task.id])
                # print("4", cst, l)
                l_j = max(l_j, l)
                j += 1
        while i < len(cs_i):
            cst += self.CT(cs_i[i])
            l_i = max(l_i, cst + self.RP[cs_i[i].to_task.id])
            # print("5", cst, l_i)
            i += 1
        while j < len(cs_j):
            cst += self.CT(cs_j[j])
            l_j = max(l_j, cst + self.RP[cs_j[j].to_task.id])
            # print("6", cst, l_j)
            j += 1
        # print(l_i, l_j)
        if l_j > 0:
            l_j -= ft_delta
        # print(l_i, l_j)
        return l_i, l_j

    def ef_XYAB(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = max(ft_i, st_j) + self._RT[tj]
        lc = sum(self._RT[t.id] for t in (self._ctasks[ti] & self._ctasks[tj]))
        ptr_i, ptr_j = self.comm_merge_length(
            ft_j - ft_i, self._rsuccs[ti], self._rsuccs[tj])
        # print(self.problem.tasks[ti], self.problem.tasks[tj], ft_i, ft_j,
        # self._rsuccs[ti], self._rsuccs[tj], ptr_i, ptr_j, self.PT_r[ti], self.PT_r[tj])
        ptr_i = max(ptr_i, self.PT_r[ti])
        ptr_j = max(ptr_j, self.PT_r[tj])
        if ft_i + ptr_i <= st_j:
            ft_ai = ft_i + max(ptr_i, self.PT_l[ti])
            ft_aj = ft_j + max(ptr_j, self.PT_l[tj] - lc)
        else:
            ft_ai = ft_i + max(ptr_i, self.PT_l[ti] + self._RT[tj])
            ft_aj = max(ft_j + ptr_j, ft_ai + self.PT_l[tj] - lc)
        return max(ft_ai, ft_aj), ft_ai, ft_aj


class CAN6(CAN3_2):
    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        cts = ctasks_i & ctasks_j
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        pt_i = self.PT[ti]
        pt_j = self.PT[tj]
        rft_i = self.PT_r[ti]
        rft_j = self.PT_r[tj]
        dt_i, dt_j = self.RA[ti], self.RA[tj]

        # for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
        for t in sorted(cts, key=lambda _t: self.RA[_t.id]):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            for c in sorted(ds_i, key=lambda c: self.RA[c.from_task.id]):
                dt_i = max(dt_i, self.RA[c.from_task.id]) + self.CT(c)
            for c in sorted(ds_j, key=lambda c: self.RA[c.from_task.id]):
                dt_j = max(dt_j, self.RA[c.from_task.id]) + self.CT(c)
            if dt_i <= dt_j:
                rft_i = max(rft_i,
                            dt_i + self.RP[t.id] - self.RA[ti])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                rft_j = max(rft_j,
                            dt_j + self.RP[t.id] - self.RA[tj])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        di = max(rft_i, sum(self._RT[t.id] for t in ctasks_i)) - pt_i
        dj = max(rft_j, sum(self._RT[t.id] for t in ctasks_j)) - pt_j
        return di, dj


class CAN6_1(CAN3_2):
    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        cts = ctasks_i & ctasks_j
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        pt_i = self.PT[ti]
        pt_j = self.PT[tj]
        rft_i = self.PT_r[ti]
        rft_j = self.PT_r[tj]
        dt_i, dt_j = self.RA[ti], self.RA[tj]

        # print(cts, rft_i, rft_j)

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            # for t in sorted(cts, key=lambda _t: self.RA[_t.id]):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            # print(t, ds_i, ds_j)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self.RA[c.from_task.id]):
                dt_i = max(dt_i, self.RA[c.from_task.id]) + self.CT(c)
                # print("dt_i", c, dt_i)
            for c in sorted(ds_j, key=lambda c: self.RA[c.from_task.id]):
                dt_j = max(dt_j, self.RA[c.from_task.id]) + self.CT(c)
                # print("dt_j", c, dt_j)
            # print(dt_i, dt_j)
            if dt_i <= dt_j:
                dt_j = dt_j0
                rft_i = max(rft_i,
                            dt_i + self.RP[t.id] - self.RA[ti])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j,
                            dt_j + self.RP[t.id] - self.RA[tj])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)
            # print("rft", rft_i, rft_j)

        di = max(rft_i, sum(self._RT[t.id] for t in ctasks_i)) - pt_i
        dj = max(rft_j, sum(self._RT[t.id] for t in ctasks_j)) - pt_j
        # print(">>", self.problem.tasks[ti], self.problem.tasks[tj], di, dj)
        return di, dj


class CAN6_2(CAN3_2):
    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        cts = ctasks_i & ctasks_j
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        pt_i = self.PT[ti]
        pt_j = self.PT[tj]
        rft_i = self.PT_r[ti]
        rft_j = self.PT_r[tj]
        dt_i, dt_j = self.RA[ti], self.RA[tj]

        # for t in self.problem.tasks:
        # print("RA", t, self.RA[t.id])

        # for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
        for t in sorted(cts, key=lambda _t: self.RA[_t.id]):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self.RA[c.from_task.id]):
                dt_i = max(dt_i, self.RA[c.from_task.id]) + self.CT(c)
                # print("dt_i", c, dt_i)
            for c in sorted(ds_j, key=lambda c: self.RA[c.from_task.id]):
                dt_j = max(dt_j, self.RA[c.from_task.id]) + self.CT(c)
                # print("dt_j", c, dt_j)
            # print(dt_i, dt_j)
            if dt_i <= dt_j:
                dt_j = dt_j0
                rft_i = max(rft_i,
                            dt_i + self.RP[t.id] - self.RA[ti])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j,
                            dt_j + self.RP[t.id] - self.RA[tj])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)
            # print("rft", rft_i, rft_j)

        di = max(rft_i, sum(self._RT[t.id] for t in ctasks_i)) - pt_i
        dj = max(rft_j, sum(self._RT[t.id] for t in ctasks_j)) - pt_j
        # print(cts, rft_i, rft_j, ctasks_i, ctasks_j, pt_i, pt_j)
        # print(">>", self.problem.tasks[ti], self.problem.tasks[tj], di, dj)
        return di, dj


class CAN6_2_1(CAN3_2):
    @memo
    def task_distance(self, tx, ty):
        if tx == ty:
            return 0
        dm = None
        for c in tx.communications(COMM_OUTPUT):
            d = self.task_distance(c.to_task, ty)
            if d != None and (dm == None or d + self.CT(c) > dm):
                dm = d + self.CT(c)
        return dm

    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        cts = ctasks_i & ctasks_j
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        pt_i = self.PT[ti]
        pt_j = self.PT[tj]
        rft_i = self.PT_r[ti]
        rft_j = self.PT_r[tj]
        dt_i, dt_j = 0, 0

        for t in sorted(cts, key=lambda _t: self.RA[_t.id]):
        # for t in sorted(cts, key=lambda _t: -self.RP[_t.id]):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self.RA[c.from_task.id]):
                dt_i = max(dt_i, self.task_distance(
                    self.problem.tasks[ti], c.from_task)) + self.CT(c)
            for c in sorted(ds_j, key=lambda c: self.RA[c.from_task.id]):
                dt_j = max(dt_j, self.task_distance(
                    self.problem.tasks[tj], c.from_task)) + self.CT(c)
            if self.RA[ti] + dt_i <= self.RA[tj] + dt_j:
                dt_j = dt_j0
                rft_i = max(rft_i, dt_i + self.RP[t.id])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j, dt_j + self.RP[t.id])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        di = max(rft_i, sum(self._RT[t.id] for t in ctasks_i)) - pt_i
        dj = max(rft_j, sum(self._RT[t.id] for t in ctasks_j)) - pt_j
        # print(">>", self.problem.tasks[ti], self.problem.tasks[tj], di, dj)
        return di, dj


class CAN6_2_2(CAN6_2_1):
    @memo
    def task_distance(self, tx, ty):
        return self.PT[tx.id] - self.RP[ty.id]


class NeighborFirstSort(Heuristic):
    def select_task(self):
        self.update_AT_and_PT()
        self._dcs = [0] * self.problem.num_tasks
        self.w = [0] * self.problem.num_tasks
        # self.w = [self.RA[i] + self.PT[i]
        # for i in range(self.problem.num_tasks)]
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and\
                        not (self.rids[tx] and self.rids[ty]) and\
                        self.has_contention(tx, ty):
                    ftx = self.est_ft(tx, ty)
                    fty = self.est_ft(ty, tx)
                    # print(self.problem.tasks[tx], self.problem.tasks[ty], ftx, fty)
                    if ftx < fty:
                        self._dcs[ty] -= 1
                    elif ftx > fty:
                        self._dcs[tx] -= 1
                if self._placed[ty]:
                    dy, dx = self.comm_succ_len(ty, tx)
                    # self.w[tx] = max(
                    # self.w[tx], self.RA[tx] + self.PT[tx] + dx, self.RA[ty] + self.PT[ty] + dy)
                    self.w[tx] = max(self.w[tx], self.comm_succ_len(ty, tx)[1])
        # print([(t, self._dcs[t.id], self.RP[t.id]) for t in self.ready_tasks])
        task = max(self.ready_tasks, key=lambda t: (
            self._dcs[t.id], self.w[t.id], self.RP[t.id]))
        # task = max(self.ready_tasks, key=lambda t: (
        # self.w[t.id], self._dcs[t.id], self.RP[t.id]))
        # print("Selected", task)
        return task


class NeighborFirstSort2(Heuristic):
    def select_task(self):
        self.update_AT_and_PT()
        self._dcs = [0] * self.problem.num_tasks
        self.w = [0] * self.problem.num_tasks
        # self.w = [self.RA[i] + self.PT[i]
        # for i in range(self.problem.num_tasks)]
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and\
                        not (self.rids[tx] and self.rids[ty]) and\
                        self.has_contention(tx, ty):
                    ftx = self.est_ft(tx, ty)
                    fty = self.est_ft(ty, tx)
                    # print(self.problem.tasks[tx], self.problem.tasks[ty], ftx, fty)
                    if ftx < fty:
                        self._dcs[ty] -= 1
                    elif ftx > fty:
                        self._dcs[tx] -= 1
                dy, dx = self.comm_succ_len(ty, tx)
                self.w[tx] = max(self.w[tx], self.RP[tx] + dx)
        # print([(t, self._dcs[t.id], self.w[t.id], self.RP[t.id])
               # for t in self.ready_tasks])
        task = max(self.ready_tasks, key=lambda t: (
            self._dcs[t.id], self.w[t.id] + self.RP[t.id]))
        # print("Selected", task)
        return task


class CAN6_2_3(CAN6_2_1):
    def _prepare_arrays(self):
        self.AT = [None] * self.problem.num_tasks
        self.ATO = [None] * self.problem.num_tasks
        self.RA = [None] * self.problem.num_tasks
        self.PT = [None] * self.problem.num_tasks
        self.PT_l = [None] * self.problem.num_tasks
        self.PT_c = [None] * self.problem.num_tasks
        self.PT_r = [None] * self.problem.num_tasks
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
        self.TST = {}

    @memo
    def calculate_PT(self, task):
        comms = sorted(task.communications(COMM_OUTPUT),
                       key=lambda c: -self.RP[c.to_task.id])
        pto = 0
        tc = 0
        for c in comms:
            tc += self.CT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])

        local_ft = 0
        local_st = 0
        comm_ft = 0
        remote_ft = 0
        self._cdeps[task.id] = Counter()
        self._ctasks[task.id] = set()
        for c in comms:
            lts = (self._ctasks[c.to_task.id] | {
                   c.to_task}) - self._ctasks[task.id]
            local_delta = sum(self._RT[t.id] for t in lts)
            t_local = max(remote_ft, local_ft + local_delta,
                          local_st + self._RT[c.to_task.id] + self.PT_r[c.to_task.id])
            t_remote = max(remote_ft, local_ft, comm_ft +
                           self.CT(c) + self.RP[c.to_task.id])
            if (t_local, local_st) <= (t_remote, comm_ft + self.CT(c)):
                local_ft += local_delta
                self.TST[c] = local_st
                local_st += self._RT[c.to_task.id]
                if self.PT_r[c.to_task.id] > 0:
                    remote_ft = max(remote_ft, local_st +
                                    self.PT_r[c.to_task.id])
                self._cdeps[task.id][c] += 1
                self._cdeps[task.id] += self._cdeps[c.to_task.id]
                self._ctasks[task.id].add(c.to_task)
                self._ctasks[task.id].update(self._ctasks[c.to_task.id])
            else:
                comm_ft += self.CT(c)
                self.TST[c] = comm_ft
                remote_ft = max(remote_ft, t_remote)
        self.PT[task.id] = max(local_ft, remote_ft)
        self.PT_l[task.id] = local_ft
        self.PT_c[task.id] = comm_ft
        self.PT_r[task.id] = remote_ft
        self.PTO[task.id] = pto
        # print(task, self._ctasks[task.id])

    @memo
    def task_distance(self, tx, ty):
        if tx == ty:
            return 0
        dm = None
        for c in tx.communications(COMM_OUTPUT):
            d = self.task_distance(c.to_task, ty)
            if d != None and (dm == None or d + self.TST[c] > dm):
                dm = d + self.TST[c]
        return dm


class CAN6_3(CAN6_2):
    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        cts = ctasks_i & ctasks_j
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        pt_i = self.PT[ti]
        pt_j = self.PT[tj]
        rft_i = self.PT_r[ti]
        rft_j = self.PT_r[tj]
        dt_i, dt_j = self.RA[ti], self.RA[tj]
        # dt_i, dt_j = self.RA[ti] + self.PT_c[ti], self.RA[tj] + self.PT_c[tj]

        # print("  CSL:>>>", self.problem.tasks[ti], self.problem.tasks[tj], deps_i, deps_j, self.PT_c[ti], self.PT_c[tj])

        # if self.problem.tasks[tj] in ctasks_i:
        # for c in sorted(self.problem.tasks[tj].communications(COMM_INPUT),
        # key=lambda c: self.RA[c.from_task.id]):
        # if c in deps_i:
        # dt_i = max(dt_i, self.RA[c.from_task.id] + self.CT(c))

        # if self.problem.tasks[ti] in ctasks_j:
        # for c in sorted(self.problem.tasks[ti].communications(COMM_INPUT),
        # key=lambda c: self.RA[c.from_task.id]):
        # if c in deps_j:
        # dt_i = max(dt_j, self.RA[c.from_task.id] + self.CT(c))

        for t in sorted(cts, key=lambda _t: self.RA[_t.id]):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self.RA[c.from_task.id]):
                dt_i = max(
                    dt_i, self.RA[c.from_task.id] + self.CST[c]) + self.CT(c)
            for c in sorted(ds_j, key=lambda c: self.RA[c.from_task.id]):
                dt_j = max(
                    dt_j, self.RA[c.from_task.id] + self.CST[c]) + self.CT(c)
            if dt_i <= dt_j:
                dt_j = dt_j0
                rft_i = max(rft_i,
                            dt_i + self.RP[t.id] - self.RA[ti])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j,
                            dt_j + self.RP[t.id] - self.RA[tj])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        di = max(rft_i, sum(self._RT[t.id] for t in ctasks_i)) - pt_i
        dj = max(rft_j, sum(self._RT[t.id] for t in ctasks_j)) - pt_j
        return di, dj

    def _prepare_arrays(self):
        self.AT = [None] * self.problem.num_tasks
        self.ATO = [None] * self.problem.num_tasks
        self.RA = [None] * self.problem.num_tasks
        self.PT = [None] * self.problem.num_tasks
        self.PT_l = [None] * self.problem.num_tasks
        self.PT_c = [None] * self.problem.num_tasks
        self.PT_r = [None] * self.problem.num_tasks
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
        self.CST = {}

    @memo
    def calculate_PT(self, task):
        comms = sorted(task.communications(COMM_OUTPUT),
                       key=lambda c: -self.RP[c.to_task.id])
        pto = 0
        tc = 0
        for c in comms:
            tc += self.CT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])

        local_ft = 0
        local_st = 0
        comm_ft = 0
        remote_ft = 0
        self._cdeps[task.id] = Counter()
        self._ctasks[task.id] = set()
        for c in comms:
            lts = (self._ctasks[c.to_task.id] | {
                   c.to_task}) - self._ctasks[task.id]
            local_delta = sum(self._RT[t.id] for t in lts)
            t_local = max(remote_ft, local_ft + local_delta,
                          local_st + self._RT[c.to_task.id] + self.PT_r[c.to_task.id])
            t_remote = max(remote_ft, local_ft, comm_ft +
                           self.CT(c) + self.RP[c.to_task.id])
            if (t_local, local_st) <= (t_remote, comm_ft + self.CT(c)):
                local_ft += local_delta
                local_st += self._RT[c.to_task.id]
                if self.PT_r[c.to_task.id] > 0:
                    remote_ft = max(remote_ft, local_st +
                                    self.PT_r[c.to_task.id])
                self._cdeps[task.id][c] += 1
                self._cdeps[task.id] += self._cdeps[c.to_task.id]
                self._ctasks[task.id].add(c.to_task)
                self._ctasks[task.id].update(self._ctasks[c.to_task.id])
                self.CST[c] = comm_ft
            else:
                comm_ft += self.CT(c)
                remote_ft = max(remote_ft, t_remote)
        self.PT[task.id] = max(local_ft, remote_ft)
        self.PT_l[task.id] = local_ft
        self.PT_c[task.id] = comm_ft
        self.PT_r[task.id] = remote_ft
        self.PTO[task.id] = pto
        # print(task, self._ctasks[task.id])


class CAN7(CAN6_2):
    def ef_XYAB(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = max(ft_i, st_j) + self._RT[tj]
        if ft_i + self.PT_l[ti] <= st_j:
            lc = sum(self._RT[t.id]
                     for t in (self._ctasks[ti] & self._ctasks[tj]))
            ft_ai = ft_i + max(self.PT_r[ti], self.PT_l[ti])
            ft_aj = ft_j + max(self.PT_r[tj], self.PT_l[tj] - lc)
        else:
            cti = copy(self._ctasks[ti])
            tst = st_i
            ft_ai, ft_aj = st_i + self.PT_r[ti], st_j + self.PT_r[tj]
            for t in sorted(list(cti), key=lambda t: self.RP[t.id], reverse=True):
                if self._RT[t.id] + tst < st_j:
                    if not self.is_successor(self.problem.tasks[tj], t):
                        tst += self._RT[t.id]
                        cti.remove(t)
                        ft_ai = max(ft_ai, tst + self.PT[t.id])
                else:
                    break
            ts = self._ctasks[ti] | self._ctasks[tj]
            tst, cst = st_j, st_j
            for t in sorted(ts, key=lambda t: self.RP[t.id], reverse=True):
                cmt = sum([self.CT(c) for c in (+self._cdeps[ti]
                                                & +self._cdeps[tj]) if c.to_task == t])
                if tst > cst + cmt:
                    cst += cmt
                    if t in self._ctasks[ti]:
                        ft_ai = max(ft_ai, cst + self.RP[t.id])
                    if t in self._ctasks[tj]:
                        ft_aj = max(ft_aj, cst + self.RP[t.id])
                else:
                    tst += self._RT[t.id]
                    if t in self._ctasks[ti]:
                        ft_ai = max(ft_ai, tst + self.PT[t.id])
                    if t in self._ctasks[tj]:
                        ft_aj = max(ft_aj, tst + self.PT[t.id])

        return max(ft_ai, ft_aj), ft_ai, ft_aj
