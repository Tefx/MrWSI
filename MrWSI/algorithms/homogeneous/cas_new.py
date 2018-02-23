from math import inf
from copy import copy
from collections import deque
from functools import reduce
import numpy as np
from .base import Heuristic, memo


class NewCAS(Heuristic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pld = np.full(self.N, False)

        toporder = self._topsort()
        self.TD = np.full((self.N, self.N), -1, dtype=int)
        self.TC = [{t} for t in self.problem.tasks]
        self.RP = np.array(self._RT)
        for t in reversed(toporder):
            self.calculate_RP(t)
        self.RA = np.array(self._RT)
        for t in toporder:
            self.calculate_RA(t)

    def rt_sum(self, ts):
        return sum(self._RT[t.id] for t in ts)

    # def _eft(self, task):
        # if not task.in_degree:
        # return -inf, -inf
        # elif not task.out_degree:
        # return inf, inf
        # else:
        # return -self.RP[task.id], -sum(self.CT(c) for c in task.in_comms)

    def _eft(self, task):
        return -self.RP[task.id]

    @memo
    def sorted_prevs(self, task):
        return list(sorted(task.prevs(), key=lambda t: self.RA[t.id]))

    def sort_tasks(self):
        self.remaining_tasks = deque(sorted(self.problem.tasks, key=self._eft))
        # print(self.remaining_tasks)
        # print([self.RP[t.id] for t in self.remaining_tasks])
        while self.remaining_tasks:
            task = self.remaining_tasks.popleft()
            self.edge_tasks = set(
                t for t in self.problem.tasks
                if self._pld[t.id] and
                any(not self._pld[_t.id] for _t in t.succs() if _t != task))
            # print(self.edge_tasks)
            yield task
            self._pld[task.id] = True

    def default_fitness(self):
        return inf, inf, [inf]

    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        all_ft = []
        cur_ft = st + self.RP[task.id]
        for t in self.edge_tasks:
            if self.PL_m(t) == machine:
                fti, ftj = self.RP2s(t, task, self.ST(t), st)
            else:
                rpi, rpj = self.RP2m(t, task)
                fti, ftj = self.ST(t) + rpi, st + rpj
            # print(t, fti, ftj)
            ft_w = max(ft_w, fti, ftj)
            all_ft.append(fti)
            cur_ft = max(cur_ft, ftj)
        all_ft.append(cur_ft)
        return ft_w, st, sorted(all_ft, reverse=True)

    def calculate_RP(self, task):
        self.TD[task.id, task.id] = 0
        st_t = st_c = self._RT[task.id]
        for t in sorted(task.succs(), key=lambda _t: -self.RP[_t.id]):
            ct = self._CT[task.id, t.id]
            t_l = max(self.rt_sum(self.TC[task.id] | self.TC[t.id]),
                      st_t + self.RP[t.id])
            t_r = st_c + ct + self.RP[t.id]
            if (t_l, st_t) <= (t_r, st_c + ct):
                st_t += self._RT[t.id]
                self.TC[task.id] |= self.TC[t.id]
                self.TD[task.id, t.id] = st_t - self._RT[task.id]
                self.RP[task.id] = max(self.RP[task.id], t_l)
            else:
                st_c += ct
                self.TD[task.id, t.id] = st_c + \
                    self._RT[t.id] - self._RT[task.id]
                self.RP[task.id] = max(self.RP[task.id], t_r)

    def RP2s(self, ti, tj, sti, stj):
        rpi, rpj = self.RP[ti.id], self.RP[tj.id]
        if sti < stj:
            rpi = max(rpi, self.rt_sum(self.TC[ti.id] | self.TC[tj.id]))
        else:
            rpj = max(rpj, self.rt_sum(self.TC[ti.id] | self.TC[tj.id]))
        return sti + rpi, stj + rpj


class RP2m_TD(NewCAS):
    def td(self, ti, tj):
        if self.TD[ti.id, tj.id] < 0:
            mts = [t for t in tj.prevs()
                   if self.is_successor(ti, t) or ti == t]
            self.TD[ti.id, tj.id] = max(
                self.td(ti, t) + self.td(t, tj) for t in mts)
        return self.TD[ti.id, tj.id]

    def rp2m_est2(self, ti, task, ft, tm, ct):
        flag = False
        for t in sorted(filter(tm.__contains__, task.prevs()),
                        key=lambda _t: self.td(ti, _t)):
            flag = True
            ct = max(ct, ft + self.td(ti, t)) + self._CT[t.id, task.id]
        return ct, flag

    @memo
    def RP2m(self, ti, tj):
        tmi, tmj = copy(self.TC[ti.id]), copy(self.TC[tj.id])
        ci, cj = fti, ftj = self.RA[ti.id], self.RA[tj.id]
        r = 0
        for t in self.remaining_tasks:
            if t == tj:
                tmi.discard(t)
            elif t == ti or self._RT[t.id] == 0:
                tmj.discard(t)
            elif t in tmi and t in tmj:
                tci, fi = self.rp2m_est2(ti, t, fti, tmi, ci)
                tcj, fj = self.rp2m_est2(tj, t, ftj, tmj, cj)
                if not fi:
                    tmi.discard(t)
                elif not fj:
                    tmj.discard(t)
                elif tcj <= tci:
                    r = max(r, tcj + self.RP[t.id])
                    cj = max(cj, tcj)
                    tmj.discard(t)
                else:
                    r = max(r, tci + self.RP[t.id])
                    ci = max(ci, tci)
                    tmi.discard(t)
        rpmi = max(self.RP[ti.id], r - fti + self._RT[ti.id])
        rpmj = max(self.RP[tj.id], r - ftj + self._RT[tj.id])
        return rpmi, rpmj


class RP2m_RA(NewCAS):
    @memo
    def RP2m(self, ti, tj):
        tmi, tmj = self.TC[ti.id] - {tj}, self.TC[tj.id] - {ti}
        ci, cj = fti, ftj = self.RA[ti.id], self.RA[tj.id]
        r = 0
        for t in self.remaining_tasks:
            if self._RT[t.id] and t in tmi and t in tmj:
                pti = filter(tmi.__contains__, self.sorted_prevs(t))
                ptj = filter(tmj.__contains__, self.sorted_prevs(t))
                if not pti:
                    tmi.discard(t)
                elif not ptj:
                    tmj.discard(t)
                else:
                    def add_comm(ct, _t):
                        return max(ct, self.RA[_t.id]) + self._CT[_t.id, t.id]
                    tci = reduce(add_comm, pti, ci)
                    tcj = reduce(add_comm, ptj, cj)
                    if tcj <= tci:
                        r = max(r, tcj + self.RP[t.id])
                        cj = tcj
                        tmj.discard(t)
                    else:
                        r = max(r, tci + self.RP[t.id])
                        ci = tci
                        tmi.discard(t)
        rpi = max(self.RP[ti.id], r - fti + self._RT[ti.id])
        rpj = max(self.RP[tj.id], r - ftj + self._RT[tj.id])
        # print(ti, tj, rpi, rpj)
        return rpi, rpj


class RP2m_RA2(NewCAS):
    def sort_tasks(self):
        self.scheduling_list = list(sorted(self.problem.tasks, key=self._eft))
        for task in self.scheduling_list:
            self.edge_tasks = set(
                t for t in self.problem.tasks
                if self._pld[t.id] and
                any(not self._pld[_t.id] for _t in t.succs() if _t != task))
            yield task
            self._pld[task.id] = True

    @memo
    def RP2m(self, ti, tj):
        tmi, tmj = self.TC[ti.id] - {tj}, self.TC[tj.id] - {ti}
        ci, cj = fti, ftj = self.RA[ti.id], self.RA[tj.id]
        r = 0
        for t in self.scheduling_list:
            if self._RT[t.id]:
                def add_comm(ct, _t):
                    return max(ct, self.RA[_t.id]) + self._CT[_t.id, t.id]
                if t in tmi and t != ti:
                    if self._pld[t.id] and self.PL_m(t) != self.PL_m(ti):
                        tmi.discard(t)
                    pti = list(filter(tmi.__contains__, self.sorted_prevs(t)))
                    if pti:
                        tci = reduce(add_comm, pti, ci)
                    else:
                        tmi.discard(t)
                if t in tmj and t != tj:
                    ptj = list(filter(tmj.__contains__, self.sorted_prevs(t)))
                    if ptj:
                        tcj = reduce(add_comm, ptj, cj)
                    else:
                        tmj.discard(t)
                if t in tmi and t in tmj:
                    if tcj <= tci:
                        r = max(r, tcj + self.RP[t.id])
                        cj = tcj
                        tmj.discard(t)
                    else:
                        r = max(r, tci + self.RP[t.id])
                        ci = tci
                        tmi.discard(t)
        rpi = max(self.RP[ti.id], r - fti + self._RT[ti.id])
        rpj = max(self.RP[tj.id], r - ftj + self._RT[tj.id])
        print(ti, tj, rpi, rpj)
        return rpi, rpj


class RAFromRP(NewCAS):
    def calculate_RA(self, task):
        self.RA[task.id] = \
            max([self.RA[t.id] + self.TD[t.id, task.id] for t in task.prevs()],
                default=self._RT[task.id])


class RAFromEntry(NewCAS):
    def calculate_RA(self, task):
        st = 0
        A = np.empty(self.N, int)
        B = np.empty(self.N, int)
        for t in self.sorted_prevs(task):
            A[t.id] = max(self.RA[t.id] - st, 0) + self._CT[t.id, task.id]
            B[t.id] = max(st - self.RA[t.id], 0)
            st += A[t.id]
        k, d = st, 0
        for t in reversed(self.sorted_prevs(task)):
            d = max(d, min(A[t.id], k))
            k = min(k, B[t.id])
        self.RA[task.id] = st + self._RT[task.id] - d
