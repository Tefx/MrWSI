from math import inf
from statistics import mean
from copy import copy
from itertools import chain
from collections import Counter

from .base import memo, Heuristic, COMM_INPUT, COMM_OUTPUT, Machine


class CAWS(Heuristic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RP = [None] * self.problem.num_tasks
        self.RPr = [None] * self.problem.num_tasks
        self.TCM = [None] * self.problem.num_tasks
        self.TDS = {}
        self._pld = [False] * self.problem.num_tasks
        self.ready_tasks = None
        self.edge_tasks = None

    def rt_sum(self, ts):
        return sum(self._RT[t.id] for t in list(ts))

    def preprocess(self):
        toporder = self._topsort()
        for t in reversed(toporder):
            self.calculate_RP1(t)

    def select_task(self):
        return max(self.ready_tasks, key=lambda t: (self.RP[t.id], len(t.prevs())))

    def sort_tasks(self):
        self.preprocess()
        self.ready_tasks = set(
            t for t in self.problem.tasks if t.in_degree == 0)
        self.edge_tasks = set()
        ids = [t.in_degree for t in self.problem.tasks]
        ods = [t.out_degree for t in self.problem.tasks]
        while self.ready_tasks:
            task = self.select_task()
            yield task
            self.ready_tasks.remove(task)
            for t in task.succs():
                ids[t.id] -= 1
                if not ids[t.id]:
                    self.ready_tasks.add(t)
            if ods[task.id]:
                self.edge_tasks.add(task)
            for t in task.prevs():
                ods[t.id] -= 1
                if not ods[t.id]:
                    self.edge_tasks.remove(t)
            self._pld[task.id] = True

    def default_fitness(self):
        return inf, inf, [inf]

    def calculate_RP1(self, task):
        self.TCM[task.id] = Counter([task])
        self.RPr[task.id] = self._RT[task.id]
        self.TDS[(task.id, task.id)] = 0
        st_t = st_c = self._RT[task.id]
        for t in sorted(task.succs(), key=lambda _t: -self.RP[_t.id]):
            ct = self._CT[task.id, t.id]
            t_l = max(
                self.rt_sum(self.TCM[task.id] | self.TCM[t.id]),
                st_t + self.RPr[t.id],
                self.RPr[task.id])
            t_r = max(
                st_c + ct + self.RP[t.id],
                self.rt_sum(self.TCM[task.id]),
                self.RPr[task.id])
            if (t_l, st_t) <= (t_r, st_c + ct):
                if self.RPr[t.id] > self._RT[t.id]:
                    self.RPr[task.id] = max(
                        self.RPr[task.id], st_t + self.RPr[t.id])
                st_t += self._RT[t.id]
                self.TCM[task.id].update(self.TCM[t.id])
                self.TDS[(task.id, t.id)] = st_t - self._RT[task.id]
            else:
                self.RPr[task.id] = max(
                    self.RPr[task.id], st_c + ct + self.RP[t.id])
                st_c += ct
                self.TDS[(task.id, t.id)] = st_c + \
                    self._RT[t.id] - self._RT[task.id]
        self.RP[task.id] = max(
            self.rt_sum(self.TCM[task.id]),
            self.RPr[task.id])

    @memo
    def td(self, ti, tj):
        if (ti.id, tj.id) not in self.TDS:
            mts = [t for t in tj.prevs() if self.is_successor(ti, t) or ti == t]
            self.TDS[(ti.id, tj.id)] = max(
                self.td(ti, t) + self.td(t, tj) for t in mts)
        return self.TDS[(ti.id, tj.id)]

    @memo
    def ra(self, task):
        return max(self.td(t, task) for t in self.entry_tasks)

    def rp2m_est2(self, ti, task, tmi, cti):
        ci = cti
        ts = [t for t in task.prevs() if t in tmi]
        for t in sorted(ts, key=self.ra):
            ci = max(ci, self.ra(ti) + self.td(ti, t)) + \
                self._CT[t.id, task.id]
        return ci

    def tcm_substract(self, tm, task):
        k = tm[task]
        tm2 = self.TCM[task.id]
        for t in list(tm2):
            tm[t] -= tm2[t] * k
            if not tm[t]:
                del tm[t]
        return tm

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ci, cj = rai, raj = self.ra(ti), self.ra(tj)
        ri = self.RPr[ti.id] - self._RT[ti.id]
        rj = self.RPr[tj.id] - self._RT[tj.id]
        T_shrd = list(self.TCM[ti.id] & self.TCM[tj.id])
        for t in sorted(T_shrd, key=lambda _t: -self.RP[_t.id]):
            if t == ti:
                tmj = self.tcm_substract(tmj, t)
            elif t == tj:
                tmi = self.tcm_substract(tmi, t)
            elif self._RT[t.id] == 0:
                tmj = self.tcm_substract(tmj, t)
            elif t in tmi and t in tmj:
                tci = self.rp2m_est2(ti, t, tmi, ci)
                tcj = self.rp2m_est2(tj, t, tmj, cj)
                if tcj <= tci:  # Put on i
                    ri = max(ri, tcj + self._RT[t.id] - rai)
                    rj = max(rj, tcj + self.RP[t.id] - raj)
                    cj = tcj
                    tmj = self.tcm_substract(tmj, t)
                else:
                    ri = max(ri, tci + self.RP[t.id] - rai)
                    rj = max(rj, tci + self._RT[t.id] - raj)
                    ci = tci
                    tmi = self.tcm_substract(tmi, t)
        rpmi = max(self.rt_sum(tmi), self._RT[ti.id] + ri)
        rpmj = max(self.rt_sum(tmj), self._RT[tj.id] + rj)
        return rpmi, rpmj

    def calculate_RP2s(self, ti, tj, sti, stj):
        tmi = set(self.TCM[ti.id])
        tmj = set(self.TCM[tj.id])
        if sti < stj:
            tmi |= tmj
        else:
            tmj |= tmi
        fti = sti + max(self.rt_sum(tmi), self.RPr[ti.id])
        ftj = stj + max(self.rt_sum(tmj), self.RPr[tj.id])
        # if stj > sti:
        # fti = sti + max(self.rt_sum(self.TCM[ti.id] | self.TCM[tj.id]),
        # self.RPr[ti.id])
        # ftj = stj + max(self.rt_sum(self.TCM[tj.id]),
        # self.RPr[tj.id])
        # else:
        # ftj, fti = self.calculate_RP2s(tj, ti, stj, sti)
        return fti, ftj

    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        all_ft = []
        cur_ft = st + self.RP[task.id]
        for t in self.edge_tasks:
            if all(self._pld[_t.id] for _t in t.succs() if _t != task):
                continue
            if self.is_successor(t, task):
                fti, ftj = self.ST(t) + self.RP[t.id], st + self.RP[task.id]
            elif self.PL_m(t) == machine:
                fti, ftj = self.calculate_RP2s(t, task, self.ST(t), st)
            else:
                rpi, rpj = self.calculate_RP2m(t, task)
                fti, ftj = self.ST(t) + rpi, st + rpj
            ft_w = max(ft_w, fti, ftj)
            all_ft.append(fti)
            cur_ft = max(cur_ft, ftj)
        all_ft.append(cur_ft)
        return ft_w, st, sorted(all_ft, reverse=True)


class RAEst(CAWS):
    def rp2m_est2(self, ti, task, tmi, cti):
        ci = cti
        ts = [t for t in task.prevs() if t in tmi]
        for t in sorted(ts, key=self.ra):
            ci = max(ci, self.ra(t)) + self._CT[t.id, task.id]
        return ci


class ForwardRA(RAEst):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RA = [None] * self.problem.num_tasks

    def preprocess(self):
        toporder = self._topsort()
        for t in toporder:
            self.calculate_RA(t)
        for t in reversed(toporder):
            self.calculate_RP1(t)

    def calculate_RA(self, task):
        comms = sorted(task.in_comms, key=lambda c: self.RA[c.from_task.id])
        st_t, st_c = 0, 0
        for c in comms:
            t = c.from_task
            if self.CT(c) > 0:
                st_c = max(st_c, self.RA[t.id]) + self.CT(c)
            else:
                st_t = max(st_t, self.RA[t.id])
        st = max(st_t, st_c)
        for t_s in task.prevs():
            st_t, st_c = self.RA[t_s.id], 0
            for c in comms:
                t = c.from_task
                if t != t_s:
                    if self.CT(c) > 0:
                        st_c = max(st_c, self.RA[t.id]) + self.CT(c)
                    else:
                        st_t = max(st_t, self.RA[t.id])
            st = min(st, max(st_t, st_c))
        self.RA[task.id] = st + self._RT[task.id]

    def ra(self, task):
        return self.RA[task.id]


class SimpleFitness(CAWS):
    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        for t in self.edge_tasks:
            if self.is_successor(t, task):
                fti, ftj = self.ST(t) + self.RP[t.id], st + self.RP[task.id]
            elif self.PL_m(t) == machine:
                fti, ftj = self.calculate_RP2s(t, task, self.ST(t), st)
            else:
                rpi, rpj = self.calculate_RP2m(t, task)
                fti, ftj = self.ST(t) + rpi, st + rpj
            ft_w = max(ft_w, fti, ftj)
        return ft_w, st


class NoRPr(CAWS):
    def calculate_RP1(self, task):
        self.TCM[task.id] = Counter([task])
        self.TDS[(task.id, task.id)] = 0
        st_t = st_c = self._RT[task.id]
        rp = self._RT[task.id]
        for t in sorted(task.succs(), key=lambda _t: -self.RP[_t.id]):
            ct = self._CT[task.id, t.id]
            t_l = max(
                self.rt_sum(self.TCM[task.id] | self.TCM[t.id]),
                st_t + self.RP[t.id])
            t_r = st_c + ct + self.RP[t.id]
            if (t_l, st_t) <= (t_r, st_c + ct):
                st_t += self._RT[t.id]
                self.TCM[task.id].update(self.TCM[t.id])
                self.TDS[(task.id, t.id)] = st_t - self._RT[task.id]
                rp = max(rp, t_l)
            else:
                st_c += ct
                self.TDS[(task.id, t.id)] = st_c + \
                    self._RT[t.id] - self._RT[task.id]
                rp = max(rp, t_r)
        self.RP[task.id] = rp

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ci, cj = rai, raj = self.ra(ti), self.ra(tj)
        ri = self.RP[ti.id] - self._RT[ti.id]
        rj = self.RP[tj.id] - self._RT[tj.id]
        T_shrd = list(self.TCM[ti.id] & self.TCM[tj.id])
        for t in sorted(T_shrd, key=lambda _t: -self.RP[_t.id]):
            if t == ti:
                tmj = self.tcm_substract(tmj, t)
            elif t == tj:
                tmi = self.tcm_substract(tmi, t)
            elif self._RT[t.id] == 0:
                tmj = self.tcm_substract(tmj, t)
            elif t in tmi and t in tmj:
                tci = self.rp2m_est2(ti, t, tmi, ci)
                tcj = self.rp2m_est2(tj, t, tmj, cj)
                if tcj <= tci:  # Put on i
                    ri = max(ri, tcj + self._RT[t.id] - rai)
                    rj = max(rj, tcj + self.RP[t.id] - raj)
                    cj = tcj
                    tmj = self.tcm_substract(tmj, t)
                else:
                    ri = max(ri, tci + self.RP[t.id] - rai)
                    rj = max(rj, tci + self._RT[t.id] - raj)
                    ci = tci
                    tmi = self.tcm_substract(tmi, t)
        rpmi = max(self.rt_sum(tmi), self._RT[ti.id] + ri)
        rpmj = max(self.rt_sum(tmj), self._RT[tj.id] + rj)
        return rpmi, rpmj

    def calculate_RP2s(self, ti, tj, sti, stj):
        tmi = set(self.TCM[ti.id])
        tmj = set(self.TCM[tj.id])
        if sti < stj:
            tmi |= tmj
        else:
            tmj |= tmi
        fti = sti + max(self.rt_sum(tmi), self.RP[ti.id])
        ftj = stj + max(self.rt_sum(tmj), self.RP[tj.id])
        return fti, ftj


class C1(NoRPr):
    def rp2m_est2(self, ti, task, st, tm, ct):
        ts = [t for t in task.prevs() if t in tm]
        for t in sorted(ts, key=lambda _t: self.RP[_t.id]):
            ct = max(ct, st + self._RT[ti.id],
                     self.td(ti, t)) + self._CT[t.id, task.id]
        return ct

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        sti, stj = self.ra(ti) - self._RT[ti.id], self.ra(tj) - self._RT[tj.id]
        ci, cj = sti + self._RT[ti.id], stj + self._RT[tj.id]
        r = 0
        T_shrd = list(self.TCM[ti.id] & self.TCM[tj.id])
        for t in sorted(T_shrd, key=lambda _t: -self.RP[_t.id]):
            if t == ti:
                tmj = self.tcm_substract(tmj, t)
            elif t == tj:
                tmi = self.tcm_substract(tmi, t)
            elif self._RT[t.id] == 0:
                tmj = self.tcm_substract(tmj, t)
            elif t in tmi and t in tmj:
                tci = self.rp2m_est2(ti, t, sti, tmi, ci)
                tcj = self.rp2m_est2(tj, t, stj, tmj, cj)
                if tcj <= tci:
                    r = max(r, tcj + self.RP[t.id])
                    cj = tcj
                    tmj = self.tcm_substract(tmj, t)
                else:
                    r = max(r, tci + self.RP[t.id])
                    ci = tci
                    tmi = self.tcm_substract(tmi, t)
        rpmi = max(self.rt_sum(tmi), self.RP[ti.id], r - sti)
        rpmj = max(self.rt_sum(tmj), self.RP[tj.id], r - stj)
        return rpmi, rpmj


class C2(NoRPr):
    def rp2m_est2(self, ti, task, ft, tm, ct):
        ts = [t for t in task.prevs() if t in tm]
        for t in sorted(ts, key=lambda _t: self.RP[_t.id]):
            ct = max(ct, ft + self.td(ti, t)) + self._CT[t.id, task.id]
        return ct

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ci, cj = fti, ftj = self.ra(ti), self.ra(tj)
        r = ri = rj = 0
        ts = [t for t in self.problem.tasks if self.is_successor(
            ti, t) or self.is_successor(tj, t)]
        ts.extend([ti, tj])
        for t in sorted(self.problem.tasks, key=lambda _t: -self.RP[_t.id]):
            if t == ti:
                tmj = self.tcm_substract(tmj, t)
            elif t == tj:
                tmi = self.tcm_substract(tmi, t)
            elif self._RT[t.id] == 0:
                tmj = self.tcm_substract(tmj, t)
            elif t in tmi and t in tmj:
                tci = self.rp2m_est2(ti, t, fti, tmi, ci)
                tcj = self.rp2m_est2(tj, t, ftj, tmj, cj)
                if tcj <= tci:
                    r = max(r, tcj + self.RP[t.id])
                    cj = tcj
                    tmj = self.tcm_substract(tmj, t)
                else:
                    r = max(r, tci + self.RP[t.id])
                    ci = tci
                    tmi = self.tcm_substract(tmi, t)
            else:
                if self.is_successor(ti, t):
                    ri = max(ri,
                             self._RT[ti.id] + self.td(ti, t) +
                             self.RP[t.id] - self._RT[t.id])
                if self.is_successor(tj, t):
                    rj = max(rj,
                             self._RT[tj.id] + self.td(tj, t) +
                             self.RP[t.id] - self._RT[t.id])
        rpmi = max(self.rt_sum(tmi), ri, r - fti + self._RT[ti.id])
        rpmj = max(self.rt_sum(tmj), rj, r - ftj + self._RT[tj.id])
        return rpmi, rpmj


class C3(NoRPr):
    def _eft(self, task):
        if not task.in_degree:
            return inf, inf
        elif not task.out_degree:
            return -inf, -inf
        else:
            return self.RP[task.id], sum(self.CT(c) for c in task.in_comms)

    def sort_tasks(self):
        self.preprocess()
        self.edge_tasks = set()
        ods = {}
        self.scheduling_list = sorted(self.problem.tasks,
                                      key=self._eft, reverse=True)
        for task in self.scheduling_list:
            yield task
            if task.out_degree:
                self.edge_tasks.add(task)
                ods[task] = task.out_degree
            for t in task.prevs():
                ods[t] -= 1
                if not ods[t]:
                    self.edge_tasks.remove(t)
            self._pld[task.id] = True

    def rp2m_est2(self, ti, task, ft, tm, ct):
        ts = [t for t in task.prevs() if t in tm]
        for t in sorted(ts, key=lambda _t: self.RP[_t.id]):
            ct = max(ct, ft + self.td(ti, t)) + self._CT[t.id, task.id]
        return ct

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ci, cj = fti, ftj = self.ra(ti), self.ra(tj)
        r = ri = rj = 0
        for t in self.scheduling_list:
            if t == ti:
                tmj = self.tcm_substract(tmj, t)
            elif t == tj:
                tmi = self.tcm_substract(tmi, t)
            elif self._RT[t.id] == 0:
                tmj = self.tcm_substract(tmj, t)
            elif t in tmi and t in tmj:
                tci = self.rp2m_est2(ti, t, fti, tmi, ci)
                tcj = self.rp2m_est2(tj, t, ftj, tmj, cj)
                if tcj <= tci:
                    r = max(r, tcj + self.RP[t.id])
                    cj = tcj
                    tmj = self.tcm_substract(tmj, t)
                else:
                    r = max(r, tci + self.RP[t.id])
                    ci = tci
                    tmi = self.tcm_substract(tmi, t)
            else:
                if self.is_successor(ti, t):
                    ri = max(ri,
                             self._RT[ti.id] + self.td(ti, t) +
                             self.RP[t.id] - self._RT[t.id])
                if self.is_successor(tj, t):
                    rj = max(rj,
                             self._RT[tj.id] + self.td(tj, t) +
                             self.RP[t.id] - self._RT[t.id])
        rpmi = max(self.rt_sum(tmi), ri, r - fti + self._RT[ti.id])
        rpmj = max(self.rt_sum(tmj), rj, r - ftj + self._RT[tj.id])
        return rpmi, rpmj


class C4(C3):
    def rp2m_est2(self, ti, task, ft, tm, ct):
        ts = [t for t in task.prevs() if t in tm]
        lt = ft
        for t in sorted(ts, key=lambda _t: self.RP[_t.id]):
            ct = max(ct, ft + self.td(ti, t)) + self._CT[t.id, task.id]
            lt = max(lt, ft + self.td(ti, t))
        return lt, ct

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ci, cj = fti, ftj = self.ra(ti), self.ra(tj)
        r = ri = rj = 0
        for t in self.scheduling_list:
            if t == ti:
                tmj = self.tcm_substract(tmj, t)
            elif t == tj:
                tmi = self.tcm_substract(tmi, t)
            elif self._RT[t.id] == 0:
                tmj = self.tcm_substract(tmj, t)
            elif t in tmi and t in tmj:
                ft0 = 0
                for _t in t.prevs():
                    if _t not in tmi and _t not in tmj:
                        if self.is_successor(ti, _t) or ti == _t:
                            ft0 = max(ft0, fti + self.td(ti, _t))
                        if self.is_successor(tj, _t) or tj == _t:
                            ft0 = max(ft0, ftj + self.td(tj, _t))
                lti, cti = self.rp2m_est2(ti, t, fti, tmi, ci)
                ltj, ctj = self.rp2m_est2(tj, t, ftj, tmj, cj)
                tci = max(lti, ft0, ctj)
                tcj = max(ltj, ft0, cti)
                if tci <= tcj:
                    r = max(r, tci + self.RP[t.id])
                    cj = ctj
                    tmj = self.tcm_substract(tmj, t)
                else:
                    r = max(r, tcj + self.RP[t.id])
                    ci = cti
                    tmi = self.tcm_substract(tmi, t)
            else:
                if self.is_successor(ti, t):
                    ri = max(ri,
                             self._RT[ti.id] + self.td(ti, t) +
                             self.RP[t.id] - self._RT[t.id])
                if self.is_successor(tj, t):
                    rj = max(rj,
                             self._RT[tj.id] + self.td(tj, t) +
                             self.RP[t.id] - self._RT[t.id])
        rpmi = max(self.rt_sum(tmi), ri, r - fti + self._RT[ti.id])
        rpmj = max(self.rt_sum(tmj), rj, r - ftj + self._RT[tj.id])
        return rpmi, rpmj


class C5(C3):
    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ci, cj = fti, ftj = self.ra(ti), self.ra(tj)
        r = 0
        for t in self.scheduling_list:
            if t == ti:
                tmj = self.tcm_substract(tmj, t)
            elif t == tj:
                tmi = self.tcm_substract(tmi, t)
            elif self._RT[t.id] == 0:
                tmj = self.tcm_substract(tmj, t)
            elif t in tmi and t in tmj:
                tci = self.rp2m_est2(ti, t, fti, tmi, ci)
                tcj = self.rp2m_est2(tj, t, ftj, tmj, cj)
                if tcj <= tci:
                    r = max(r, tcj + self.RP[t.id])
                    cj = tcj
                    tmj = self.tcm_substract(tmj, t)
                else:
                    r = max(r, tci + self.RP[t.id])
                    ci = tci
                    tmi = self.tcm_substract(tmi, t)
        rpmi = max(self.RP[ti.id], r - fti + self._RT[ti.id])
        rpmj = max(self.RP[tj.id], r - ftj + self._RT[tj.id])
        return rpmi, rpmj


class C6(C3):
    def solve(self):
        for task in self.sort_tasks():
            self._order.append(task)
            machines = self.available_machines()
            mps = [(m, self.placement_on(task, m)) for m in machines]
            self.ft_avg = mean(pls[-1] for m, pls in mps) + self._RT[task.id]
            placement_bst, fitness_bst = None, self.default_fitness()
            for machine, placement in mps:
                assert machine.vm_type.capacities >= task.demands()
                fitness = self.fitness(task, *placement)
                if "fit" in self.log:
                    print(task, machine, fitness, placement)
                if self.compare_fitness(fitness, fitness_bst):
                    placement_bst, fitness_bst = placement, fitness
            self.perform_placement(task, placement_bst)
        self.have_solved = True

        if "alg" in self.log:
            self.log_alg("./")

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ci, cj = fti, ftj = self.FT(ti), self.ft_avg
        r = 0
        for t in self.scheduling_list:
            if t == ti:
                tmj = self.tcm_substract(tmj, t)
            elif t == tj:
                tmi = self.tcm_substract(tmi, t)
            elif self._RT[t.id] == 0:
                tmj = self.tcm_substract(tmj, t)
            elif t in tmi and t in tmj:
                tci = self.rp2m_est2(ti, t, fti, tmi, ci)
                tcj = self.rp2m_est2(tj, t, ftj, tmj, cj)
                if tcj <= tci:
                    r = max(r, tcj + self.RP[t.id])
                    cj = tcj
                    tmj = self.tcm_substract(tmj, t)
                else:
                    r = max(r, tci + self.RP[t.id])
                    ci = tci
                    tmi = self.tcm_substract(tmi, t)
        rpmi = max(self.RP[ti.id], r - fti + self._RT[ti.id])
        rpmj = max(self.RP[tj.id], r - ftj + self._RT[tj.id])
        return rpmi, rpmj


class C7(C3):
    def calculate_RP2m(self, ti, tj, stj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ci, cj = fti, ftj = self.FT(ti), stj + self._RT[tj.id]
        r = 0
        for t in self.scheduling_list:
            if t == ti:
                tmj = self.tcm_substract(tmj, t)
            elif t == tj:
                tmi = self.tcm_substract(tmi, t)
            elif self._RT[t.id] == 0:
                tmj = self.tcm_substract(tmj, t)
            elif t in tmi and t in tmj:
                tci = self.rp2m_est2(ti, t, fti, tmi, ci)
                tcj = self.rp2m_est2(tj, t, ftj, tmj, cj)
                if tcj <= tci:
                    r = max(r, tcj + self.RP[t.id])
                    cj = tcj
                    tmj = self.tcm_substract(tmj, t)
                else:
                    r = max(r, tci + self.RP[t.id])
                    ci = tci
                    tmi = self.tcm_substract(tmi, t)
        rpmi = max(self.RP[ti.id], r - fti + self._RT[ti.id])
        rpmj = max(self.RP[tj.id], r - ftj + self._RT[tj.id])
        return rpmi, rpmj

    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        all_ft = []
        cur_ft = st + self.RP[task.id]
        for t in self.edge_tasks:
            if all(self._pld[_t.id] for _t in t.succs() if _t != task):
                continue
            if self.is_successor(t, task):
                fti, ftj = self.ST(t) + self.RP[t.id], st + self.RP[task.id]
            elif self.PL_m(t) == machine:
                fti, ftj = self.calculate_RP2s(t, task, self.ST(t), st)
            else:
                rpi, rpj = self.calculate_RP2m(t, task, st)
                fti, ftj = self.ST(t) + rpi, st + rpj
            ft_w = max(ft_w, fti, ftj)
            all_ft.append(fti)
            cur_ft = max(cur_ft, ftj)
        all_ft.append(cur_ft)
        return ft_w, st, sorted(all_ft, reverse=True)


class CN(C3):
    def _est_on(self, task, mts, fts, ci, avm):
        st = 0
        for t in sorted(task.prevs(), key=lambda _t: -self.RP[_t.id]):
            if t not in mts and self._CT[t.id, task.id] > 0:
                ci = max(ci, fts[t.id]) + self._CT[t.id, task.id]
                st = max(st, ci)
            else:
                st = max(st, fts[t.id])
        return max(st, avm) + self._RT[t.id], ci

    def _est_not_on(self, task, mts, fts, co):
        st = 0
        for t in sorted(task.prevs(), key=lambda _t: -self.RP[_t.id]):
            if t in mts and self._CT[t.id, task.id] > 0:
                co = max(co, fts[t.id]) + self._CT[t.id, task.id]
                st = max(st, co)
            else:
                st = max(st, fts[t.id])
        return st + self._RT[t.id], co

    def fitness(self, task, machine, comm_pls, st):
        mts = set()
        fts = [None] * self.problem.num_tasks
        avm = 0
        ci = co = 0
        for t in self.scheduling_list:
            if self._pld[t.id]:
                fts[t.id] = self.FT(t)
                if self.PL_m(t) == machine:
                    mts.add(t)
                    avm = max(avm, fts[t.id])
            elif task == t:
                fts[t.id] = st + self._RT[t.id]
                mts.add(t)
                avm = max(avm, fts[t.id])
            else:
                ftl, nci = self._est_on(t, mts, fts, ci, avm)
                ftr, nco = self._est_not_on(t, mts, fts, co)
                if ftl <= ftr:
                    mts.add(t)
                    fts[t.id] = ftl
                    avm = max(avm, fts[t.id])
                    ci = nci
                else:
                    fts[t.id] = ftr
                    co = nco
        return max(fts), st


class CN2(C3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m = min(self.problem.num_tasks, self.problem.platform_limits[0])
        self.machines = [Machine(self.problem.vm_type, self.context)
                         for _ in range(m)]

    def available_machines(self):
        return sorted(self.machines, key=lambda m: -len(m.tasks))

    def _meft(self, task, machine, fts, pls, avm, cis, cos):
        st = 0
        ci = cis[machine]
        tcos = {}
        for t in sorted(task.prevs(), key=lambda _t: fts[_t.id]):
            m = pls[t.id]
            if m == machine or self._CT[t.id, task.id] == 0:
                st = max(st, fts[t.id])
            else:
                ci = max(ci, fts[t.id]) + self._CT[t.id, task.id]
                if m not in tcos:
                    tcos[m] = cos[m]
                tcos[m] = max(tcos[m], fts[t.id]) + self._CT[t.id, task.id]
                st = max(st, ci, tcos[m])
        return max(st, avm) + self._RT[task.id], machine, ci, tcos

    def fitness(self, task, machine, comm_pls, st):
        fts = [None] * self.problem.num_tasks
        pls = [None] * self.problem.num_tasks
        avms = {m: 0 for m in self.machines}
        cis = {m: 0 for m in self.machines}
        cos = {m: 0 for m in self.machines}
        for t in self.scheduling_list:
            if self._pld[t.id]:
                m = self.PL_m(t)
                fts[t.id] = self.FT(t)
                pls[t.id] = m
                avms[m] = max(avms[m], fts[t.id])
            elif task == t:
                fts[t.id] = st + self._RT[t.id]
                pls[t.id] = machine
                avms[machine] = max(avms[machine], fts[t.id])
            else:
                pl_b = (inf, None, None, None)
                for m in self.machines:
                    pl = self._meft(t, m, fts, pls, avms[m], cis, cos)
                    if pl[0] < pl_b[0]:
                        pl_b = pl
                ft_b, m_b, ci_b, tcos_b = pl_b
                fts[t.id] = ft_b
                pls[t.id] = m_b
                avms[m_b] = max(avms[m_b], ft_b)
                cis[m_b] = ci_b
                for _m, _co in tcos_b.items():
                    cos[_m] = _co
        return max(fts), st


class C8(C5):
    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        all_ft = []
        cur_ft = st + self.RP[task.id]
        for t in self.edge_tasks:
            if any(not self._pld[_t.id] for _t in t.succs() if _t != task):
                if self.PL_m(t) == machine:
                    fti, ftj = self.calculate_RP2s(t, task, self.ST(t), st)
                else:
                    rpi, rpj = self.calculate_RP2m(t, task)
                    fti, ftj = self.ST(t) + rpi, st + rpj
                ft_w = max(ft_w, fti, ftj)
                all_ft.append(fti)
                cur_ft = max(cur_ft, ftj)
        all_ft.append(cur_ft)
        return ft_w, st, sorted(all_ft, reverse=True)


class C9(C5):
    def calculate_RP2s(self, ti, tj, sti, stj):
        rpi, rpj = self.RP[ti.id], self.RP[tj.id]
        if sti < stj:
            rpi = max(rpi, self.rt_sum(self.TCM[ti.id] | self.TCM[tj.id]))
        else:
            rpj = max(rpj, self.rt_sum(self.TCM[ti.id] | self.TCM[tj.id]))
        return sti + rpi, stj + rpj

    def rp2s(self, task, st, m, edge_tasks):
        sts = [(t, self.ST(t))
               for t in m.tasks if t in edge_tasks] + [(task, st)]
        sts.sort(key=lambda x: x[1], reverse=True)
        fts = []
        tm = set()
        for t, _st in sts:
            tm |= set(self.TCM[t.id])
            ft = _st + max(self.rt_sum(tm), self.RP[t.id])
            if t == task:
                ftt = ft
            else:
                fts.append(ft)
        return fts, ftt

    def rp2m_est_not_on(self, task, tm, ct):
        ts = [t for t in task.prevs() if t in tm]
        for t in sorted(ts, key=lambda _t: self.RP[_t.id]):
            ct = max(ct, self.ra(t)) + self._CT[t.id, task.id]
        return ct

    def rp2m_est_on(self, task, tm, cs):
        ts = [t for t in task.prevs() if t in tm]
        for t in sorted(ts, key=self.ra):
            cs = max(cs, self.ra(t)) + self._CT[t.id, task.id]
        return cs

    @memo
    def rp2m(self, task, m):
        ts = set(t for t in m.tasks if t in self.edge_tasks
                 if any(not self._pld[_t.id] for _t in t.succs() if _t != task))
        tmt = copy(self.TCM[task.id])
        tms = Counter()
        for t in ts:
            tms.update(self.TCM[t.id])
        ct = ftt = self.ra(task)
        cs = min([self.ra(t) for t in ts], default=0)
        r = 0
        for t in self.scheduling_list:
            if t == task:
                tms = self.tcm_substract(tms, t)
            elif t in ts:
                tmt = self.tcm_substract(tmt, t)
            elif self._RT[t.id] == 0:
                tms = self.tcm_substract(tms, t)
            elif t in tmt and t in tms:
                tcs = self.rp2m_est_not_on(t, tmt, ct)
                tct = self.rp2m_est_on(t, tms, cs)
                if tct <= tcs:
                    r = max(r, tct + self.RP[t.id])
                    cs = max(cs, tct)
                    tms = self.tcm_substract(tms, t)
                else:
                    r = max(r, tcs + self.RP[t.id])
                    ct = max(ct, tcs)
                    tmt = self.tcm_substract(tmt, t)
        rpt = max(self.RP[task.id], r - ftt + self._RT[task.id])
        fts = [self.ST(t) + max(self.RP[t.id], r - self.ra(t) + self._RT[t.id]) for t in ts]
        return fts, rpt

    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        all_ft = []
        cur_ft = st + self.RP[task.id]
        edge_tasks = set(t for t in self.edge_tasks if any(
            not self._pld[_t.id] for _t in t.succs() if _t != task))
        for m in self.platform:
            if m == machine:
                fts, ftt = [], 0
                for t in m.tasks:
                    if t in edge_tasks:
                        fti, ftj = self.calculate_RP2s(t, task, self.ST(t), st)
                        fts.append(fti)
                        ftt = max(ftt, ftj)
            else:
                fts, rpt = self.rp2m(task, m)
                ftt = st + rpt
            ft_w = max(ft_w, max(fts, default=0), ftt)
            cur_ft = max(cur_ft, ftt)
            all_ft.extend(fts)
        all_ft.append(cur_ft)
        return ft_w, st, sorted(all_ft, reverse=True)
