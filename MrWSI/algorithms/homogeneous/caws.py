from math import inf
from statistics import mean
from copy import copy
from itertools import chain
from collections import Counter

from .base import memo, Heuristic


class CAS(Heuristic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RA = [None] * self.problem.num_tasks
        self.RP = [None] * self.problem.num_tasks
        self.RPr = [None] * self.problem.num_tasks
        self.TC = [None] * self.problem.num_tasks
        self.DC = [None] * self.problem.num_tasks
        self._pld = [False] * self.problem.num_tasks
        self.ready_tasks = None
        self.edge_tasks = None

    def _rt_sum(self, ts):
        return sum(self._RT[t.id] for t in ts)

    def sort_tasks(self):
        toporder = self._topsort()
        for t in toporder:
            self.calculate_RA(t)
        for t in reversed(toporder):
            self.calculate_RP1(t)
        self.ready_tasks = set(
            t for t in self.problem.tasks if t.in_degree == 0)
        self.edge_tasks = set()
        ids = [t.in_degree for t in self.problem.tasks]
        ods = [t.out_degree for t in self.problem.tasks]
        while self.ready_tasks:
            task = max(self.ready_tasks, key=lambda t: self.RP[t.id])
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

    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        all_ft = []
        cur_ft = st + self.RP[task.id]

        for t in self.edge_tasks:
            if self.is_successor(t, task):
                rpi, rpj = self.RP[t.id], self.RP[task.id]
            elif self.PL_m(t) == machine:
                rpi, rpj = self.calculate_RP2s(t, task)
            else:
                rpi, rpj = self.calculate_RP2m(t, task)
            ft_w = max(ft_w, self.ST(t) + rpi, st + rpj)
            all_ft.append(self.ST(t) + rpi)
            cur_ft = max(cur_ft, st + rpj)
        all_ft.append(cur_ft)
        return ft_w, st, sorted(all_ft, reverse=True)

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

    def calculate_RP1(self, task):
        self.TC[task.id] = {task}
        self.DC[task.id] = Counter()
        self.RPr[task.id] = self._RT[task.id]
        st_t = st_c = self._RT[task.id]
        for c in sorted(task.out_comms,
                        key=lambda c: self.RP[c.to_task.id], reverse=True):
            t = c.to_task
            t_l = max(
                self._rt_sum(self.TC[task.id] | self.TC[t.id]),
                st_t + self.RPr[t.id])
            t_r = st_c + self.CT(c) + self.RP[t.id]
            if (t_l, st_t) <= (t_r, st_c + self.CT(c)):
                if self.RPr[t.id] > self._RT[t.id]:
                    self.RPr[task.id] = max(
                        self.RPr[task.id], st_t + self.RPr[t.id])
                st_t += self._RT[t.id]
                self.TC[task.id].update(self.TC[t.id])
                self.DC[task.id][c] += 1
                self.DC[task.id].update(self.DC[t.id])
            else:
                self.RPr[task.id] = max(self.RPr[task.id], t_r)
                st_c += self.CT(c)
        self.RP[task.id] = max(
            self._rt_sum(self.TC[task.id]),
            self.RPr[task.id])

    @memo
    def calculate_RP2s(self, ti, tj):
        ri = self._RT[ti.id]
        rj = self._RT[tj.id]
        st_t = st_c = self._RT[ti.id] + self._RT[tj.id]
        st_d = self._RT[tj.id]
        Tj, Tij = {tj}, {ti, tj}
        for t in sorted(set(chain(ti.succs(), tj.succs())),
                        key=lambda _t: self.RP[_t.id],
                        reverse=True):
            cti = self._CT[ti.id, t.id]
            ctj = self._CT[tj.id, t.id]
            t_l = max(self._rt_sum(Tij | self.TC[t.id]),
                      st_t + self.RPr[t.id])
            if cti > st_d:
                st = st_c - st_d + cti + ctj
            elif ctj > 0:
                st = st_c + ctj
            else:
                st = self._RT[ti.id] + self._RT[tj.id] - st_d + cti
            t_r = st + self.RP[t.id]
            if (t_l, st_t) <= (t_r, st):
                if cti and self.RPr[t.id] > self._RT[t.id]:
                    ri = max(ri, st_t + self.RPr[t.id])
                if ctj:
                    if self.RPr[t.id] > self._RT[t.id]:
                        rj = max(rj, st_t + self.RPr[t.id] - self._RT[ti.id])
                    Tj.update(self.TC[t.id])
                st_t += self._RT[t.id]
                Tij.update(self.TC[t.id])
            else:
                if cti:
                    ri = max(ri, t_r)
                if ctj:
                    rj = max(rj, t_r - self._RT[ti.id])
                if cti > st_d:
                    st_c += cti + ctj - st_d
                    st_d = 0
                else:
                    st_c += ctj
                    st_d -= cti
        rpsi = max(self._rt_sum(Tij), ri)
        rpsj = max(self._rt_sum(Tj), rj)
        return rpsi, rpsj

    @memo
    def calculate_RP2m(self, ti, tj):
        di, dj = copy(self.DC[ti.id]), copy(self.DC[tj.id])
        li, lj = self._RT[ti.id], self._RT[tj.id]
        ri, rj = self.RPr[ti.id], self.RPr[tj.id]
        sti_c, stj_c = self.RA[ti.id], self.RA[tj.id]
        ati = self.RA[ti.id] - self._RT[ti.id]
        atj = self.RA[tj.id] - self._RT[tj.id]
        for t in sorted(self.TC[ti.id] & self.TC[tj.id],
                        key=lambda _t: self.RP[_t.id],
                        reverse=True):
            if t == ti or t == tj or self._RT[t.id] == 0:
                continue
            dsi = Counter(c for c in +(di - dj) if c.to_task == t)
            dsj = Counter(c for c in +(dj - di) if c.to_task == t)
            if not dsi or not dsj:
                continue
            tci, tcj = fti, ftj = sti_c, stj_c
            for c in sorted(set(dsi), key=lambda _c: self.RA[_c.from_task.id]):
                if self.CT(c) > 0:
                    tci = max(tci, self.RA[c.from_task.id]) + self.CT(c)
                else:
                    fti = max(fti, self.RA[c.from_task.id])
            fti = max(fti, tci)
            for c in sorted(set(dsj), key=lambda _c: self.RA[_c.from_task.id]):
                if self.CT(c) > 0:
                    tcj = max(tcj, self.RA[c.from_task.id]) + self.CT(c)
                else:
                    ftj = max(ftj, self.RA[c.from_task.id])
            ftj = max(ftj, tcj)
            if fti < ftj:  # Put on j
                ri = max(ri, fti + self.RP[t.id] - ati)
                lj = max(lj, fti + self.RP[t.id] - atj)
                sti_c = tci
                di.subtract(self.DC[t.id])
                di.subtract(dsi)
            else:
                rj = max(rj, ftj + self.RP[t.id] - atj)
                li = max(li, ftj + self.RP[t.id] - ati)
                stj_c = tcj
                dj.subtract(self.DC[t.id])
                dj.subtract(dsj)
        rpmi = max(self._RT[ti.id] + self._rt_sum(set(c.to_task for c in +di)),
                   li, ri)
        rpmj = max(self._RT[tj.id] + self._rt_sum(set(c.to_task for c in +dj)),
                   lj, rj)
        return rpmi, rpmj


class CAS2(CAS):
    def _sort_succs2(self, ti, tj, ts):
        w = {t: 0 for t in ts}
        for tx in ts:
            for ty in ts:
                if tx != ty:
                    ct_ix = self._CT[ti.id, tx.id]
                    ct_iy = self._CT[ti.id, ty.id]
                    ct_jx = self._CT[tj.id, tx.id]
                    ct_jy = self._CT[tj.id, ty.id]
                    x, y = ct_ix + ct_jx, ct_iy + ct_jy
                    A, B = self.RP[tx.id], self.RP[ty.id]
                    if min(A, y) + B < A + x:
                        w[ty] += 1
                    if A + min(B, x) < B + y:
                        w[tx] += 1
        return sorted(ts, key=lambda t: (w[t], -self._CT[ti.id, t.id] - self._CT[tj.id, t.id] - self.RP[t.id]))

    @memo
    def calculate_RP2s(self, ti, tj):
        ri = self._RT[ti.id]
        rj = self._RT[tj.id]
        st_t = st_c = self._RT[ti.id] + self._RT[tj.id]
        st_d = self._RT[tj.id]
        Tj, Tij = {tj}, {ti, tj}
        for t in self._sort_succs2(ti, tj, set(chain(ti.succs(), tj.succs()))):
            cti = self._CT[ti.id, t.id]
            ctj = self._CT[tj.id, t.id]
            t_l = max(self._rt_sum(Tij | self.TC[t.id]),
                      st_t + self.RPr[t.id])
            if cti > st_d:
                st = st_c - st_d + cti + ctj
            elif ctj > 0:
                st = st_c + ctj
            else:
                st = self._RT[ti.id] + self._RT[tj.id] - st_d + cti
            t_r = st + self.RP[t.id]
            if (t_l, st_t) <= (t_r, st):
                if cti and self.RPr[t.id] > self._RT[t.id]:
                    ri = max(ri, st_t + self.RPr[t.id])
                if ctj:
                    if self.RPr[t.id] > self._RT[t.id]:
                        rj = max(rj, st_t + self.RPr[t.id] - self._RT[ti.id])
                    Tj.update(self.TC[t.id])
                st_t += self._RT[t.id]
                Tij.update(self.TC[t.id])
            else:
                if cti:
                    ri = max(ri, t_r)
                if ctj:
                    rj = max(rj, t_r - self._RT[ti.id])
                if cti > st_d:
                    st_c += cti + ctj - st_d
                    st_d = 0
                else:
                    st_c += ctj
                    st_d -= cti
        rpsi = max(self._rt_sum(Tij), ri)
        rpsj = max(self._rt_sum(Tj), rj)
        return rpsi, rpsj


class CAS3(CAS):
    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        all_ft = []
        cur_ft = st + self.RP[task.id]

        for t in self.edge_tasks:
            if not any(not self._pld[c.to_task.id] for c in t.out_comms if c.to_task != task):
                continue
            if self.PL_m(t) == machine:
                if self.is_successor(t, task):
                    rpi, rpj = self.RP[t.id], self.RP[task.id]
                else:
                    rpi, rpj = self.calculate_RP2s(t, task)
            else:
                rpi, rpj = self.calculate_RP2m(t, task)
            ft_w = max(ft_w, self.ST(t) + rpi, st + rpj)
            all_ft.append(self.ST(t) + rpi)
            cur_ft = max(cur_ft, st + rpj)
        all_ft.append(cur_ft)
        return ft_w, st, sorted(all_ft, reverse=True)


class CAS4(CAS):
    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        cur_ft = st + self.RP[task.id]

        for t in self.edge_tasks:
            if self.is_successor(t, task):
                rpi, rpj = self.RP[t.id], self.RP[task.id]
            elif self.PL_m(t) == machine:
                rpi, rpj = self.calculate_RP2s(t, task)
            else:
                rpi, rpj = self.calculate_RP2m(t, task)
            ft_w = max(ft_w, self.ST(t) + rpi, st + rpj)
            cur_ft = max(cur_ft, st + rpj)
        return cur_ft, st


class CAS5(CAS):
    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        for t in self.edge_tasks:
            if self.is_successor(t, task):
                rpi, rpj = self.RP[t.id], self.RP[task.id]
            elif self.PL_m(t) == machine:
                rpi, rpj = self.calculate_RP2s(t, task)
            else:
                rpi, rpj = self.calculate_RP2m(t, task)
            ft_w = max(ft_w, self.ST(t) + rpi, st + rpj)
        return ft_w, st


class CAS6(CAS):
    def calculate_RP1(self, task):
        self.TC[task.id] = {task}
        self.DC[task.id] = Counter()
        self.RPr[task.id] = self._RT[task.id]
        st_t = st_c = self._RT[task.id]
        for c in sorted(task.out_comms,
                        key=lambda c: self.RP[c.to_task.id], reverse=True):
            t = c.to_task
            t_l = max(
                self._rt_sum(self.TC[task.id] | self.TC[t.id]),
                st_t + self.RPr[t.id],
                self.RPr[task.id])
            t_r = max(
                st_c + self.CT(c) + self.RP[t.id],
                self._rt_sum(self.TC[task.id]),
                self.RPr[task.id])
            if (t_l, st_t) <= (t_r, st_c + self.CT(c)):
                if self.RPr[t.id] > self._RT[t.id]:
                    self.RPr[task.id] = max(
                        self.RPr[task.id], st_t + self.RPr[t.id])
                st_t += self._RT[t.id]
                self.TC[task.id].update(self.TC[t.id])
                self.DC[task.id][c] += 1
                self.DC[task.id].update(self.DC[t.id])
            else:
                self.RPr[task.id] = max(
                    self.RPr[task.id], st_c + self.CT(c) + self.RP[t.id])
                st_c += self.CT(c)
        self.RP[task.id] = max(
            self._rt_sum(self.TC[task.id]),
            self.RPr[task.id])

    @memo
    def calculate_RP2s(self, ti, tj):
        ri = self._RT[ti.id]
        rj = self._RT[tj.id]
        st_t = st_c = self._RT[ti.id] + self._RT[tj.id]
        st_d = self._RT[tj.id]
        Tj, Tij = {tj}, {ti, tj}
        for t in sorted(set(chain(ti.succs(), tj.succs())),
                        key=lambda _t: self.RP[_t.id],
                        reverse=True):
            cti = self._CT[ti.id, t.id]
            ctj = self._CT[tj.id, t.id]
            t_l = max(self._rt_sum(Tij | self.TC[t.id]),
                      st_t + self.RPr[t.id], ri, rj + self._RT[ti.id])
            if cti > st_d:
                st = st_c - st_d + cti + ctj
            elif ctj > 0:
                st = st_c + ctj
            else:
                st = self._RT[ti.id] + self._RT[tj.id] - st_d + cti
            t_r = max(st + self.RP[t.id],
                      self._rt_sum(Tij), ri, rj + self._RT[ti.id])
            if (t_l, st_t) <= (t_r, st):
                if cti and self.RPr[t.id] > self._RT[t.id]:
                    ri = max(ri, st_t + self.RPr[t.id])
                if ctj:
                    if self.RPr[t.id] > self._RT[t.id]:
                        rj = max(rj, st_t + self.RPr[t.id] - self._RT[ti.id])
                    Tj.update(self.TC[t.id])
                st_t += self._RT[t.id]
                Tij.update(self.TC[t.id])
            else:
                if cti:
                    ri = max(ri, st + self.RP[t.id])
                if ctj:
                    rj = max(rj, st + self.RP[t.id] - self._RT[ti.id])
                if cti > st_d:
                    st_c += cti + ctj - st_d
                    st_d = 0
                else:
                    st_c += ctj
                    st_d -= cti
        rpsi = max(self._rt_sum(Tij), ri)
        rpsj = max(self._rt_sum(Tj), rj)
        return rpsi, rpsj


class CAS7(CAS6):
    @memo
    def calculate_RP2s(self, ti, tj):
        ri = self._RT[ti.id]
        rj = self._RT[tj.id]
        st_t = st_c = self._RT[ti.id] + self._RT[tj.id]
        st_d = self._RT[tj.id]
        Tj, Tij = {tj}, {ti, tj}
        Tr = set()
        for t in sorted(set(chain(ti.succs(), tj.succs())),
                        key=lambda _t: self.RP[_t.id],
                        reverse=True):
            cti = self._CT[ti.id, t.id]
            ctj = self._CT[tj.id, t.id]
            t_l = max(self._rt_sum(Tij | self.TC[t.id]),
                      st_t + self.RPr[t.id], ri, rj + self._RT[ti.id])
            if cti > st_d:
                st = st_c - st_d + cti + ctj
            elif ctj > 0:
                st = st_c + ctj
            else:
                st = self._RT[ti.id] + self._RT[tj.id] - st_d + cti
            t_r = max(st + self.RP[t.id],
                      self._rt_sum(Tij), ri, rj + self._RT[ti.id])
            if (t_l, st_t) <= (t_r, st):
                if cti and self.RPr[t.id] > self._RT[t.id]:
                    ri = max(ri, st_t + self.RPr[t.id])
                if ctj:
                    if self.RPr[t.id] > self._RT[t.id]:
                        rj = max(rj, st_t + self.RPr[t.id] - self._RT[ti.id])
                    Tj.update(self.TC[t.id])
                st_t += self._RT[t.id]
                Tij.update(self.TC[t.id])
            else:
                Tr.add(t)
                if cti > st_d:
                    st_c += cti + ctj - st_d
                    st_d = 0
                else:
                    st_c += ctj
                    st_d -= cti
        rpsi = max(self._rt_sum(Tij), ri)
        rpsj = max(self._rt_sum(Tj), rj)
        return rpsi, rpsj, Tr

    def rp2s_cal(self, ti, tj, st_i, st_j):
        rpsi, rpsj, Tr = self.calculate_RP2s(ti, tj)
        st_c = st_j + self._RT[tj.id]
        st_d = st_c - st_i - self._RT[ti.id]
        ft = max(st_i + rpsi, st_j + rpsj)
        for t in sorted(Tr, key=lambda _t: self.RP[_t.id], reverse=True):
            cti = self._CT[ti.id, t.id]
            ctj = self._CT[tj.id, t.id]
            if cti > st_d:
                st = st_c - st_d + cti + ctj
            elif ctj > 0:
                st = st_c + ctj
            else:
                st = st_j + self._RT[tj.id] - st_d + cti
            ft = max(ft, st + self.RP[t.id])
            if cti > st_d:
                st_c += cti + ctj - st_d
                st_d = 0
            else:
                st_c += ctj
                st_d -= cti
        return ft - st_i, ft - st_j

    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        all_ft = []
        cur_ft = st + self.RP[task.id]

        for t in self.edge_tasks:
            if self.is_successor(t, task):
                rpi, rpj = self.RP[t.id], self.RP[task.id]
            elif self.PL_m(t) == machine:
                rpi, rpj = self.rp2s_cal(t, task, self.ST(t), st)
            else:
                rpi, rpj = self.calculate_RP2m(t, task)
            ft_w = max(ft_w, self.ST(t) + rpi, st + rpj)
            all_ft.append(self.ST(t) + rpi)
            cur_ft = max(cur_ft, st + rpj)
        all_ft.append(cur_ft)
        return ft_w, st, sorted(all_ft, reverse=True)


class CAS8(CAS7):
    @staticmethod
    def _spt(D):
        return set(c.to_task for c in +D)

    @memo
    def calculate_RP2m(self, ti, tj):
        di, dj = copy(self.DC[ti.id]), copy(self.DC[tj.id])
        ri, rj = self.RPr[ti.id], self.RPr[tj.id]
        sti_c, stj_c = self.RA[ti.id], self.RA[tj.id]
        ati = self.RA[ti.id] - self._RT[ti.id]
        atj = self.RA[tj.id] - self._RT[tj.id]
        for t in sorted(self.TC[ti.id] & self.TC[tj.id],
                        key=lambda _t: self.RP[_t.id],
                        reverse=True):
            if t == ti or t == tj or self._RT[t.id] == 0:
                continue
            dsi = Counter(c for c in +(di - dj) if c.to_task == t)
            dsj = Counter(c for c in +(dj - di) if c.to_task == t)
            if not dsi or not dsj:
                continue
            tci, tcj = sti_c, stj_c
            # fti = ftj = max(self.RA[_t.id] for _t in t.succs())
            fti, ftj = sti_c, stj_c
            for c in sorted(set(dsi), key=lambda _c: self.RA[_c.from_task.id]):
                if self.CT(c) > 0:
                    tci = max(tci, self.RA[c.from_task.id]) + self.CT(c)
                else:
                    fti = max(fti, self.RA[c.from_task.id])
                ftj = max(ftj, self.RA[c.from_task.id])
            fti = max(fti, tci)
            for c in sorted(set(dsj), key=lambda _c: self.RA[_c.from_task.id]):
                if self.CT(c) > 0:
                    tcj = max(tcj, self.RA[c.from_task.id]) + self.CT(c)
                else:
                    ftj = max(ftj, self.RA[c.from_task.id])
                fti = max(fti, self.RA[c.from_task.id])
            ftj = max(ftj, tcj)
            if fti < ftj:  # Put on j
                # if (fti, tci) < (ftj, tcj):# Put on j
                ri = max(ri, fti + self.RP[t.id] - ati)
                rj = max(rj, fti + self.RPr[t.id] - atj)
                sti_c = tci
                di.subtract(self.DC[t.id])
                di.subtract(dsi)
            else:
                ri = max(ri, ftj + self.RPr[t.id] - ati)
                rj = max(rj, ftj + self.RP[t.id] - atj)
                stj_c = tcj
                dj.subtract(self.DC[t.id])
                dj.subtract(dsj)
        rpmi = max(self._RT[ti.id] +
                   self._rt_sum(set(c.to_task for c in +di)), ri)
        rpmj = max(self._RT[tj.id] +
                   self._rt_sum(set(c.to_task for c in +dj)), rj)
        return rpmi, rpmj


class CAS9(CAS7):
    @memo
    def calculate_RP2m(self, ti, tj):
        di, dj = copy(self.DC[ti.id]), copy(self.DC[tj.id])
        ri, rj = self.RPr[ti.id], self.RPr[tj.id]
        sti_c, stj_c = self.RA[ti.id], self.RA[tj.id]
        ati = self.RA[ti.id] - self._RT[ti.id]
        atj = self.RA[tj.id] - self._RT[tj.id]
        for t in sorted(self.TC[ti.id] & self.TC[tj.id],
                        key=lambda _t: self.RP[_t.id],
                        reverse=True):
            if t == ti or t == tj or self._RT[t.id] == 0:
                continue
            dsi = Counter(c for c in +(di - dj) if c.to_task == t)
            dsj = Counter(c for c in +(dj - di) if c.to_task == t)
            if not dsi or not dsj:
                continue
            tci, tcj = fti, ftj = sti_c, stj_c
            for c in sorted(set(dsi), key=lambda _c: self.RA[_c.from_task.id]):
                if self.CT(c) > 0:
                    tci = max(tci, self.RA[c.from_task.id]) + self.CT(c)
                else:
                    ftj = max(ftj, self.RA[c.from_task.id])
            for c in sorted(set(dsj), key=lambda _c: self.RA[_c.from_task.id]):
                if self.CT(c) > 0:
                    tcj = max(tcj, self.RA[c.from_task.id]) + self.CT(c)
                else:
                    fti = max(fti, self.RA[c.from_task.id])
            fti = max(fti, tcj)
            ftj = max(ftj, tci)
            if fti <= ftj:  # Put on i
                ri = max(ri, fti + self._RT[t.id] - ati)
                # ri = max(ri, fti + self.RPr[t.id] - ati)
                rj = max(rj, tcj + self.RP[t.id] - atj)
                stj_c = tcj
                dj.subtract(self.DC[t.id])
                dj.subtract(dsj)
            else:
                ri = max(ri, tci + self.RP[t.id] - ati)
                # rj = max(rj, ftj + self.RPr[t.id] - atj)
                rj = max(rj, ftj + self._RT[t.id] - atj)
                sti_c = tci
                di.subtract(self.DC[t.id])
                di.subtract(dsi)
            # print(">>", t, tcj, tci, ri, rj, fti, ftj, sti_c, stj_c)
        rpmi = max(self._RT[ti.id] +
                   self._rt_sum(set(c.to_task for c in +di)), ri)
        rpmj = max(self._RT[tj.id] +
                   self._rt_sum(set(c.to_task for c in +dj)), rj)
        # print(">>", ti, tj, set(c.to_task for c in +di), set(c.to_task for c in +dj), ri, rj)
        return rpmi, rpmj


class CAS10(CAS9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dis = {}

    @memo
    def _td(self, tx, ty):
        if tx == ty:
            return 0
        dm = self._dis.get((tx.id, ty.id), None)
        for t in tx.succs():
            d = self._td(t, ty)
            if d != None:
                dt = self._dis.get((tx.id, t.id), None)
                if dt != None and (dm == None or dm < d + dt):
                    dm = d + dt
        return dm

    def calculate_RP1(self, task):
        self.TC[task.id] = {task}
        self.DC[task.id] = Counter()
        self.RPr[task.id] = self._RT[task.id]
        st_t = st_c = self._RT[task.id]
        for c in sorted(task.out_comms,
                        key=lambda c: self.RP[c.to_task.id], reverse=True):
            t = c.to_task
            t_l = max(
                self._rt_sum(self.TC[task.id] | self.TC[t.id]),
                st_t + self.RPr[t.id],
                self.RPr[task.id])
            t_r = max(
                st_c + self.CT(c) + self.RP[t.id],
                self._rt_sum(self.TC[task.id]),
                self.RPr[task.id])
            if (t_l, st_t) <= (t_r, st_c + self.CT(c)):
                if self.RPr[t.id] > self._RT[t.id]:
                    self.RPr[task.id] = max(
                        self.RPr[task.id], st_t + self.RPr[t.id])
                self._dis[(task.id, t.id)] = st_t + self._RT[t.id]
                st_t += self._RT[t.id]
                self.TC[task.id].update(self.TC[t.id])
                self.DC[task.id][c] += 1
                self.DC[task.id].update(self.DC[t.id])
            else:
                self._dis[(task.id, t.id)] = st_c + self.CT(c) + self._RT[t.id]
                self.RPr[task.id] = max(
                    self.RPr[task.id], st_c + self.CT(c) + self.RP[t.id])
                st_c += self.CT(c)
        self.RP[task.id] = max(
            self._rt_sum(self.TC[task.id]),
            self.RPr[task.id])

    @memo
    def calculate_RP2m(self, ti, tj):
        di, dj = copy(self.DC[ti.id]), copy(self.DC[tj.id])
        ri, rj = self.RPr[ti.id], self.RPr[tj.id]
        ati = self.RA[ti.id] - self._RT[ti.id]
        atj = self.RA[tj.id] - self._RT[tj.id]
        sti_c = ati + self._RT[ti.id]
        stj_c = atj + self._RT[tj.id]
        for t in sorted(self.TC[ti.id] & self.TC[tj.id],
                        key=lambda _t: self.RP[_t.id],
                        reverse=True):
            if t == ti or t == tj or self._RT[t.id] == 0:
                continue
            dsi = Counter(c for c in di.elements() if c.to_task == t)
            dsj = Counter(c for c in dj.elements() if c.to_task == t)
            if (dsi & dsj):
                print(ti, tj, t, self.DC[ti.id], self.DC[tj.id], dsi, dsj)
            assert not (dsi & dsj)
            if not dsi or not dsj:
                continue
            tci, tcj = sti_c, stj_c
            fci = fti = ati + self._RT[ti.id]
            fcj = ftj = atj + self._RT[tj.id]
            # for c in sorted(set(dsi), key=lambda _c: self.RA[_c.from_task.id]):
            for c in sorted(set(dsi), key=lambda _c: -self.RP[_c.from_task.id]):
                # print("TD", ti, c.from_task, self._td(ti, c.from_task))
                if self.CT(c) > 0:
                    # tci = max(tci, self.RA[c.from_task.id]) + self.CT(c)
                    tci = max(tci, ati + self._td(ti,
                                                  c.from_task)) + self.CT(c)
                else:
                    # fci = max(fci, self.RA[c.from_task.id])
                    fci = max(fci, ati + self._td(ti, c.from_task))
                # fti = max(fti, self.RA[c.from_task.id])
                fti = max(fti, ati + self._td(ti, c.from_task))
            fci = max(fci, tci)
            # for c in sorted(set(dsj), key=lambda _c: self.RA[_c.from_task.id]):
            for c in sorted(set(dsj), key=lambda _c: -self.RP[_c.from_task.id]):
                if self.CT(c) > 0:
                    # tcj = max(tcj, self.RA[c.from_task.id]) + self.CT(c)
                    tcj = max(tcj, atj + self._td(tj,
                                                  c.from_task)) + self.CT(c)
                else:
                    # fcj = max(fcj, self.RA[c.from_task.id])
                    fcj = max(fcj, atj + self._td(tj, c.from_task))
                # ftj = max(ftj, self.RA[c.from_task.id])
                ftj = max(ftj, atj + self._td(tj, c.from_task))
            fcj = max(fcj, tcj)
            # fti = max(fti, tcj)
            # ftj = max(ftj, tci)
            # print("+++", ti, tj, t, fti, ftj, max(fti, fcj) <= max(ftj, fci))
            if max(fti, fcj) <= max(ftj, fci):  # Put on i
                ri = max(ri, fti + self.RPr[t.id] - ati)
                rj = max(rj, fcj + self.RP[t.id] - atj)
                stj_c = tcj
                # print(dj, dsj)
                dj.subtract(dsj)
                for c in dsj.elements():
                    dj.subtract(self.DC[t.id])
                # print("<", dj)
            else:
                ri = max(ri, fci + self.RP[t.id] - ati)
                rj = max(rj, ftj + self.RPr[t.id] - atj)
                sti_c = tci
                # print(di, dsi)
                di.subtract(dsi)
                for c in dsi.elements():
                    di.subtract(self.DC[t.id])
                # print("<", di)
            assert not -di
            # assert not -dj
        # if (set(c.to_task for c in +di) & set(c.to_task for c in +dj)):
            # print(+di, +dj, set(c.to_task for c in +di) & set(c.to_task for c in +dj))
        # assert not (set(c.to_task for c in +di) & set(c.to_task for c in +dj))
        rpmi = max(self._RT[ti.id] +
                   self._rt_sum(set(c.to_task for c in +di)), ri)
        rpmj = max(self._RT[tj.id] +
                   self._rt_sum(set(c.to_task for c in +dj)), rj)
        # print(">>", ti, tj, di, dj, ri, rj)
        return rpmi, rpmj


class CAS11(CAS7):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dis = {}

    @memo
    def _td(self, tx, ty):
        if tx == ty:
            return 0
        dm = self._dis.get((tx.id, ty.id), None)
        for t in tx.succs():
            d = self._td(t, ty)
            if d != None:
                dt = self._dis.get((tx.id, t.id), None)
                if dt != None and (dm == None or dm < d + dt):
                    dm = d + dt
        return dm

    def calculate_RP1(self, task):
        self.TC[task.id] = {task}
        self.DC[task.id] = Counter()
        self.RPr[task.id] = self._RT[task.id]
        st_t = st_c = self._RT[task.id]
        for c in sorted(task.out_comms,
                        key=lambda c: self.RP[c.to_task.id], reverse=True):
            t = c.to_task
            t_l = max(
                self._rt_sum(self.TC[task.id] | self.TC[t.id]),
                st_t + self.RPr[t.id],
                self.RPr[task.id])
            t_r = max(
                st_c + self.CT(c) + self.RP[t.id],
                self._rt_sum(self.TC[task.id]),
                self.RPr[task.id])
            if (t_l, st_t) <= (t_r, st_c + self.CT(c)):
                if self.RPr[t.id] > self._RT[t.id]:
                    self.RPr[task.id] = max(
                        self.RPr[task.id], st_t + self.RPr[t.id])
                self._dis[(task.id, t.id)] = st_t + self._RT[t.id]
                st_t += self._RT[t.id]
                self.TC[task.id].update(self.TC[t.id])
                self.DC[task.id][c] += 1
                self.DC[task.id].update(self.DC[t.id])
            else:
                # self._dis[(task.id, t.id)] = st_c + self.CT(c) + self._RT[t.id]
                self.RPr[task.id] = max(
                    self.RPr[task.id], st_c + self.CT(c) + self.RP[t.id])
                st_c += self.CT(c)
        self.RP[task.id] = max(
            self._rt_sum(self.TC[task.id]),
            self.RPr[task.id])

    @memo
    def calculate_RP2m(self, ti, tj):
        di, dj = copy(self.DC[ti.id]), copy(self.DC[tj.id])
        ri, rj = self.RPr[ti.id], self.RPr[tj.id]
        sti_c, stj_c = self.RA[ti.id], self.RA[tj.id]
        ati = self.RA[ti.id] - self._RT[ti.id]
        atj = self.RA[tj.id] - self._RT[tj.id]
        for t in sorted(self.TC[ti.id] & self.TC[tj.id],
                        key=lambda _t: self.RP[_t.id],
                        reverse=True):
            if t == ti or t == tj or self._RT[t.id] == 0:
                continue
            dsi = Counter(c for c in di.elements() if c.to_task == t)
            dsj = Counter(c for c in dj.elements() if c.to_task == t)
            if not dsi or not dsj:
                continue
            tci, tcj = sti_c, stj_c
            # fti, ftj = atj + self._RT[tj.id], ati + self._RT[ti.id]
            fti = max(self.RA[c.from_task.id] for c in t.in_comms
                      if self.is_successor(ti, c.from_task) or ti == c.from_task)
            ftj = max(self.RA[c.from_task.id] for c in t.in_comms
                      if self.is_successor(tj, c.from_task) or tj == c.from_task)
            for c in sorted(set(dsi), key=lambda _c: -self.RP[_c.from_task.id]):
                if self.CT(c) > 0:
                    tci = max(tci, self.RA[c.from_task.id]) + self.CT(c)
                    # tci = max(tci, ati + self._td(
                    # ti, c.from_task)) + self.CT(c)
                else:
                    ftj = max(ftj, self.RA[c.from_task.id])
                    # ftj = max(ftj, ati + self._td(ti, c.from_task))
            for c in sorted(set(dsj), key=lambda _c: -self.RP[_c.from_task.id]):
                if self.CT(c) > 0:
                    tcj = max(tcj, self.RA[c.from_task.id]) + self.CT(c)
                    # tcj = max(tcj, atj + self._td(
                    # tj, c.from_task)) + self.CT(c)
                else:
                    fti = max(fti, self.RA[c.from_task.id])
                    # fti = max(fti, atj + self._td(tj, c.from_task))
            fti = max(fti, tcj)
            ftj = max(ftj, tci)
            if fti <= ftj:  # Put on i
                # ri = max(ri, fti + self.RPr[t.id] - ati)
                rj = max(rj, fti + self.RP[t.id] - atj)
                stj_c = tcj
                dj.subtract(dsj)
                for c in dsj.elements():
                    dj.subtract(self.DC[t.id])
            else:
                ri = max(ri, ftj + self.RP[t.id] - ati)
                # rj = max(rj, ftj + self.RPr[t.id] - atj)
                sti_c = tci
                di.subtract(dsi)
                for c in dsi.elements():
                    di.subtract(self.DC[t.id])
        rpmi = max(self._RT[ti.id] +
                   self._rt_sum(set(c.to_task for c in +di)), ri)
        rpmj = max(self._RT[tj.id] +
                   self._rt_sum(set(c.to_task for c in +dj)), rj)
        return rpmi, rpmj


class CAS12(CAS7):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dis = {}

    def calculate_RP1(self, task):
        self.TC[task.id] = {task}
        self.DC[task.id] = Counter()
        self.RPr[task.id] = self._RT[task.id]
        st_t = st_c = self._RT[task.id]
        for c in sorted(task.out_comms,
                        key=lambda c: self.RP[c.to_task.id], reverse=True):
            t = c.to_task
            t_l = max(
                self._rt_sum(self.TC[task.id] | self.TC[t.id]),
                st_t + self.RPr[t.id],
                self.RPr[task.id])
            t_r = max(
                st_c + self.CT(c) + self.RP[t.id],
                self._rt_sum(self.TC[task.id]),
                self.RPr[task.id])
            if (t_l, st_t) <= (t_r, st_c + self.CT(c)):
                if self.RPr[t.id] > self._RT[t.id]:
                    self.RPr[task.id] = max(
                        self.RPr[task.id], st_t + self.RPr[t.id])
                self._dis[(task.id, t.id)] = st_t + self._RT[t.id]
                st_t += self._RT[t.id]
                self.TC[task.id].update(self.TC[t.id])
                self.DC[task.id][c] += 1
                self.DC[task.id].update(self.DC[t.id])
            else:
                self._dis[(task.id, t.id)] = st_c + self.CT(c) + self._RT[t.id]
                self.RPr[task.id] = max(
                    self.RPr[task.id], st_c + self.CT(c) + self.RP[t.id])
                st_c += self.CT(c)
        self.RP[task.id] = max(
            self._rt_sum(self.TC[task.id]),
            self.RPr[task.id])

    @memo
    def _td(self, tx, ty):
        if tx == ty:
            return 0
        dm = -inf
        for t in ty.prevs():
            if t == tx:
                dm = max(dm, self._dis[(t.id, ty.id)])
            elif self.is_successor(tx, t):
                dm = max(dm, self._td(tx, t) + self._dis[(t.id, ty.id)])
        return dm

    @memo
    def _RA2(self, t):
        return self.RA[t.id]
        # return self._td(self.entry_tasks[0], t)

    @memo
    def _td2(self, tx, ty):
        return self._RA2(ty) - self._RA2(tx) + self._RT[tx.id]

    @memo
    def calculate_RP2m(self, ti, tj):
        di, dj = copy(self.DC[ti.id]), copy(self.DC[tj.id])
        ri, rj = self.RPr[ti.id], self.RPr[tj.id]
        ati = self._RA2(ti) - self._RT[ti.id]
        atj = self._RA2(tj) - self._RT[tj.id]
        sti_co = ati + self._RT[ti.id]
        stj_co = atj + self._RT[tj.id]
        for t in sorted(self.TC[ti.id] & self.TC[tj.id],
                        key=lambda _t: self.RP[_t.id],
                        reverse=True):
            if t == ti or t == tj or self._RT[t.id] == 0:
                continue
            dsi = Counter(c for c in di.elements() if c.to_task == t)
            dsj = Counter(c for c in dj.elements() if c.to_task == t)
            if not dsi or not dsj:
                continue
            ft = 0
            tco_i, tco_j = sti_co, stj_co
            for c in sorted(t.in_comms, key=lambda _c: -self.RP[_c.from_task.id]):
                if c in dsi:
                    ft = max(ft, ati + self._td2(ti, c.from_task))
                    if self.CT(c) > 0:
                        tco_i = max(
                            tco_i, ati + self._td2(ti, c.from_task)) + self.CT(c)
                if c in dsj:
                    ft = max(ft, atj + self._td2(tj, c.from_task))
                    if self.CT(c) > 0:
                        tco_j = max(
                            tco_j, atj + self._td2(tj, c.from_task)) + self.CT(c)
            if max(ft, tco_j) <= max(ft, tco_i):  # Put on i
                ri = max(ri, max(ft, tco_j) + self.RPr[t.id] - ati)
                rj = max(rj, tco_j + self.RP[t.id] - atj)
                stj_co = tco_j
                dj.subtract(dsj)
                for c in dsj.elements():
                    dj.subtract(self.DC[t.id])
            else:
                ri = max(ri, tco_i + self.RP[t.id] - ati)
                rj = max(rj, max(ft, tco_i) + self.RPr[t.id] - atj)
                sti_co = tco_i
                di.subtract(dsi)
                for c in dsi.elements():
                    di.subtract(self.DC[t.id])
            # print(">>", t, ft, tco_j, tco_i, ri, rj, sti_co, stj_co)
        rpmi = max(self._RT[ti.id] +
                   self._rt_sum(set(c.to_task for c in +di)), ri)
        rpmj = max(self._RT[tj.id] +
                   self._rt_sum(set(c.to_task for c in +dj)), rj)
        # print(">>", ti, tj, set(c.to_task for c in +di), set(c.to_task for c in +dj), ri, rj)
        return rpmi, rpmj


class CAS13(CAS7):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dis = {}

    def calculate_RP1(self, task):
        self.TC[task.id] = {task}
        self.DC[task.id] = Counter()
        self.RPr[task.id] = self._RT[task.id]
        st_t = st_c = self._RT[task.id]
        for c in sorted(task.out_comms,
                        key=lambda c: self.RP[c.to_task.id], reverse=True):
            t = c.to_task
            t_l = max(
                self._rt_sum(self.TC[task.id] | self.TC[t.id]),
                st_t + self.RPr[t.id],
                self.RPr[task.id])
            t_r = max(
                st_c + self.CT(c) + self.RP[t.id],
                self._rt_sum(self.TC[task.id]),
                self.RPr[task.id])
            if (t_l, st_t) <= (t_r, st_c + self.CT(c)):
                if self.RPr[t.id] > self._RT[t.id]:
                    self.RPr[task.id] = max(
                        self.RPr[task.id], st_t + self.RPr[t.id])
                self._dis[(task.id, t.id)] = st_t + self._RT[t.id]
                st_t += self._RT[t.id]
                self.TC[task.id].update(self.TC[t.id])
                self.DC[task.id][c] += 1
                self.DC[task.id].update(self.DC[t.id])
            else:
                self._dis[(task.id, t.id)] = st_c + self.CT(c) + self._RT[t.id]
                self.RPr[task.id] = max(
                    self.RPr[task.id], st_c + self.CT(c) + self.RP[t.id])
                st_c += self.CT(c)
        self.RP[task.id] = max(
            self._rt_sum(self.TC[task.id]),
            self.RPr[task.id])

    @memo
    def _td(self, tx, ty):
        if tx == ty:
            return 0
        dm = -inf
        for t in ty.prevs():
            if t == tx:
                dm = max(dm, self._dis[(t.id, ty.id)])
            elif self.is_successor(tx, t):
                dm = max(dm, self._td(tx, t) + self._dis[(t.id, ty.id)])
        return dm

    @memo
    def _RA2(self, t):
        return self.RA[t.id]
        # return self._td(self.entry_tasks[0], t)

    @memo
    def _td2(self, tx, ty):
        return self._RA2(ty) - self._RA2(tx) + self._RT[tx.id]

    @memo
    def calculate_RP2m(self, ti, tj):
        di, dj = copy(self.DC[ti.id]), copy(self.DC[tj.id])
        ri, rj = self.RPr[ti.id], self.RPr[tj.id]
        ati = self._RA2(ti) - self._RT[ti.id]
        atj = self._RA2(tj) - self._RT[tj.id]
        sti_co = ati + self._RT[ti.id]
        stj_co = atj + self._RT[tj.id]
        for t in sorted(self.TC[ti.id] & self.TC[tj.id],
                        key=lambda _t: self.RP[_t.id],
                        reverse=True):
            if t == ti or t == tj or self._RT[t.id] == 0:
                continue
            dsi = Counter(c for c in di.elements() if c.to_task == t)
            dsj = Counter(c for c in dj.elements() if c.to_task == t)
            if not dsi or not dsj:
                continue
            ft = 0
            tco_i, tco_j = sti_co, stj_co
            for c in sorted(t.in_comms, key=lambda _c: -self.RP[_c.from_task.id]):
                ft = max(ft, ati + self._td2(ti, c.from_task))
                if c in dsi:
                    # ft = max(ft, ati + self._td2(ti, c.from_task))
                    if self.CT(c) > 0:
                        tco_i = max(
                            tco_i, ati + self._td2(ti, c.from_task)) + self.CT(c)
                if c in dsj:
                    # ft = max(ft, atj + self._td2(tj, c.from_task))
                    if self.CT(c) > 0:
                        tco_j = max(
                            tco_j, atj + self._td2(tj, c.from_task)) + self.CT(c)
            if max(ft, tco_j) <= max(ft, tco_i):  # Put on i
                # ri = max(ri, max(ft, tco_j) + self.RPr[t.id] - ati)
                rj = max(rj, tco_j + self.RP[t.id] - atj)
                stj_co = tco_j
                dj.subtract(dsj)
                for c in dsj.elements():
                    dj.subtract(self.DC[t.id])
            else:
                ri = max(ri, tco_i + self.RP[t.id] - ati)
                # rj = max(rj, max(ft, tco_i) + self.RPr[t.id] - atj)
                sti_co = tco_i
                di.subtract(dsi)
                for c in dsi.elements():
                    di.subtract(self.DC[t.id])
            # print(">>", t, ft, tco_j, tco_i, ri, rj, sti_co, stj_co)
        rpmi = max(self._RT[ti.id] +
                   self._rt_sum(set(c.to_task for c in +di)), ri)
        rpmj = max(self._RT[tj.id] +
                   self._rt_sum(set(c.to_task for c in +dj)), rj)
        # print(">>", ti, tj, set(c.to_task for c in +di), set(c.to_task for c in +dj), ri, rj)
        return rpmi, rpmj


class CAS14(CAS13):
    @memo
    def calculate_RP2m(self, ti, tj, sti, stj):
        di, dj = copy(self.DC[ti.id]), copy(self.DC[tj.id])
        ri, rj = self.RPr[ti.id], self.RPr[tj.id]
        ati, atj = sti, stj
        # ati = self._RA2(ti) - self._RT[ti.id]
        # atj = self._RA2(tj) - self._RT[tj.id]
        sti_co = ati + self._RT[ti.id]
        stj_co = atj + self._RT[tj.id]
        for t in sorted(self.TC[ti.id] & self.TC[tj.id],
                        key=lambda _t: self.RP[_t.id],
                        reverse=True):
            if t == ti or t == tj or self._RT[t.id] == 0:
                continue
            dsi = Counter(c for c in di.elements() if c.to_task == t)
            dsj = Counter(c for c in dj.elements() if c.to_task == t)
            if not dsi or not dsj:
                continue
            ft = 0
            tco_i, tco_j = sti_co, stj_co
            for c in sorted(t.in_comms, key=lambda _c: -self.RP[_c.from_task.id]):
                ft = max(ft, ati + self._td2(ti, c.from_task))
                if c in dsi:
                    if self.CT(c) > 0:
                        tco_i = max(
                            tco_i, ati + self._td2(ti, c.from_task)) + self.CT(c)
                if c in dsj:
                    if self.CT(c) > 0:
                        tco_j = max(
                            tco_j, atj + self._td2(tj, c.from_task)) + self.CT(c)
            if max(ft, tco_j) <= max(ft, tco_i):  # Put on i
                stj_co = tco_j
                dj.subtract(dsj)
                for c in dsj.elements():
                    dj.subtract(self.DC[t.id])
            else:
                sti_co = tco_i
                di.subtract(dsi)
                for c in dsi.elements():
                    di.subtract(self.DC[t.id])
        return self.TC[ti.id] - set(c.to_task for c in +dj) - {ti}, self.TC[tj.id] - set(c.to_task for c in +di) - {tj}

    def rp2m_cal(self, ti, tj, sti, stj):
        Ti, Tj = self.calculate_RP2m(ti, tj, sti, stj)
        stt_i = stc_i = sti + self._RT[ti.id]
        stt_j = stc_j = stj + self._RT[tj.id]
        ri = sti + self.RPr[ti.id]
        rj = stj + self.RPr[tj.id]
        fts = [None] * self.problem.num_tasks
        for t in sorted(Ti | Tj, key=lambda _t: self.RP[_t.id], reverse=True):
            if t in Ti:
                cs = sorted([c for c in t.in_comms if c.from_task in Tj],
                            key=lambda _c: self.RP[_c.from_task.id], reverse=True)
                st = stt_i
                for c in cs:
                    if self.CT(c) > 0:
                        stc_j = max(stc_j, fts[c.from_task.id]) + self.CT(c)
                        st = max(st, stc_j)
                    else:
                        st = max(st, fts[c.from_task.id])
                fts[t.id] = stt_i = st + self._RT[t.id]
                ri = max(ri, st + self.RPr[t.id])
            elif t in Tj:
                cs = sorted([c for c in t.in_comms if c.from_task in Ti],
                            key=lambda _c: self.RP[_c.from_task.id], reverse=True)
                st = stt_j
                for c in cs:
                    if self.CT(c) > 0:
                        stc_i = max(stc_i, fts[c.from_task.id]) + self.CT(c)
                        st = max(st, stc_i)
                    else:
                        st = max(st, fts[c.from_task.id])
                fts[t.id] = stt_j = st + self._RT[t.id]
                rj = max(rj, st + self.RPr[t.id])
        rpi = max(stt_i, ri) - sti
        rpj = max(stt_j, rj) - stj
        return rpi, rpj

    def fitness(self, task, machine, comm_pls, st):
        return st, 0, []
        ft_w = st + self.RP[task.id]
        all_ft = []
        cur_ft = st + self.RP[task.id]

        for t in self.edge_tasks:
            if self.is_successor(t, task):
                rpi, rpj = self.RP[t.id], self.RP[task.id]
            elif self.PL_m(t) == machine:
                rpi, rpj = self.rp2s_cal(t, task, self.ST(t), st)
            else:
                # rpi, rpj = self.calculate_RP2m(t, task)
                rpi, rpj = self.rp2m_cal(t, task, self.ST(t), st)
                # print(t, task, self.ST(t) + rpi, st + rpj)
            ft_w = max(ft_w, self.ST(t) + rpi, st + rpj)
            all_ft.append(self.ST(t) + rpi)
            cur_ft = max(cur_ft, st + rpj)
        all_ft.append(cur_ft)
        return ft_w, st, sorted(all_ft, reverse=True)


class CAS15(CAS9):
    @memo
    def calculate_RP2m(self, ti, tj):
        di, dj = copy(self.DC[ti.id]), copy(self.DC[tj.id])
        ri, rj = self.RPr[ti.id], self.RPr[tj.id]
        fts = [None] * self.problem.num_tasks
        ati = self.RA[ti.id] - self._RT[ti.id]
        atj = self.RA[tj.id] - self._RT[tj.id]
        stc_i = stt_i = fts[ti.id] = ati + self._RT[ti.id]
        stc_j = stt_j = fts[tj.id] = atj + self._RT[tj.id]
        for t in sorted(self.TC[ti.id] | self.TC[tj.id],
                        key=lambda _t: self.RP[_t.id],
                        reverse=True):
            if t == ti or t == tj or self._RT[t.id] == 0:
                continue
            dsi = Counter(c for c in di.elements() if c.to_task == t)
            dsj = Counter(c for c in dj.elements() if c.to_task == t)
            if not dsi:
                rj = max(rj, stt_j + self.RPr[t.id])
                stt_j += self._RT[t.id]
                fts[t.id] = stt_j
                continue
            if not dsj:
                ri = max(ri, stt_i + self.RPr[t.id])
                stt_i += self._RT[t.id]
                fts[t.id] = stt_i
                continue
            tc_i, tc_j = stc_i, stc_j
            ft_i, ft_j = stt_i, stt_j
            for c in sorted(set(dsj), key=lambda _c: fts[_c.from_task.id]):
                if self.CT(c) > 0:
                    tc_j = max(tc_j, fts[c.from_task.id]) + self.CT(c)
                    ft_i = max(ft_i, tc_j)
                else:
                    ft_i = max(ft_i, fts[c.from_task.id])
            for c in sorted(set(dsi), key=lambda _c: fts[_c.from_task.id]):
                if self.CT(c) > 0:
                    tc_i = max(tc_i, fts[c.from_task.id]) + self.CT(c)
                    ft_j = max(ft_j, tc_i)
                else:
                    ft_j = max(ft_j, fts[c.from_task.id])
            if ft_i <= ft_j:
                fts[t.id] = stt_i = ft_i + self._RT[t.id]
                stc_j = tc_j
                ri = ft_i + self.RPr[t.id]
                rj = stc_j + self.RP[t.id]
                dj.subtract(dsj)
                for c in dsj.elements():
                    dj.subtract(self.DC[t.id])
            else:
                fts[t.id] = stt_j = ft_j + self._RT[t.id]
                stc_i = tc_i
                ri = stc_i + self.RP[t.id]
                rj = ft_j + self.RPr[t.id]
                di.subtract(dsi)
                for c in dsi.elements():
                    di.subtract(self.DC[t.id])
        return max(stt_i, ri) - ati, max(stt_j, rj) - atj

    def sort_tasks(self):
        toporder = self._topsort()
        for t in reversed(toporder):
            self.calculate_RP1(t)
        self.ready_tasks = set(
            t for t in self.problem.tasks if t.in_degree == 0)
        self.edge_tasks = set()
        ids = [t.in_degree for t in self.problem.tasks]
        ods = [t.out_degree for t in self.problem.tasks]
        while self.ready_tasks:
            task = max(self.ready_tasks, key=lambda t: self.RP[t.id])
            self.calculate_RA(task)
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
            self.RA[task.id] = self.FT(task)

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

        Tm = set(self.PL_m(c.from_task)
                 for c in comms if self._pld[c.from_task.id])
        for m in Tm:
            st_t, st_c = 0, 0
            for c in comms:
                t = c.from_task
                if self._pld[t.id] and self.PL_m(t) == m or self.CT(c) == 0:
                    st_t = max(st_t, self.RA[t.id])
                else:
                    st_c = max(st_c, self.RA[t.id]) + self.CT(c)
            st = min(st, max(st_t, st_c))

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


class CAS_n1(CAS9):
    @memo
    def calculate_RP2s(self, ti, tj, st):
        if st > self.ST(ti):
            ri = max(self._rt_sum(self.TC[ti.id] | self.TC[tj.id]),
                     self.RPr[ti.id])
            rj = max(self._rt_sum(self.TC[tj.id]),
                     self.RPr[tj.id])
        else:
            ri = max(self._rt_sum(self.TC[ti.id]),
                     self.RPr[ti.id])
            rj = max(self._rt_sum(self.TC[ti.id] | self.TC[tj.id]),
                     self.RPr[tj.id])
        return ri, rj

    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        all_ft = []
        cur_ft = st + self.RP[task.id]

        for t in self.edge_tasks:
            if self.is_successor(t, task):
                rpi, rpj = self.RP[t.id], self.RP[task.id]
                # print("<1>", t, self.ST(t) + rpi, st + rpj, rpi, rpj)
            elif self.PL_m(t) == machine:
                rpi, rpj = self.calculate_RP2s(t, task, st)
                # print("<2>", t, self.ST(t) + rpi, st + rpj, rpi, rpj)
            else:
                rpi, rpj = self.calculate_RP2m(t, task)
                # print("<3>", t, self.ST(t) + rpi, st + rpj, rpi, rpj)
            ft_w = max(ft_w, self.ST(t) + rpi, st + rpj)
            all_ft.append(self.ST(t) + rpi)
            cur_ft = max(cur_ft, st + rpj)
        all_ft.append(cur_ft)
        return ft_w, st, sorted(all_ft, reverse=True)


class CAS_n2(CAS_n1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RA = [None] * self.problem.num_tasks
        self.RP = [None] * self.problem.num_tasks
        self.RPr = [None] * self.problem.num_tasks
        self.TC = [None] * self.problem.num_tasks
        self.TCM = [None] * self.problem.num_tasks
        self._pld = [False] * self.problem.num_tasks
        self.ready_tasks = None
        self.edge_tasks = None

    def calculate_RP1(self, task):
        self.TC[task.id] = {task}
        self.TCM[task.id] = Counter()
        self.RPr[task.id] = self._RT[task.id]
        st_t = st_c = self._RT[task.id]
        for c in sorted(task.out_comms,
                        key=lambda c: self.RP[c.to_task.id], reverse=True):
            t = c.to_task
            t_l = max(
                self._rt_sum(self.TC[task.id] | self.TC[t.id]),
                st_t + self.RPr[t.id],
                self.RPr[task.id])
            t_r = max(
                st_c + self.CT(c) + self.RP[t.id],
                self._rt_sum(self.TC[task.id]),
                self.RPr[task.id])
            if (t_l, st_t) <= (t_r, st_c + self.CT(c)):
                if self.RPr[t.id] > self._RT[t.id]:
                    self.RPr[task.id] = max(
                        self.RPr[task.id], st_t + self.RPr[t.id])
                st_t += self._RT[t.id]
                self.TC[task.id].update(self.TC[t.id])
                self.TCM[task.id][t] += 1
                self.TCM[task.id].update(self.TCM[t.id])
            else:
                self.RPr[task.id] = max(
                    self.RPr[task.id], st_c + self.CT(c) + self.RP[t.id])
                st_c += self.CT(c)
        self.RP[task.id] = max(
            self._rt_sum(self.TC[task.id]),
            self.RPr[task.id])

    def est2(self, ti, t, tmi=None, rt=None):
        rt = rt or self.RA[ti.id]
        tmi = tmi or self.TC[ti.id]
        lt = self.RA[ti.id]
        for c in sorted(t.in_comms, key=lambda _c: self.RA[_c.from_task.id]):
            if c.from_task in tmi or c.from_task == ti:
                if self.CT(c) > 0:
                    rt = max(rt, self.RA[c.from_task.id]) + self.CT(c)
                lt = max(lt, self.RA[c.from_task.id])
        return lt, rt

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ri = self.RPr[ti.id] - self._RT[ti.id]
        rj = self.RPr[tj.id] - self._RT[tj.id]
        rai, raj = self.RA[ti.id], self.RA[tj.id]
        cti, ctj = rai, raj
        T_shrd = set(self.TCM[ti.id] & self.TCM[tj.id])
        for t in sorted(T_shrd, key=lambda _t: -self.RP[_t.id]):
            if t not in +tmi or t not in +tmj:
                continue
            lti, rti = self.est2(ti, t, +tmi, cti)
            ltj, rtj = self.est2(tj, t, +tmj, ctj)
            sti, stj = max(lti, rtj), max(ltj, rti)
            if sti <= stj:
                # if rtj <= rti:
                ri = max(ri, lti + self.RPr[t.id] - rai)
                rj = max(rj, rtj + self.RP[t.id] - raj)
                ctj = max(ctj, rtj)
                for _ in range(tmj[t]):
                    tmj.subtract(self.TCM[t.id])
                del tmj[t]
            else:
                ri = max(ri, rti + self.RP[t.id] - rai)
                rj = max(rj, ltj + self.RPr[t.id] - raj)
                cti = max(cti, rti)
                for _ in range(tmi[t]):
                    tmi.subtract(self.TCM[t.id])
                del tmi[t]
        rpmi = self._RT[ti.id] + max(self._rt_sum(set(+tmi)), ri)
        rpmj = self._RT[tj.id] + max(self._rt_sum(set(+tmj)), rj)
        return rpmi, rpmj


class CAS_n3(CAS_n1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RA = [None] * self.problem.num_tasks
        self.RP = [None] * self.problem.num_tasks
        self.RPr = [None] * self.problem.num_tasks
        self.TC = [None] * self.problem.num_tasks
        self.TCM = [None] * self.problem.num_tasks
        self.TDS = {}
        self.TDSR = {}
        self._pld = [False] * self.problem.num_tasks
        self.ready_tasks = None
        self.edge_tasks = None

    def calculate_RP1(self, task):
        self.TC[task.id] = {task}
        self.TCM[task.id] = Counter()
        self.RPr[task.id] = self._RT[task.id]
        self.TDS[(task.id, task.id)] = 0
        self.TDSR[(task.id, task.id)] = 0
        st_t = st_c = self._RT[task.id]
        for c in sorted(task.out_comms,
                        key=lambda c: self.RP[c.to_task.id], reverse=True):
            t = c.to_task
            t_l = max(
                self._rt_sum(self.TC[task.id] | self.TC[t.id]),
                st_t + self.RPr[t.id],
                self.RPr[task.id])
            t_r = max(
                st_c + self.CT(c) + self.RP[t.id],
                self._rt_sum(self.TC[task.id]),
                self.RPr[task.id])
            self.TDSR[(task.id, t.id)] = st_c - self._RT[task.id]
            if (t_l, st_t) <= (t_r, st_c + self.CT(c)):
                if self.RPr[t.id] > self._RT[t.id]:
                    self.RPr[task.id] = max(
                        self.RPr[task.id], st_t + self.RPr[t.id])
                st_t += self._RT[t.id]
                self.TC[task.id].update(self.TC[t.id])
                self.TCM[task.id][t] += 1
                self.TCM[task.id].update(self.TCM[t.id])
                self.TDS[(task.id, t.id)] = st_t - self._RT[task.id]
            else:
                self.RPr[task.id] = max(
                    self.RPr[task.id], st_c + self.CT(c) + self.RP[t.id])
                st_c += self.CT(c)
                self.TDS[(task.id, t.id)] = st_c + \
                    self._RT[t.id] - self._RT[task.id]
        self.RP[task.id] = max(
            self._rt_sum(self.TC[task.id]),
            self.RPr[task.id])

    def _td(self, ti, tj):
        if (ti.id, tj.id) not in self.TDS:
            mts = [t for t in tj.prevs() if self.is_successor(ti, t) or ti == t]
            self.TDS[(ti.id, tj.id)] = max(
                self._td(ti, t) + self._td(t, tj) for t in mts)
        return self.TDS[(ti.id, tj.id)]

    def _ra(self, task):
        # return self.RA[task.id]
        return self._td(self.entry_tasks[0], task)

    def est2(self, ti, t, rai, tmi=None, rt=None):
        rt = rt or rai
        lt = rai
        tmi = tmi or self.TC[ti.id]
        ts = [c for c in t.in_comms if c.from_task in tmi or c.from_task == ti]
        for c in sorted(ts, key=lambda _c: self._td(ti, _c.from_task)):
            if self.CT(c) > 0:
                rt = max(rt,
                         rai + self._td(ti, c.from_task) + self.TDSR[(c.from_task.id, t.id)]) + self.CT(c)
                # self._ra(c.from_task) + self.TDSR[(c.from_task.id, t.id)]) + self.CT(c)
            lt = max(lt,
                     rai + self._td(ti, c.from_task) + self.TDS[(c.from_task.id, t.id)] - self._RT[t.id])
            # self._ra(c.from_task) + self.TDS[(c.from_task.id, t.id)] - self._RT[t.id])
        return lt, rt

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ri = self.RPr[ti.id] - self._RT[ti.id]
        rj = self.RPr[tj.id] - self._RT[tj.id]
        rai, raj = self._ra(ti), self._ra(tj)
        cti, ctj = rai, raj
        T_shrd = set(self.TCM[ti.id] & self.TCM[tj.id])
        for t in sorted(T_shrd, key=lambda _t: -self.RP[_t.id]):
            if t not in +tmi or t not in +tmj:
                continue
            lti, rti = self.est2(ti, t, rai, +tmi, cti)
            ltj, rtj = self.est2(tj, t, raj, +tmj, ctj)
            # lti, rti = self.est2(ti, t, None, None)
            # ltj, rtj = self.est2(tj, t, None, None)
            # sti, stj = max(lti, rtj), max(ltj, rti)
            # if sti <= stj:
            if rtj <= rti:
                # ri = max(ri, sti + self._RT[t.id] - rai)
                rj = max(rj, rtj + self.RP[t.id] - raj)
                ctj = max(ctj, rtj)
                for _ in range(tmj[t]):
                    tmj.subtract(self.TCM[t.id])
                del tmj[t]
            else:
                ri = max(ri, rti + self.RP[t.id] - rai)
                # rj = max(rj, stj + self._RT[t.id] - raj)
                cti = max(cti, rti)
                for _ in range(tmi[t]):
                    tmi.subtract(self.TCM[t.id])
                del tmi[t]
        rpmi = self._RT[ti.id] + max(self._rt_sum(set(+tmi)), ri)
        rpmj = self._RT[tj.id] + max(self._rt_sum(set(+tmj)), rj)
        return rpmi, rpmj


class CAS_n4(CAS_n3):
    def _td(self, ti, tj):
        if (ti.id, tj.id) not in self.TDS:
            mts = [t for t in ti.succs() if self.is_successor(t, tj) or t == tj]
            self.TDS[(ti.id, tj.id)] = max(
                self.TDS[(ti.id, t.id)] + self._td(t, tj) for t in mts)
        return self.TDS[(ti.id, tj.id)]

    def _ra(self, task):
        return self._td(self.entry_tasks[0], task)

    # @memo
    def est2(self, ti, t, cti, tmi=None):
        ot = 0
        mts = [_t for _t in t.prevs() if self.is_successor(ti, _t) or ti == _t]
        cts = []
        tmi = tmi or self.TC[ti.id]
        for _t in mts:
            if (_t in tmi or _t == ti) and self._CT[_t.id, t.id] > 0:
                cts.append(
                    (self._td(ti, _t) + self.TDSR[(_t.id, t.id)], self._CT[_t.id, t.id]))
            else:
                ot = max(ot, self._td(ti, _t) +
                         self._td(_t, t) - self._RT[t.id])

        rt = cti
        for st, ct in sorted(cts, key=lambda x: x[0]):
            rt = max(rt, st) + ct
        return max(rt, ot), rt

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ri = self.RPr[ti.id] - self._RT[ti.id]
        rj = self.RPr[tj.id] - self._RT[tj.id]
        rai, raj = self._ra(ti), self._ra(tj)
        T_shrd = set(self.TCM[ti.id] & self.TCM[tj.id])
        cti, ctj = 0, 0
        for t in sorted(T_shrd, key=lambda _t: -self.RP[_t.id]):
            if t not in +tmi or t not in +tmj:
                continue
            rti, ci = self.est2(ti, t, cti, +tmi)
            rtj, cj = self.est2(tj, t, ctj, +tmj)
            if raj + rtj <= rai + rti:
                rj = max(rj, rtj + self.RP[t.id])
                ctj = max(ctj, cj)
                for _ in range(tmj[t]):
                    tmj.subtract(self.TCM[t.id])
                del tmj[t]
            else:
                ri = max(ri, rti + self.RP[t.id])
                cti = max(cti, ci)
                for _ in range(tmi[t]):
                    tmi.subtract(self.TCM[t.id])
                del tmi[t]
        rpmi = self._RT[ti.id] + max(self._rt_sum(set(+tmi)), ri)
        rpmj = self._RT[tj.id] + max(self._rt_sum(set(+tmj)), rj)
        return rpmi, rpmj


class CAS_n5(CAS_n4):
    def est2(self, ti, t, rai, tmi, cti):
        mts = [_t for _t in t.prevs() if _t in tmi or _t == ti]
        # mts = [_t for _t in t.prevs() if _t in tmi]
        # for _t in sorted(mts, key=lambda _t: self._ra(_t)):
        # cti = max(cti, self._ra(_t)) + self._CT[_t.id, t.id]
        for _t in sorted(mts, key=lambda _t: self._td(ti, _t)):
            cti = max(cti, rai + self._td(ti, _t)) + self._CT[_t.id, t.id]
        # for _t in sorted(mts, key=lambda _t: self._td(ti, _t) + self.TDSR[_t.id, t.id]):
            # cti = max(cti, rai + self._td(ti, _t) + self.TDSR[_t.id, t.id]) + self._CT[_t.id, t.id]
        return cti

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ri = self.RPr[ti.id] - self._RT[ti.id]
        rj = self.RPr[tj.id] - self._RT[tj.id]
        rai, raj = self._ra(ti), self._ra(tj)
        # rai, raj = sti + self._RT[ti.id], stj + self._RT[tj.id]
        T_shrd = set(self.TCM[ti.id] & self.TCM[tj.id])
        cti, ctj = rai, raj
        for t in sorted(T_shrd, key=lambda _t: -self.RP[_t.id]):
            if t not in +tmi or t not in +tmj:
                continue
            rti = self.est2(ti, t, rai, +tmi, cti)
            rtj = self.est2(tj, t, raj, +tmj, ctj)
            if rtj <= rti:
                ri = max(ri, rti + self._RT[t.id] - rai)
                rj = max(rj, rtj + self.RP[t.id] - raj)
                ctj = max(ctj, rtj)
                for _ in range(tmj[t]):
                    tmj.subtract(self.TCM[t.id])
                del tmj[t]
            else:
                ri = max(ri, rti + self.RP[t.id] - rai)
                rj = max(rj, rti + self._RT[t.id] - raj)
                cti = max(cti, rti)
                for _ in range(tmi[t]):
                    tmi.subtract(self.TCM[t.id])
                del tmi[t]
        rpmi = self._RT[ti.id] + max(self._rt_sum(set(+tmi)), ri)
        rpmj = self._RT[tj.id] + max(self._rt_sum(set(+tmj)), rj)
        return rpmi, rpmj


class CAS_n6(CAS_n4):
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

    def est2(self, ti, t, rai, tmi, cti):
        mts = [_t for _t in t.prevs() if _t in tmi or _t == ti]
        # mts = [_t for _t in t.prevs() if _t in tmi]
        for _t in sorted(mts, key=lambda _t: self._td(ti, _t)):
            cti = max(cti, rai + self._td(ti, _t)) + self._CT[_t.id, t.id]
        return cti

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ri = self.RPr[ti.id] - self._RT[ti.id]
        rj = self.RPr[tj.id] - self._RT[tj.id]
        rai, raj = self.FT(ti), self.ft_avg
        # rai, raj = sti + self._RT[ti.id], stj + self._RT[tj.id]
        T_shrd = set(self.TCM[ti.id] & self.TCM[tj.id])
        cti, ctj = rai, raj
        for t in sorted(T_shrd, key=lambda _t: -self.RP[_t.id]):
            if t not in +tmi or t not in +tmj:
                continue
            rti = self.est2(ti, t, rai, +tmi, cti)
            rtj = self.est2(tj, t, raj, +tmj, ctj)
            if rtj <= rti:
                ri = max(ri, rtj + self.RPr[t.id] - rai)
                rj = max(rj, rtj + self.RP[t.id] - raj)
                ctj = max(ctj, rtj)
                for _ in range(tmj[t]):
                    tmj.subtract(self.TCM[t.id])
                del tmj[t]
            else:
                ri = max(ri, rti + self.RP[t.id] - rai)
                rj = max(rj, rti + self.RPr[t.id] - raj)
                cti = max(cti, rti)
                for _ in range(tmi[t]):
                    tmi.subtract(self.TCM[t.id])
                del tmi[t]
        rpmi = self._RT[ti.id] + max(self._rt_sum(set(+tmi)), ri)
        rpmj = self._RT[tj.id] + max(self._rt_sum(set(+tmj)), rj)
        return rpmi, rpmj


class CAS_n7(CAS_n4):
    def _mlcl(self, sts):
        l = 0
        s = set()
        for st, ts in sorted(sts, key=lambda x: -x[0]):
            s |= ts
            l = max(l, st + self._rt_sum(s))
        return l

    def fitness(self, task, machine, comm_pls, st):
        ft_w = st + self.RP[task.id]
        ms = {}
        for t in self.edge_tasks:
            m = self.PL_m(t)
            if m not in ms:
                ms[m] = []
            ms[m].append(t)
        for m, ts in ms.items():
            if machine == m:
                sts = [(st, self.TC[task.id])]
                for t in ts:
                    sts.append((self.ST(t), self.TC[t.id]))
                l = self._mlcl(sts)
                l = max(l, st + self.RPr[task.id])
                for t in ts:
                    l = max(l, self.ST(t) + self.RPr[t.id])
            else:
                l = self.ft_on_dm(ts, task, st)
            ft_w = max(ft_w, l)
        return ft_w, st

    def est2(self, t, tmi, cti, ts):
        mts = [_t for _t in t.prevs() if _t in tmi or _t in ts]
        for _t in sorted(mts, key=lambda _t: self._ra(_t)):
            cti = max(cti, self._ra(_t)) + self._CT[_t.id, t.id]
        return cti

    def ft_on_dm(self, ts, task, st):
        if not ts:
            return st + self.RP[task.id]
        tmi = copy(self.TCM[task.id])
        tmj = Counter()
        for t in ts:
            tmj.update(self.TCM[t.id])
        T_shrd = set(tmi & tmj)
        cti = st + self._RT[task.id]
        ctj = min(self.FT(t) for t in ts)
        r = st + self.RPr[task.id]
        for t in ts:
            r = max(r, self.ST(t) + self.RPr[task.id])
        for t in sorted(T_shrd, key=lambda _t: -self.RP[_t.id]):
            if t == task:
                ctj = max(ctj, st)
                for _ in range(tmj[t]):
                    tmj.subtract(self.TCM[t.id])
                del tmj[t]
                continue
            if t not in +tmi or t not in +tmj:
                continue
            rti = self.est2(t, +tmi, cti, {task})
            rtj = self.est2(t, +tmj, ctj, set(ts))
            if rtj <= rti:
                r = max(r, rtj + self.RP[t.id])
                ctj = max(ctj, rtj)
                for _ in range(tmj[t]):
                    tmj.subtract(self.TCM[t.id])
                del tmj[t]
            else:
                r = max(r, rti + self.RP[t.id])
                cti = max(cti, rti)
                for _ in range(tmi[t]):
                    tmi.subtract(self.TCM[t.id])
                del tmi[t]
        sts = {t: (self.ST(t), {t}) for t in ts}
        for t in set(+tmj):
            for _t in sts:
                if self.is_successor(_t, t):
                    sts[_t][1].add(t)
        l = self._mlcl(sts.values())
        return max(r, l, st + self._RT[task.id] + self._rt_sum(set(+tmi)))


class CAS_n8(CAS_n4):
    # def _ra(self, task):
        # return self.RA[task.id]

    def est2(self, ti, tj, t, rai, raj, tmi, tmj, ctj):
        ri = 0
        for _t in sorted(t.prevs(), key=self._ra):
            if not (_t in tmi or _t == ti):
                ri = max(ri, self._ra(_t)) + self._CT[_t.id, t.id]
                if _t in tmj or tj == _t:
                    ctj = max(ctj, raj + self._td(tj, _t)) + \
                        self._CT[_t.id, t.id]
            else:
                ri = max(ri, self._ra(_t))
        return max(ri, ctj), ctj

    @memo
    def calculate_RP2m(self, ti, tj):
        tmi, tmj = copy(self.TCM[ti.id]), copy(self.TCM[tj.id])
        ri = self.RPr[ti.id] - self._RT[ti.id]
        rj = self.RPr[tj.id] - self._RT[tj.id]
        rai, raj = self._ra(ti), self._ra(tj)
        # rai, raj = sti + self._RT[ti.id], stj + self._RT[tj.id]
        T_shrd = set(self.TCM[ti.id] & self.TCM[tj.id])
        cti, ctj = rai, raj
        for t in sorted(T_shrd, key=lambda _t: -self.RP[_t.id]):
            if t not in +tmi or t not in +tmj:
                continue
            rti, cj = self.est2(ti, tj, t, rai, raj, +tmi, +tmj, ctj)
            rtj, ci = self.est2(tj, ti, t, raj, rai, +tmj, +tmi, cti)
            if rti <= rtj:
                # ri = max(ri, rti + self.RPr[t.id] - rai)
                rj = max(rj, cj + self.RP[t.id] - raj)
                ctj = max(ctj, cj)
                for _ in range(tmj[t]):
                    tmj.subtract(self.TCM[t.id])
                del tmj[t]
            else:
                ri = max(ri, ci + self.RP[t.id] - rai)
                # rj = max(rj, rtj + self.RPr[t.id] - raj)
                cti = max(cti, ci)
                for _ in range(tmi[t]):
                    tmi.subtract(self.TCM[t.id])
                del tmi[t]
        rpmi = self._RT[ti.id] + max(self._rt_sum(set(+tmi)), ri)
        rpmj = self._RT[tj.id] + max(self._rt_sum(set(+tmj)), rj)
        return rpmi, rpmj


class CAS_nr(CAS_n1):
    def sort_tasks(self):
        toporder = self._topsort()
        self._ranks = [0] * self.problem.num_tasks
        for task in reversed(toporder):
            self._ranks[task.id] = max([self._ranks[c.to_task.id] + self.CT(c)
                                        for c in task.out_comms],
                                       default=0) + (self._RT[task.id] or 0.1)
        for t in toporder:
            self.calculate_RA(t)
        for t in reversed(toporder):
            self.calculate_RP1(t)
        self.ready_tasks = set(
            t for t in self.problem.tasks if t.in_degree == 0)
        self.edge_tasks = set()
        ids = [t.in_degree for t in self.problem.tasks]
        ods = [t.out_degree for t in self.problem.tasks]
        while self.ready_tasks:
            task = max(self.ready_tasks, key=lambda t: self._ranks[t.id])
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

    def calculate_RP1(self, task):
        self.TC[task.id] = {task}
        self.DC[task.id] = Counter()
        self.RPr[task.id] = self._RT[task.id]
        st_t = st_c = self._RT[task.id]
        for c in sorted(task.out_comms,
                        key=lambda c: self._ranks[c.to_task.id], reverse=True):
            t = c.to_task
            t_l = max(
                self._rt_sum(self.TC[task.id] | self.TC[t.id]),
                st_t + self.RPr[t.id],
                self.RPr[task.id])
            t_r = max(
                st_c + self.CT(c) + self.RP[t.id],
                self._rt_sum(self.TC[task.id]),
                self.RPr[task.id])
            if (t_l, st_t) <= (t_r, st_c + self.CT(c)):
                if self.RPr[t.id] > self._RT[t.id]:
                    self.RPr[task.id] = max(
                        self.RPr[task.id], st_t + self.RPr[t.id])
                st_t += self._RT[t.id]
                self.TC[task.id].update(self.TC[t.id])
                self.DC[task.id][c] += 1
                self.DC[task.id].update(self.DC[t.id])
            else:
                self.RPr[task.id] = max(
                    self.RPr[task.id], st_c + self.CT(c) + self.RP[t.id])
                st_c += self.CT(c)
        self.RP[task.id] = max(
            self._rt_sum(self.TC[task.id]),
            self.RPr[task.id])


class CAS_n9(CAS_n1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CRS = {}

    def sorted_in_comms(self, task):
        # return task.in_comms
        return sorted(task.in_comms, key=self.CRS.get, reverse=True)

    def calculate_RP1(self, task):
        self.TC[task.id] = {task}
        self.DC[task.id] = Counter()
        self.RPr[task.id] = self._RT[task.id]
        st_t = st_c = self._RT[task.id]
        cs = 0
        for c in sorted(task.out_comms,
                        key=lambda c: self.RP[c.to_task.id], reverse=True):
            t = c.to_task
            t_l = max(
                self._rt_sum(self.TC[task.id] | self.TC[t.id]),
                st_t + self.RPr[t.id],
                self.RPr[task.id])
            self.CRS[c] = cs
            cs += self.CT(c)
            t_r = max(
                st_c + self.CT(c) + self.RP[t.id],
                self._rt_sum(self.TC[task.id]),
                self.RPr[task.id])
            if (t_l, st_t) <= (t_r, st_c + self.CT(c)):
                if self.RPr[t.id] > self._RT[t.id]:
                    self.RPr[task.id] = max(
                        self.RPr[task.id], st_t + self.RPr[t.id])
                st_t += self._RT[t.id]
                self.TC[task.id].update(self.TC[t.id])
                self.DC[task.id][c] += 1
                self.DC[task.id].update(self.DC[t.id])
            else:
                self.RPr[task.id] = max(
                    self.RPr[task.id], st_c + self.CT(c) + self.RP[t.id])
                st_c += self.CT(c)
        for c in task.out_comms:
            self.CRS[c] = cs - self.CRS[c]
        self.RP[task.id] = max(
            self._rt_sum(self.TC[task.id]),
            self.RPr[task.id])


class CAS_n10(CAS_n4):
    def sort_tasks(self):
        toporder = self._topsort()
        for t in toporder:
            self.calculate_RA(t)
        for t in reversed(toporder):
            self.calculate_RP1(t)
        self.ready_tasks = set(
            t for t in self.problem.tasks if t.in_degree == 0)
        self.edge_tasks = set()
        ids = [t.in_degree for t in self.problem.tasks]
        ods = [t.out_degree for t in self.problem.tasks]
        while self.ready_tasks:
            task = max(self.ready_tasks, key=lambda t: self.RP[t.id])
            yield task
            self._pld[task.id] = True
            self.ready_tasks.remove(task)
            for t in task.succs():
                ids[t.id] -= 1
                if not ids[t.id]:
                    self.ready_tasks.add(t)
            for t in task.prevs():
                ods[t.id] -= 1
                if not ods[t.id]:
                    self.edge_tasks.remove(t)
            for t in self.edge_tasks:
                if task in self.TC[t.id] and self.PL_m(task) != self.PL_m(t):
                    for _ in range(self.TCM[t.id][task]):
                        self.TCM[t.id].subtract(self.TCM[task.id])
                    del self.TCM[t.id][task]
                    self.TC[t.id] = set(+self.TCM[t.id]) | {t}
                elif self.PL_m(task) == self.PL_m(t) and self.is_successor(t, task) and task not in self.TC[t.id]:
                    self.TCM[t.id][task] += 1
                    self.TCM[t.id].update(self.TCM[task.id])
                    self.TC[t.id] = set(+self.TCM[t.id]) | {t}
            if ods[task.id]:
                self.edge_tasks.add(task)
