from collections import Counter
from copy import copy
from math import inf
from .ca3 import *
import networkx as nx


class MoreContention(Heuristic):
    def has_contention(self, tx, ty):
        return True


class MDComparison(Heuristic):
    def select_task(self):
        self.update_AT_and_PT()
        self._dcs = [0] * self.problem.num_tasks
        self._ws = [0] * self.problem.num_tasks
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and\
                        not (self.rids[tx] and self.rids[ty]) and\
                        self.has_contention(tx, ty):
                    ftx, _, wy = self.est_ft(tx, ty, True)
                    fty, _, wx = self.est_ft(ty, tx, True)
                    self._ws[tx] = max(self._ws[tx], wx)
                    # self._flc[_t] += 1
                    # self._flc[t.id] += i
                    self._ws[ty] = max(self._ws[ty], wx)
                    # print(self.problem.tasks[tx],
                    # self.problem.tasks[ty], ftx, fty)
                    if ftx < fty:
                        self._dcs[ty] -= 1
                    elif ftx > fty:
                        self._dcs[tx] -= 1
        # print([(t, self._dcs[t.id], self._fls[t.id], self.RP[t.id]) for t in self.ready_tasks])
        task = max(self.ready_tasks, key=lambda t: (
            self._dcs[t.id], self._ws[t.id], self.RP[t.id]))
        # print("Selected", task)
        return task


class RSort(Heuristic):
    def select_task(self):
        self.update_AT_and_PT()
        return max(self.ready_tasks, key=lambda t: self.RP[t.id] + self.ATCS[t.id])


class Sort2(Heuristic):
    def select_task(self):
        self.update_AT_and_PT()
        self._dcs = [0] * self.problem.num_tasks
        self._bdcs = [0] * self.problem.num_tasks
        self._mft_m = [0] * self.problem.num_tasks
        self._mft_s = [0] * self.problem.num_tasks
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and\
                        not (self.rids[tx] and self.rids[ty]) and\
                        self.has_contention(tx, ty):
                    ftx = self.est_ft(tx, ty)
                    fty = self.est_ft(ty, tx)
                    # print(self.problem.tasks[tx],
                    # self.problem.tasks[ty], ftx, fty, self.AT[tx], self.AT[ty])
                    if ftx < fty:
                        self._dcs[ty] += 1
                        self._bdcs[tx] += 1
                    elif ftx > fty:
                        self._dcs[tx] += 1
                        self._bdcs[ty] += 1
        # print([(t, self._dcs[t.id], self._bdcs[t.id], self.RP[t.id])
               # for t in self.ready_tasks])
        task = max(self.ready_tasks, key=lambda t: (
            -self._dcs[t.id], self._bdcs[t.id], self.RP[t.id]))
        # print("Selected", task)
        return task


class Sort3(Heuristic):
    def select_task(self):
        self.update_AT_and_PT()
        self._dcs = [0] * self.problem.num_tasks
        self._bdcs = [0] * self.problem.num_tasks
        self._mft_m = [0] * self.problem.num_tasks
        self._mft_s = [0] * self.problem.num_tasks
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and\
                        not self.rids[tx] and not self.rids[ty]:
                    ftx = self.est_ft(tx, ty)
                    fty = self.est_ft(ty, tx)
                    # print(self.problem.tasks[tx],
                    # self.problem.tasks[ty], ftx, fty, self.AT[tx], self.AT[ty])
                    if ftx < fty:
                        self._dcs[ty] += 1
                        self._bdcs[tx] += 1
                    elif ftx > fty:
                        self._dcs[tx] += 1
                        self._bdcs[ty] += 1
        # print([(t, self._dcs[t.id], self._bdcs[t.id], self.RP[t.id])
               # for t in self.ready_tasks])
        task = max(self.ready_tasks, key=lambda t: (
            -self._dcs[t.id], self._bdcs[t.id], self.RP[t.id]))
        # print("Selected", task)
        return task


class Sort4(Heuristic):
    def select_task(self):
        self.update_AT_and_PT()
        self._dcs = [0] * self.problem.num_tasks
        self._bdcs = [0] * self.problem.num_tasks
        self._fl = [[] for _ in self.problem.tasks]
        self._flc = [0] * self.problem.num_tasks
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and\
                        not self.rids[tx] and not self.rids[ty]:
                    ftx = self.est_ft(tx, ty)
                    fty, mf = self.est_ft(ty, tx, True)
                    # print(self.problem.tasks[tx],
                    # self.problem.tasks[ty], ftx, fty, self.AT[tx], self.AT[ty], mf)
                    if ftx < fty:
                        self._dcs[ty] += 1
                        self._bdcs[tx] += 1
                    elif ftx > fty:
                        self._dcs[tx] += 1
                        self._bdcs[ty] += 1
                    else:
                        # if (self.BM[tx] & self.BM[ty]) and self.AT[tx] != 0:
                        if (self.BM[tx] & self.BM[ty]) and mf == 0:
                            if self.RP[ty] > self.RP[tx]:
                                self._fl[tx].append(ty)
                            if self.RP[tx] > self.RP[ty]:
                                self._fl[ty].append(tx)
        ts2 = set()
        max_vs = -inf, -inf
        for t in self.ready_tasks:
            vs = -self._dcs[t.id], self._bdcs[t.id]
            if vs > max_vs:
                max_vs = vs
                ts2 = {t}
            elif vs == max_vs:
                ts2.add(t)

        # print([(t, self._dcs[t.id], self._bdcs[t.id], [self.problem.tasks[_t] for _t in self._fl[t.id]]) for t in ts2])
        for t in ts2:
            if len(self._fl[t.id]) < 2:
                continue
            self._fl[t.id].sort(key=lambda _t: self.RP[_t])
            ct = self.PTO[t.id] - self.PT[t.id]
            x = 0
            for i, _t in enumerate(self._fl[t.id]):
                if self.problem.tasks[_t] not in ts2:
                    continue
                x += self._RT[_t] + self.PT_l[_t]
                if ct != 0 and x >= ct:
                    for j in self._fl[:i + 1]:
                        self._dcs[_t] += 1
                        self._bdcs[t.id] += 1
                    break
        # print([(t, self._dcs[t.id], self._bdcs[t.id], self._flc[t.id], self.RP[t.id])
               # for t in self.ready_tasks])
        task = max(ts2, key=lambda t: (
            -self._dcs[t.id], self._bdcs[t.id], self.RP[t.id]))
        # print("Selected", task)
        return task


class Sort5(Heuristic):
    @memo
    def rrank(self, task):
        comms = sorted(task.communications(COMM_OUTPUT),
                       key=lambda c: -self.rrank(c.to_task))
        cst = self._RT[task.id]
        ft = self._RT[task.id]
        for c in comms:
            cst += self.CT(c)
            ft = max(ft, cst + self.rrank(c.to_task))
        return ft

    def select_task(self):
        self.update_AT_and_PT()
        self._dcs = [0] * self.problem.num_tasks
        self._bdcs = [0] * self.problem.num_tasks
        self._mft_m = [0] * self.problem.num_tasks
        self._mft_s = [0] * self.problem.num_tasks
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and\
                        not self.rids[tx] and not self.rids[ty]:
                    ftx = self.est_ft(tx, ty)
                    fty = self.est_ft(ty, tx)
                    # print(self.problem.tasks[tx],
                    # self.problem.tasks[ty], ftx, fty, self.AT[tx], self.AT[ty])
                    if ftx < fty:
                        self._dcs[ty] += 1
                        self._bdcs[tx] += 1
                    elif ftx > fty:
                        self._dcs[tx] += 1
                        self._bdcs[ty] += 1
        # print([(t, self._dcs[t.id], self._bdcs[t.id], self.RP[t.id])
               # for t in self.ready_tasks])
        task = max(self.ready_tasks, key=lambda t: (
            -self._dcs[t.id], self._bdcs[t.id], self.rrank(t)))
        # print("Selected", task)
        return task


class Sort6(Heuristic):
    def select_task(self):
        self.update_AT_and_PT()
        self._dcs = [0] * self.problem.num_tasks
        self._bdcs = [0] * self.problem.num_tasks
        self.w = [self.RP[t] for t in range(self.problem.num_tasks)]
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and\
                        not self.rids[tx] and not self.rids[ty]:
                    ftx, _, fty_0 = self.est_ft(tx, ty, ret_ftx=True)
                    fty, _, ftx_0 = self.est_ft(ty, tx, ret_ftx=True)
                    # print(self.problem.tasks[tx],
                    # self.problem.tasks[ty], ftx, fty, self.AT[tx], self.AT[ty])
                    if ftx < fty:
                        self._dcs[ty] += 1
                        self._bdcs[tx] += 1
                    elif ftx > fty:
                        self._dcs[tx] += 1
                        self._bdcs[ty] += 1
                    self.w[tx] += ftx_0 - self.RA[tx] - self.PT[tx]
                    self.w[ty] += fty_0 - self.RA[ty] - self.PT[ty]
        # print([(t, self._dcs[t.id], self._bdcs[t.id], self.RP[t.id])
               # for t in self.ready_tasks])
        task = max(self.ready_tasks, key=lambda t: (
            -self._dcs[t.id], self._bdcs[t.id], self.w[t.id]))
        # print("Selected", task)
        return task


class Sort7(Heuristic):

    def ready_graph(self):
        self._edges.sort(key=lambda i: i[-1], reverse=True)
        rg = nx.DiGraph()
        for tx, ty, w in self._edges:
            rg.add_edge(tx, ty)
            try:
                nx.find_cycle(rg)
                rg.remove_edge(tx, ty)
            except nx.exception.NetworkXNoCycle:
                pass
        for t in self.ready_tasks:
            if t.id not in rg:
                rg.add_node(t.id)
        for t, in_d in rg.in_degree(rg):
            if in_d == 0:
                yield self.problem.tasks[t]

    def select_task(self):
        self.update_AT_and_PT()
        self._edges = []
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and\
                        not self.rids[tx] and not self.rids[ty]:
                    ftx = self.est_ft(tx, ty)
                    fty = self.est_ft(ty, tx)
                    if ftx < fty:
                        self._edges.append((tx, ty, fty))
                    elif ftx > fty:
                        self._edges.append((ty, tx, ftx))
        task = max(self.ready_graph(), key=lambda t: self.RP[t.id])
        # print("Selected", task)
        return task


class ESTFTv1(Heuristic):
    def est_ft(self, ti, tj, ret_fts=False):
        p0, p1 = self.BM[ti], self.BM[tj]
        cti, ctj = self.ATCS[ti], self.ATCS[tj]
        ati, atj = self.AT[ti], self.AT[tj]
        if cti > 0 and ctj > 0 and -cti < atj - ati < ctj:
            st_j = min(ati + ctj, atj + cti)
        else:
            st_j = atj
        ft0, ft0i, ft0j = self.ef_XYAB(ti, tj, ati, st_j)
        # print(">>", self.problem.tasks[ti], self.problem.tasks[tj], ati, atj, st_j, cti, ctj, ft0i, ft0j)
        if (len(p0) == 1 and len(p1) == 1) and (p0 & p1):
            ft1, ft1i, ft1j = self.ef_XA_YB(ti, tj, self.AT[ti], self.ATO[tj])
        else:
            ft1, ft1i, ft1j = self.ef_XA_YB(ti, tj, self.AT[ti], self.AT[tj])
        # print("EST_FT", self.problem.tasks[ti], self.problem.tasks[tj], ft0, ft1, self.AT[ti], self.AT[tj], p0, p1)
        if ret_fts:
            return min(ft0, ft1), max(ft0i, ft1i), max(ft0j, ft1j)
        else:
            return min(ft0, ft1)


class ESTFTv2(Heuristic):
    def rsts(self, cti, ctj, ati, atj):
        if cti > 0 and ctj > 0 and -cti < atj - ati < ctj:
            atj = min(ati + ctj, atj + cti)
        return ati, atj

    def est_ft(self, ti, tj, ret_typ=False, ret_ftx=False):
        # print(">>", self.problem.tasks[ti], self.problem.tasks[tj])
        p0, p1 = self.BM[ti], self.BM[tj]
        cti, ctj = self.ATCS[ti], self.ATCS[tj]
        sti, stj = self.rsts(cti, ctj, self.ATO[ti], self.ATO[tj])
        ft_m, ftx, fty = self.ef_XYAB(ti, tj, sti, stj)
        typ = 0
        # print(">1", ft_m)
        if p0 & p1:
            sti, stj = self.rsts(cti, ctj, self.AT[ti], self.AT[tj])
            ft_1, ftx_1, fty_1 = self.ef_XYAB(ti, tj, sti, stj)
            if ft_1 < ft_m:
                typ = 0
                ft_m, ftx, fty = ft_1, ftx_1, fty_1
            # print(">2", ft_m, sti, stj, self.AT[ti], self.AT[tj], cti, ctj)
        if any(m not in p1 for m in p0):
            sti, stj = self.rsts(cti, ctj, self.AT[ti], self.ATO[tj])
            ft_1, ftx_1, fty_1 = self.ef_XYAB(ti, tj, sti, stj)
            if ft_1 < ft_m:
                typ = 0
                ft_m, ftx, fty = ft_1, ftx_1, fty_1
            # print(">3", ft_m)
        if any(m not in p0 for m in p1):
            sti, stj = self.rsts(cti, ctj, self.ATO[ti], self.AT[tj])
            ft_1, ftx_1, fty_1 = self.ef_XYAB(ti, tj, sti, stj)
            if ft_1 < ft_m:
                typ = 0
                ft_m, ftx, fty = ft_1, ftx_1, fty_1
            # print(">4", ft_m)
        if not (len(p0) == 1 and len(p1) == 1 and (p0 & p1)):
            ft_1, ftx_1, fty_1 = self.ef_XA_YB(
                ti, tj, self.AT[ti], self.AT[tj])
            if ft_1 < ft_m:
                typ = 1
                ft_m, ftx, fty = ft_1, ftx_1, fty_1
            # print(">5", ft_m, p0, p1)
        ft_1, ftx_1, fty_1 = self.ef_XA_YB(ti, tj, self.AT[ti], self.ATO[tj])
        if ft_1 < ft_m:
            typ = 1
            ft_m, ftx, fty = ft_1, ftx_1, fty_1
        # print(">6", ft_m, self.AT[ti], self.ATO[tj], p0, p1)
        # print("<<")
        if ret_typ:
            return ft_m, typ
        elif ret_ftx:
            return ft_m, ftx, fty
        else:
            return ft_m


class ESTFTv3(ESTFTv2):

    def est_ft(self, ti, tj):
        # print(">>", self.problem.tasks[ti], self.problem.tasks[tj])
        p0, p1 = self.BM[ti], self.BM[tj]
        cti, ctj = self.ATCS[ti], self.ATCS[tj]
        sti, stj = self.rsts(cti, ctj, self.ATO[ti], self.ATO[tj])
        ft_m = self.ef_XYAB(ti, tj, sti, stj)[0]
        # print(">1", ft_m)
        if p0 & p1:
            sti, stj = self.rsts(cti, ctj, self.AT[ti], self.AT[tj])
            ft_m = min(ft_m, self.ef_XYAB(ti, tj, sti, stj)[0])
            # print(">2", ft_m, sti, stj, self.AT[ti], self.AT[tj], cti, ctj)
        if any(m not in p1 for m in p0):
            sti, stj = self.rsts(cti, ctj, self.AT[ti], self.ATO[tj])
            ft_m = min(ft_m, self.ef_XYAB(ti, tj, sti, stj)[0])
            # print(">3", ft_m)
        if any(m not in p0 for m in p1):
            sti, stj = self.rsts(cti, ctj, self.ATO[ti], self.AT[tj])
            ft_m = min(ft_m, self.ef_XYAB(ti, tj, sti, stj)[0])
            # print(">4", ft_m)
        if not (len(p0) == 1 and len(p1) == 1 and (p0 & p1)):
            ft_m = min(ft_m,
                       self.ef_XA_YB(ti, tj, self.AT[ti], self.AT[tj])[0])
            # print(">5", ft_m, p0, p1)
        else:
            ft_m = min(ft_m, self.ef_XA_YB(
                ti, tj, self.AT[ti], self.ATO[tj])[0])
        # print(">6", ft_m, self.AT[ti], self.ATO[tj], p0, p1)
        # print("<<")
        return ft_m


class ESTFTv4(ESTFTv2):

    def est_ft(self, ti, tj):
        p0, p1 = self.BM[ti], self.BM[tj]
        cti, ctj = self.ATCS[ti], self.ATCS[tj]
        ft_m = inf
        if p0 & p1:
            sti, stj = self.rsts(cti, ctj, self.AT[ti], self.AT[tj])
            ft_m = min(ft_m, self.ef_XYAB(ti, tj, sti, stj)[0])
        else:
            sti, stj = self.rsts(cti, ctj, self.AT[ti], self.ATO[tj])
            ft_m = min(ft_m, self.ef_XYAB(ti, tj, sti, stj)[0])
        if (p0 & p1) and len(p1) == 1:
            ft_m = min(ft_m, self.ef_XA_YB(
                ti, tj, self.AT[ti], self.ATO[tj])[0])
        else:
            ft_m = min(ft_m,
                       self.ef_XA_YB(ti, tj, self.AT[ti], self.AT[tj])[0])
        return ft_m


class CAN8(CAN6_2_2):
    def sort_succs_2(self, ts):
        w = {c: 0 for c in ts.keys()}
        for t0, (ct_i0, ct_j0) in ts.items():
            for t1, (ct_i1, ct_j1) in ts.items():
                if t0 == t1:
                    continue
                x, y = ct_i0 + ct_j0, ct_i1 + ct_j1
                A, B = self.RP[t0.id], self.RP[t1.id]
                if min(A, y) + B < A + x:
                    w[t1] += 1
                if A + min(B, x) < B + y:
                    w[t0] += 1
        return sorted(ts.items(),
                      key=lambda t: (w[t[0]], -sum(t[1]) - self.RP[t[0].id]))

    def update_cst_hlc(self, hlc, cst, ct_i, ct_j):
        if hlc > 0:
            if ct_i >= hlc:
                cst += ct_i + ct_j - hlc
                hlc = 0
            else:
                cst += ct_j
                hlc -= ct_i
        else:
            cst += ct_i + ct_j
        return hlc, cst

    @memo
    def ef_XYAB(self, ti, tj, st_i, st_j):
        ts = {}
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i = st_i + self._RT[ti]
        ft_j = max(ft_i, st_j) + self._RT[tj]
        l_i, l_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        ctasks = set()
        m_i = self.PL_m(self.problem.tasks[ti]) if self._placed[ti] else None
        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            if self._placed[tx.id]:
                if self.PL_m(tx) == m_i:
                    for t in self._ctasks[tx.id] | {tx}:
                        if t in ctasks:
                            continue
                        elif self._placed[t.id] and self.PL_m(t) == m_i:
                            tst = max(tst, self.FT(t))
                        else:
                            lft += self._RT[t.id]
                    ctasks.add(tx)
                    ctasks.update(self._ctasks[tx.id])
                    tst = max(tst, self.FT(tx))
                    l_i = max(l_i, lft, self.FT(tx) + self.PT_r[tx.id])
                else:
                    l_i = max(l_i, self.FT(tx) + self.PT[tx.id])
                    hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
            else:
                d = sum(self._RT[t.id]
                        for t in (self._ctasks[tx.id] | {tx}) - ctasks)
                fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
                if hlc > 0:
                    if ct_i >= hlc:
                        fr = cst + ct_i + ct_j - hlc + self.RP[tx.id]
                    else:
                        if ct_j > 0:
                            fr = cst + ct_j + self.RP[tx.id]
                        else:
                            fr = cst - hlc + ct_i + self.RP[tx.id]
                else:
                    fr = cst + ct_i + ct_j + self.RP[tx.id]
                if tx in ctasks or (fl, tst) <= (fr, fr - self.RP[tx.id]):
                    if ct_i:
                        l_i = max(l_i, fl)
                    if ct_j:
                        l_j = max(l_j, fl)
                    tst += self._RT[tx.id]
                    lft += d
                    ctasks.add(tx)
                    ctasks.update(self._ctasks[tx.id])
                else:
                    if ct_i:
                        l_i = max(l_i, fr)
                    if ct_j:
                        l_j = max(l_j, fr)
                    hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
        # print(self.problem.tasks[ti], self.problem.tasks[tj], st_i, st_j, l_i, l_j)
        return max(l_i, l_j), l_i, l_j


class CAN8_1(CAN8):
    def est_ft(self, ti, tj, ret_fts=False):
        ft0, ft0i, ft0j = self.ef_XYAB(ti, tj, self.AT[ti], self.AT[tj])
        p0 = self.BM[ti]
        p1 = self.BM[tj]
        if (len(p0) == 1 and len(p1) == 1) and (p0 & p1):
            ft1, ft1i, ft1j = self.ef_XA_YB(ti, tj, self.AT[ti], self.ATO[tj])
        else:
            ft1, ft1i, ft1j = self.ef_XA_YB(ti, tj, self.AT[ti], self.AT[tj])
        # print("EST_FT", ft0, ft1, self.AT[ti], self.AT[tj], self.ATO[tj])
        # print(self.problem.tasks[ti], self.problem.tasks[tj], ft0, ft1)
        if ret_fts:
            return min(ft0, ft1), max(ft0i, ft1i), max(ft0j, ft1j)
        else:
            return min(ft0, ft1)

    def fitness(self, task, machine, comm_pls, st):
        all_ft = []
        task_ft = st + self._RT[task.id] + self.PT[task.id]
        wrk_ft = task_ft
        cur_ft = task_ft

        for t in self.problem.tasks:
            if self._placed[t.id] and \
                    any(not self._placed[c.to_task.id]
                        for c in t.communications(COMM_OUTPUT)
                        if c.to_task != task):
                # print("MEET", t, task)
                st_t = self.ST(t)
                if self.PL_m(t) != machine:
                    ft_a, ft_i, ft_j = self.ef_XA_YB(t.id, task.id, st_t, st)
                    # print("CSEA 1", t, ft_a, ft_i, ft_j, st_t, st)
                elif self.is_successor(t, task):
                    ft_j = st + self._RT[task.id] + self.PT[task.id]
                    ft_a = ft_i = max(ft_j, self.FT(t) + self.PT[t.id])
                    # print("CASE 2", self.PT[task.id], st, self.PT[t.id], self.PT_r[t.id], self.PT_l[t.id], self._ctasks[t.id])
                else:
                    ft_a, ft_i, ft_j = self.ef_XYAB(t.id, task.id, st_t, st)
                    # print("CASE 3", self.PT[task.id], st, self.PT[t.id], self.PT_r[t.id], self.PT_l[t.id], self._ctasks[t.id], ft_a)
                all_ft.append(ft_i)
                cur_ft = max(cur_ft, ft_j)
                wrk_ft = max(wrk_ft, ft_a)
                # print(t, ft_a, ft_i, ft_j)
        all_ft.append(cur_ft)
        return wrk_ft, task_ft, sorted(all_ft, reverse=True)


class CAN8_2(CAN8_1):
    def filter_ctasks(self, tx, tj_id, ctasks_x):
        s = set()
        if tx not in ctasks_x:
            s.add(tx)
        for t in self._ctasks[tx.id]:
            if t not in ctasks_x and t.id != tj_id:
                s.add(t)
        return s

    @memo
    def ef_XA_YB(self, ti, tj, st_i, st_j):
        ts = {}
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] = self.CT(c)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] = self.CT(c)

        cst_i = tst_i = l_i = lft_i = st_i + self._RT[ti]
        cst_j = tst_j = l_j = lft_j = st_j + self._RT[tj]
        ctasks_i, ctasks_j = set(), set()
        m_i = self.PL_m(self.problem.tasks[ti]) if self._placed[ti] else None
        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            if self._placed[tx.id]:
                if self.PL_m(tx) == m_i:
                    # lft_i += sum(self._RT[t.id] for t in (
                        # (self._ctasks[tx.id] | {tx}) - ctasks_i - {self.problem.tasks[tj]} - self._ctasks[tj]))
                    lft_i += sum(self._RT[t.id]
                                 for t in self.filter_ctasks(tx, tj, ctasks_i))
                    ctasks_i.add(tx)
                    ctasks_i.update(self._ctasks[tx.id])
                    tst_i = max(tst_i, self.FT(tx))
                    l_i = max(l_i, lft_i, self.FT(tx) + self.PT_r[tx.id])
                else:
                    cst_i += ct_i
                    l_i = max(l_i, self.FT(tx) + self.PT[tx.id])
            elif tx.id == tj:
                cst_i += ct_i
            else:
                in_i = tx in self.problem.tasks[ti].succs()
                in_j = tx in self.problem.tasks[tj].succs()
                # print(in_i, in_j)
                if in_i and not in_j:
                    d = sum(self._RT[t.id]
                            for t in self.filter_ctasks(tx, tj, ctasks_i))
                    # d = sum(self._RT[t.id] for t in (
                    # (self._ctasks[tx.id] | {tx}) - ctasks_i))
                    fl = max(lft_i + d, tst_i +
                             self._RT[tx.id] + self.PT_r[tx.id])
                    fr = cst_i + ct_i + self.RP[tx.id]
                    # print(d, lft_i, tst_i, cst_i, fl, fr)
                    if (fl, tst_i) <= (fr, fr - self.RP[tx.id]):
                        l_i = max(l_i, fl)
                        tst_i += self._RT[tx.id]
                        lft_i += d
                        ctasks_i.add(tx)
                        ctasks_i.update(self._ctasks[tx.id])
                    else:
                        l_i = max(l_i, fr)
                        cst_i += ct_i
                elif not in_i and in_j:
                    d = sum(self._RT[t.id]
                            for t in self.filter_ctasks(tx, ti, ctasks_j))
                    # d = sum(self._RT[t.id] for t in (
                    # (self._ctasks[tx.id] | {tx}) - ctasks_j))
                    fl = max(lft_j + d, tst_j +
                             self._RT[tx.id] + self.PT_r[tx.id])
                    fr = cst_j + ct_j + self.RP[tx.id]
                    if (fl, tst_j) <= (fr, fr - self.RP[tx.id]):
                        l_j = max(l_j, fl)
                        tst_j += self._RT[tx.id]
                        lft_j += d
                        ctasks_j.add(tx)
                        ctasks_j.update(self._ctasks[tx.id])
                    else:
                        l_j = max(l_j, fr)
                        cst_j += ct_i
                elif in_i and in_j:
                    st_on_i = max(tst_i, cst_j + ct_j)
                    st_on_j = max(tst_j, cst_i + ct_i)
                    st_on_n = max(cst_i + ct_i, cst_j + ct_j)
                    st = min(st_on_i, st_on_j, st_on_n)
                    # print(">>>>", tx, st_on_i, st_on_j, st_on_n, tst_i, tst_j, cst_i, cst_j)
                    if st == st_on_i:
                        cd = st - tst_i
                        tst_i = st + self._RT[tx.id]
                        cst_j += ct_j
                        lft_i += sum(self._RT[t.id]
                                     for t in self.filter_ctasks(tx, tj, ctasks_i)) + cd
                        # lft_i += sum(self._RT[t.id] for t in (
                        # (self._ctasks[tx.id] | {tx}) - ctasks_i)) + cd
                        ctasks_i.add(tx)
                        ctasks_i.update(self._ctasks[tx.id])
                        fl = max(lft_i, tst_i + self.PT_r[tx.id])
                        l_i = max(l_i, fl)
                        l_j = max(l_j, fl)
                        # print(l_i, l_j)
                    elif st == st_on_j:
                        cd = st - tst_j
                        tst_j = st + self._RT[tx.id]
                        cst_i += ct_i
                        lft_j += sum(self._RT[t.id]
                                     for t in self.filter_ctasks(tx, ti, ctasks_j)) + cd
                        # print("lft_i", lft_i, self.filter_ctasks(
                        # tx, ti, ctasks_j), st, tst_j)
                        # lft_j += sum(self._RT[t.id] for t in (
                        # (self._ctasks[tx.id] | {tx}) - ctasks_j)) + cd
                        ctasks_j.add(tx)
                        ctasks_j.update(self._ctasks[tx.id])
                        fl = max(lft_j, tst_j + self.PT_r[tx.id])
                        l_i = max(l_i, fl)
                        l_j = max(l_j, fl)
                    else:
                        cst_i += ct_i
                        cst_j += ct_j
                        fr = st + self.RP[tx.id]
                        l_i = max(l_i, fr)
                        l_j = max(l_j, fr)
            # print(tx, l_i, l_j)
        # print(self.problem.tasks[ti], self.problem.tasks[tj],
              # st_i, st_j, self.PT[ti], self.PT[tj], l_i, l_j)
        return max(l_i, l_j), l_i, l_j


class CAN8_3(CAN8_1):
    @memo
    def merge_gain(self, ti, tj):
        ts = {}
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i = self.RA[ti]
        ft_j = max(ft_i, self.AT[tj]) + self._RT[tj]
        l_i, l_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        ctasks = set()
        m_i = self.PL_m(self.problem.tasks[ti]) if self._placed[ti] else None
        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            if self._placed[tx.id]:
                if self.PL_m(tx) == m_i:
                    for t in self._ctasks[tx.id] | {tx}:
                        if t in ctasks:
                            continue
                        elif self._placed[t.id] and self.PL_m(t) == m_i:
                            tst = max(tst, self.FT(t))
                        else:
                            lft += self._RT[t.id]
                    ctasks.add(tx)
                    ctasks.update(self._ctasks[tx.id])
                    tst = max(tst, self.FT(tx))
                    l_i = max(l_i, lft, self.FT(tx) + self.PT_r[tx.id])
                else:
                    l_i = max(l_i, self.FT(tx) + self.PT[tx.id])
                    hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
            else:
                d = sum(self._RT[t.id]
                        for t in (self._ctasks[tx.id] | {tx}) - ctasks)
                fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
                if hlc > 0:
                    if ct_i >= hlc:
                        fr = cst + ct_i + ct_j - hlc + self.RP[tx.id]
                    else:
                        if ct_j > 0:
                            fr = cst + ct_j + self.RP[tx.id]
                        else:
                            fr = cst - hlc + ct_i + self.RP[tx.id]
                else:
                    fr = cst + ct_i + ct_j + self.RP[tx.id]
                if tx in ctasks or (fl, tst) <= (fr, fr - self.RP[tx.id]):
                    if ct_i:
                        l_i = max(l_i, fl)
                    if ct_j:
                        l_j = max(l_j, fl)
                    tst += self._RT[tx.id]
                    lft += d
                    ctasks.add(tx)
                    ctasks.update(self._ctasks[tx.id])
                else:
                    if ct_i:
                        l_i = max(l_i, fr)
                    if ct_j:
                        l_j = max(l_j, fr)
                    hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
        return l_i - ft_i - self.PT[ti], l_j - ft_j - self.PT[tj]

    def ef_XYAB(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = max(ft_i, st_j) + self._RT[tj]
        di, dj = self.merge_gain(ti, tj)
        ft_ai = ft_i + self.PT[ti] + di
        ft_aj = ft_j + self.PT[tj] + dj
        # print(self.problem.tasks[ti], self.problem.tasks[tj], st_i, st_j, ft_ai, ft_aj, di, dj)
        return max(ft_ai, ft_aj), ft_ai, ft_aj


class CAN8_4(CAN8_1):
    @memo
    def merge_gain(self, ti, tj):
        ts = {}
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i = self.RA[ti]
        ft_j = max(ft_i, self.AT[tj]) + self._RT[tj]
        l_i, l_j = ft_i, ft_j
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            d = sum(self._RT[t.id]
                    for t in (self._ctasks[tx.id] | {tx}) - ctasks)
            fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
            if hlc > 0:
                if ct_i >= hlc:
                    fr = cst + ct_i + ct_j - hlc + self.RP[tx.id]
                else:
                    if ct_j > 0:
                        fr = cst + ct_j + self.RP[tx.id]
                    else:
                        fr = cst - hlc + ct_i + self.RP[tx.id]
            else:
                fr = cst + ct_i + ct_j + self.RP[tx.id]
            # if tx in ctasks or (fl, tst) <= (fr, fr - self.RP[tx.id]):
            if (fl, tst) <= (fr, fr - self.RP[tx.id]):
                tst += self._RT[tx.id]
                lft += d
                ctasks.add(tx)
                ctasks.update(self._ctasks[tx.id])
                if ct_i:
                    l_i = max(l_i, lft)
                    r_i = max(r_i, tst + self.PT_r[tx.id])
                    ctasks_i.add(tx)
                    ctasks_i.update(self._ctasks[tx.id])
                if ct_j:
                    l_j = max(l_j, lft)
                    r_j = max(r_j, tst + self.PT_r[tx.id])
                    ctasks_j.add(tx)
                    ctasks_j.update(self._ctasks[tx.id])
            else:
                if ct_i:
                    r_i = max(r_i, fr)
                if ct_j:
                    r_j = max(r_j, fr)
                hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
        ld_i = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        ld_j = sum(self._RT[t.id] for t in ctasks_j)
        return ft_j - ft_i, l_i - ft_i, r_i - ft_i, l_j - ft_j, r_j - ft_j, ld_i, ld_j

    def ef_XYAB(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = max(ft_i, st_j) + self._RT[tj]
        d, dl_i, dr_i, dl_j, dr_j, ld_i, ld_j = self.merge_gain(ti, tj)
        d2 = ft_j - ft_i - d
        if ld_i == 0:
            if d < dl_i <= d2 + d:
                ft_ai = ft_i + dl_i - self._RT[tj]
            else:
                ft_ai = ft_i + dl_i
        else:
            ft_ai = ft_i + max(d + d2 + ld_i, dl_i)
        ft_ai = max(ft_ai, ft_i + dr_i)
        ft_aj = ft_j + max(dl_j - d2, ld_j, dr_j)
        # print(self.problem.tasks[ti], self.problem.tasks[tj], ft_i, ft_j, self.merge_gain(ti, tj))
        return max(ft_ai, ft_aj), ft_ai, ft_aj


class CAN8_5(CAN8_1):
    @memo
    def merge_gain(self, ti, tj):
        ts = {}
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i = self.RA[ti]
        ft_j = max(ft_i, self.AT[tj]) + self._RT[tj]
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            d = sum(self._RT[t.id]
                    for t in (self._ctasks[tx.id] | {tx}) - ctasks)
            fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
            if hlc > 0:
                if ct_i >= hlc:
                    fr = cst + ct_i + ct_j - hlc + self.RP[tx.id]
                else:
                    if ct_j > 0:
                        fr = cst + ct_j + self.RP[tx.id]
                    else:
                        fr = cst - hlc + ct_i + self.RP[tx.id]
            else:
                fr = cst + ct_i + ct_j + self.RP[tx.id]
            # if tx in ctasks or (fl, tst) <= (fr, fr - self.RP[tx.id]):
            if (fl, tst) <= (fr, fr - self.RP[tx.id]):
                tst += self._RT[tx.id]
                lft += d
                ctasks.add(tx)
                ctasks.update(self._ctasks[tx.id])
                if ct_i:
                    if self.PT_r[tx.id] > 0:
                        r_i = max(r_i, tst + self.PT_r[tx.id])
                    ctasks_i.add(tx)
                    ctasks_i.update(self._ctasks[tx.id])
                if ct_j:
                    if self.PT_r[tx.id] > 0:
                        r_j = max(r_j, tst + self.PT_r[tx.id])
                    ctasks_j.add(tx)
                    ctasks_j.update(self._ctasks[tx.id])
            else:
                if ct_i:
                    r_i = max(r_i, fr)
                if ct_j:
                    r_j = max(r_j, fr)
                hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j

    def ef_XYAB(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = max(st_j, ft_i) + self._RT[tj]
        pt_l_i, pt_l_ij, pt_l_j, r_i, r_j = self.merge_gain(ti, tj)
        if ft_i + pt_l_i - pt_l_ij <= ft_j - self._RT[tj]:
            if pt_l_ij:
                ft_ai = max(ft_j + pt_l_ij, ft_i + r_i)
            else:
                ft_ai = ft_i + max(pt_l_i, r_i)
            ft_aj = ft_j + max(pt_l_j, r_j)
        else:
            ft_ai = ft_i + max(pt_l_i + self._RT[tj], r_i)
            ft_aj = max(ft_i + pt_l_i + pt_l_j + self._RT[tj] - pt_l_ij,
                        ft_j + r_j)
            # print("XYAB:", self.problem.tasks[ti], self.problem.tasks[tj],
            # ft_i, ft_j, self.merge_gain(ti, tj), ft_ai, ft_aj)
        return max(ft_ai, ft_aj), ft_ai, ft_aj


class CAN8_6(CAN8_5):
    @memo
    def merge_gain(self, ti, tj):
        ts = {}
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i, ft_j = 0, self._RT[tj]
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            d = sum(self._RT[t.id]
                    for t in (self._ctasks[tx.id] | {tx}) - ctasks)
            fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
            if hlc > 0:
                if ct_i >= hlc:
                    frt = cst + ct_i + ct_j - hlc
                else:
                    if ct_j > 0:
                        frt = cst + ct_j
                    else:
                        frt = cst - hlc + ct_i
            else:
                frt = cst + ct_i + ct_j
            fr = frt + self.RP[tx.id]
            # fr = self.ef_XA_YB(ti, tx.id, 0, frt)[2]
            # fr = max(fr, self.ef_XA_YB(tj, tx.id, self._RT[tj], frt)[2])
            # print(tx, fl, fr)
            if (fl, tst) <= (fr, fr - self.RP[tx.id]):
                # print("fl", tx, fl, tst)
                tst += self._RT[tx.id]
                lft += d
                ctasks.add(tx)
                ctasks.update(self._ctasks[tx.id])
                if ct_i:
                    if self.PT_r[tx.id] > 0:
                        r_i = max(r_i, tst + self.PT_r[tx.id])
                    ctasks_i.add(tx)
                    ctasks_i.update(self._ctasks[tx.id])
                if ct_j:
                    if self.PT_r[tx.id] > 0:
                        r_j = max(r_j, tst + self.PT_r[tx.id])
                    ctasks_j.add(tx)
                    ctasks_j.update(self._ctasks[tx.id])
            else:
                # print("fr", tx, fr)
                if ct_i:
                    r_i = max(r_i, fr)
                if ct_j:
                    r_j = max(r_j, fr)
                hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        # print(">", self.problem.tasks[ti], self.problem.tasks[tj], ctasks_i, ctasks_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j


class CAN8_6_1(CAN8_6):
    def default_fts_in_mg(self, ti, tj):
        return 0, self._RT[tj]

    @memo
    def merge_gain(self, ti, tj):
        ft_i, ft_j = self.default_fts_in_mg(ti, tj)
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        ts = {}
        mi = self.PL_m(self.problem.tasks[ti]) if self._placed[ti] else None
        task_j = self.problem.tasks[tj]
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task.id == tj:
                continue
            if self.is_successor(c.to_task, task_j):
                ft_j += self._RT[c.to_task.id]
                if self.PL_m(c.to_task) == mi:
                    ctasks.add(c.to_task)
                    ctasks.update(
                        self._ctasks[c.to_task.id] - self._ctasks[tj])
                    ctasks_i.add(c.to_task)
                    ctasks_i.update(
                        self._ctasks[c.to_task.id] - self._ctasks[tj])
                    if self.PT_r[c.to_task.id] > 0:
                        r_i = max(r_i, self.FT(c.to_task) +
                                  self.PT_r[c.to_task.id])
                else:
                    r_i = max(r_i, self.FT(c.to_task) + self.PT[c.to_task.id])
                continue
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)
        if task_j in ctasks:
            ctasks.remove(task_j)
        if task_j in ctasks_i:
            ctasks_i.remove(task_j)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            d = sum(self._RT[t.id]
                    for t in (self._ctasks[tx.id] | {tx}) - ctasks)
            fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
            if hlc > 0:
                if ct_i >= hlc:
                    frt = cst + ct_i + ct_j - hlc
                else:
                    if ct_j > 0:
                        frt = cst + ct_j
                    else:
                        frt = cst - hlc + ct_i
            else:
                frt = cst + ct_i + ct_j
            fr = frt + self.RP[tx.id]
            # print("!", tx, fl, fr)
            if (fl, tst) <= (fr, fr - self.RP[tx.id]):
                tst += self._RT[tx.id]
                lft += d
                ctasks.add(tx)
                ctasks.update(self._ctasks[tx.id])
                if ct_i:
                    if self.PT_r[tx.id] > 0:
                        r_i = max(r_i, tst + self.PT_r[tx.id])
                    ctasks_i.add(tx)
                    ctasks_i.update(self._ctasks[tx.id])
                if ct_j:
                    if self.PT_r[tx.id] > 0:
                        r_j = max(r_j, tst + self.PT_r[tx.id])
                    ctasks_j.add(tx)
                    ctasks_j.update(self._ctasks[tx.id])
            else:
                if ct_i:
                    r_i = max(r_i, fr)
                if ct_j:
                    r_j = max(r_j, fr)
                hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        # print(self.problem.tasks[ti], self.problem.tasks[tj], pt_l_i, pt_l_j, pt_l_ij, r_i - ft_i, r_j - ft_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j

    def fitness(self, task, machine, comm_pls, st):
        all_ft = []
        task_ft = st + self._RT[task.id] + self.PT[task.id]
        wrk_ft = task_ft
        cur_ft = task_ft

        for t in self.problem.tasks:
            if self._placed[t.id] and \
                    any(not self._placed[c.to_task.id]
                        for c in t.communications(COMM_OUTPUT)
                        if c.to_task != task):
                st_t = self.ST(t)
                if self.PL_m(t) != machine:
                    ft_a, ft_i, ft_j = self.ef_XA_YB(t.id, task.id, st_t, st)
                    # print(task, "C1", t, ft_a, ft_i, t_j)
                else:
                    ft_a, ft_i, ft_j = self.ef_XYAB(t.id, task.id, st_t, st)
                    # print(task, "C2", t, ft_a, ft_i, ft_j)
                all_ft.append(ft_i)
                cur_ft = max(cur_ft, ft_j)
                wrk_ft = max(wrk_ft, ft_a)
        all_ft.append(cur_ft)
        # print(task, wrk_ft, task_ft)
        return wrk_ft, task_ft, sorted(all_ft, reverse=True)


class CAN8_6_1_1(CAN8_6_1):
    def default_fts_in_mg(self, ti, tj):
        ft_i = self.RA[ti]
        ft_j = max(ft_i, self.AT[tj]) + self._RT[tj]
        return ft_i, ft_j

    @memo
    def merge_gain(self, ti, tj):
        ft_i, ft_j = self.default_fts_in_mg(ti, tj)
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        ts = {}
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task.id == tj or self.is_successor(c.to_task, self.problem.tasks[tj]):
                continue
            if self._placed[c.to_task.id]:
                if self.PL_m(c.to_task) == self.PL_m(self.problem.tasks[ti]):
                    ctasks.add(c.to_task)
                    ctasks.update(self._ctasks[c.to_task.id])
                    ctasks_i.add(c.to_task)
                    ctasks_i.update(self._ctasks[c.to_task.id])
                    if self.PT_r[c.to_task.id]:
                        r_i = max(r_i, self.FT(c.to_task) +
                                  self.PT_r[c.to_task.id])
                else:
                    r_i = max(r_i, self.FT(c.to_task) + self.PT[c.to_task.id])
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            d = sum(self._RT[t.id]
                    for t in (self._ctasks[tx.id] | {tx}) - ctasks)
            fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
            if hlc > 0:
                if ct_i >= hlc:
                    frt = cst + ct_i + ct_j - hlc
                else:
                    if ct_j > 0:
                        frt = cst + ct_j
                    else:
                        frt = cst - hlc + ct_i
            else:
                frt = cst + ct_i + ct_j
            fr = frt + self.RP[tx.id]
            # print("!", tx, fl, fr)
            if (fl, tst) <= (fr, fr - self.RP[tx.id]):
                tst += self._RT[tx.id]
                lft += d
                ctasks.add(tx)
                ctasks.update(self._ctasks[tx.id])
                if ct_i:
                    if self.PT_r[tx.id] > 0:
                        r_i = max(r_i, tst + self.PT_r[tx.id])
                    ctasks_i.add(tx)
                    ctasks_i.update(self._ctasks[tx.id])
                if ct_j:
                    if self.PT_r[tx.id] > 0:
                        r_j = max(r_j, tst + self.PT_r[tx.id])
                    ctasks_j.add(tx)
                    ctasks_j.update(self._ctasks[tx.id])
            else:
                if ct_i:
                    r_i = max(r_i, fr)
                if ct_j:
                    r_j = max(r_j, fr)
                hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        # print(self.problem.tasks[ti], self.problem.tasks[tj], pt_l_i, pt_l_j, pt_l_ij, r_i - ft_i, r_j - ft_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j


class CAN8_6_2(CAN8_6):
    def sort_succs_2(self, ts):
        # return sorted(ts.items(), key=lambda x:self.RP[x[0].id]+x[1][1], reverse=True)
        return sorted(ts.items(), key=lambda x: self.RP[x[0].id], reverse=True)

    def default_fts_in_mg(self, ti, tj):
        return 0, self._RT[tj]

    @memo
    def merge_gain(self, ti, tj):
        ft_i, ft_j = self.default_fts_in_mg(ti, tj)
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        jc_started = False
        scti = 0
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        ts = {}
        mi = self.PL_m(self.problem.tasks[ti]) if self._placed[ti] else None
        task_j = self.problem.tasks[tj]
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task.id == tj:
                continue
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            d = sum(self._RT[t.id]
                    for t in (self._ctasks[tx.id] | {tx}) - ctasks - {task_j})
            if self.is_successor(tx, task_j):
                fl = max(lft + d, tst -
                         self._RT[tj] + self._RT[tx.id] + self.PT_r[tx.id])
            else:
                fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
            if hlc > 0:
                if ct_i >= hlc:
                    frt = cst + ct_i + ct_j - hlc
                else:
                    if ct_j > 0:
                        frt = cst + ct_j
                    else:
                        frt = cst - hlc + ct_i
            else:
                frt = cst + ct_i + ct_j
            fr = frt + self.RP[tx.id]
            # print("!", tx, fl, fr, tst, r_i, r_j, self.PT_r[tx.id])
            if (fl, tst) <= (fr, frt):
                tst += self._RT[tx.id]
                lft += d
                ctasks.add(tx)
                ctasks.update(self._ctasks[tx.id])
                if task_j in ctasks:
                    ctasks.remove(task_j)
                if ct_i:
                    if self.PT_r[tx.id] > 0:
                        if self.is_successor(tx, task_j):
                            r_i = max(
                                r_i, tst + self.PT_r[tx.id] - self._RT[tj])
                        else:
                            r_i = max(r_i, tst + self.PT_r[tx.id])
                    ctasks_i.add(tx)
                    ctasks_i.update(self._ctasks[tx.id])
                    if task_j in ctasks_i:
                        ctasks_i.remove(task_j)
                if ct_j:
                    if self.PT_r[tx.id] > 0:
                        r_j = max(r_j, tst + self.PT_r[tx.id])
                    ctasks_j.add(tx)
                    ctasks_j.update(self._ctasks[tx.id])
                    if task_j in ctasks_j:
                        ctasks_j.remove(task_j)
            else:
                if ct_i:
                    r_i = max(r_i, fr)
                    if not jc_started and not ct_j:
                        # if not ct_j:
                        scti += ct_i
                if ct_j:
                    jc_started = True
                    r_j = max(r_j, fr)
                hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        # print(">>", self.problem.tasks[ti], self.problem.tasks[tj], ctasks_i,
        # ctasks_j, pt_l_i, pt_l_j, pt_l_ij, r_i - ft_i, r_j - ft_j, ft_i, ft_j, scti)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j, scti

    def fitness(self, task, machine, comm_pls, st):
        all_ft = []
        task_ft = st + self._RT[task.id] + self.PT[task.id]
        wrk_ft = task_ft
        cur_ft = task_ft

        for t in self.problem.tasks:
            if self._placed[t.id] and \
                    any(not self._placed[c.to_task.id]
                        for c in t.communications(COMM_OUTPUT)
                        if c.to_task != task):
                st_t = self.ST(t)
                if self.PL_m(t) != machine:
                    ft_a, ft_i, ft_j = self.ef_XA_YB(t.id, task.id, st_t, st)
                    # print(task, "C1", t, ft_a, ft_i, ft_j)
                else:
                    ft_a, ft_i, ft_j = self.ef_XYAB(t.id, task.id, st_t, st)
                    # print(task, "C2", t, ft_a, ft_i, ft_j)
                all_ft.append(ft_i)
                cur_ft = max(cur_ft, ft_j)
                wrk_ft = max(wrk_ft, ft_a)
        all_ft.append(cur_ft)
        # print(task, wrk_ft, task_ft)
        return wrk_ft, task_ft, sorted(all_ft, reverse=True)

    def ef_XYAB(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = max(st_j, ft_i) + self._RT[tj]
        pt_l_i, pt_l_ij, pt_l_j, r_i, r_j, scti = self.merge_gain(ti, tj)
        if ft_i + pt_l_i - pt_l_ij <= ft_j - self._RT[tj]:
            if pt_l_ij:
                ft_ai = max(ft_j + pt_l_ij, ft_i + r_i)
            else:
                ft_ai = ft_i + max(pt_l_i, r_i)
            # ft_aj = ft_j + max(pt_l_j, r_j)
            ft_aj = ft_j + pt_l_j
        else:
            ft_ai = ft_i + max(pt_l_i + self._RT[tj], r_i)
            ft_aj = ft_i + pt_l_i + pt_l_j + self._RT[tj] - pt_l_ij
        ft_aj = max(ft_aj,
                    r_j + min(self._RT[tj], scti) + max(ft_i, ft_j - scti))
        # print("XYAB:", self.problem.tasks[ti], self.problem.tasks[tj],
        # ft_i, ft_j, self.merge_gain(ti, tj), ft_ai, ft_aj)
        return max(ft_ai, ft_aj), ft_ai, ft_aj


class CAN8_6_3(CAN8_5):
    @memo
    def merge_gain(self, ti, tj):
        ts = {}
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i, ft_j = 0, self._RT[tj]
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        jc_started = False
        scti = 0
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            d = sum(self._RT[t.id]
                    for t in (self._ctasks[tx.id] | {tx}) - ctasks)
            fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
            if hlc > 0:
                if ct_i >= hlc:
                    frt = cst + ct_i + ct_j - hlc
                else:
                    if ct_j > 0:
                        frt = cst + ct_j
                    else:
                        frt = cst - hlc + ct_i
            else:
                frt = cst + ct_i + ct_j
            fr = frt + self.RP[tx.id]
            # fr = self.ef_XA_YB(ti, tx.id, 0, frt)[2]
            # fr = max(fr, self.ef_XA_YB(tj, tx.id, self._RT[tj], frt)[2])
            # print(tx, fl, fr)
            if (fl, tst) <= (fr, fr - self.RP[tx.id]):
                # print("fl", tx, fl, tst)
                tst += self._RT[tx.id]
                lft += d
                ctasks.add(tx)
                ctasks.update(self._ctasks[tx.id])
                if ct_i:
                    if self.PT_r[tx.id] > 0:
                        r_i = max(r_i, tst + self.PT_r[tx.id])
                    ctasks_i.add(tx)
                    ctasks_i.update(self._ctasks[tx.id])
                if ct_j:
                    if self.PT_r[tx.id] > 0:
                        r_j = max(r_j, tst + self.PT_r[tx.id])
                    ctasks_j.add(tx)
                    ctasks_j.update(self._ctasks[tx.id])
            else:
                # print("fr", tx, fr)
                if ct_i:
                    r_i = max(r_i, fr)
                    if not jc_started and not ct_j:
                        scti += ct_i
                if ct_j:
                    r_j = max(r_j, fr)
                hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        # print(">", self.problem.tasks[ti], self.problem.tasks[tj], ctasks_i, ctasks_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j, scti

    def ef_XYAB(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = max(st_j, ft_i) + self._RT[tj]
        pt_l_i, pt_l_ij, pt_l_j, r_i, r_j, scti = self.merge_gain(ti, tj)
        if ft_i + pt_l_i - pt_l_ij <= ft_j - self._RT[tj]:
            if pt_l_ij:
                ft_ai = max(ft_j + pt_l_ij, ft_i + r_i)
            else:
                ft_ai = ft_i + max(pt_l_i, r_i)
            # ft_aj = ft_j + max(pt_l_j, r_j)
            ft_aj = ft_j + pt_l_j
        else:
            ft_ai = ft_i + max(pt_l_i + self._RT[tj], r_i)
            ft_aj = ft_i + pt_l_i + pt_l_j + self._RT[tj] - pt_l_ij
        ft_aj = max(ft_aj,
                    r_j + min(self._RT[tj], scti) + max(ft_i, ft_j - scti))
        # print("XYAB:", self.problem.tasks[ti], self.problem.tasks[tj], st_i, st_j,
        # ft_i, ft_j, self.merge_gain(ti, tj), ft_ai, ft_aj)
        return max(ft_ai, ft_aj), ft_ai, ft_aj


class CAN8_6_4(CAN8_6_3):
    def fitness(self, task, machine, comm_pls, st):
        all_ft = []
        task_ft = st + self._RT[task.id] + self.PT[task.id]
        wrk_ft = task_ft
        cur_ft = task_ft

        for t in self.problem.tasks:
            if self._placed[t.id] and \
                    any(not self._placed[c.to_task.id]
                        for c in t.communications(COMM_OUTPUT)
                        if c.to_task != task):
                # print("MEET", t, task)
                st_t = self.ST(t)
                if self.PL_m(t) != machine:
                    ft_a, ft_i, ft_j = self.ef_XA_YB(t.id, task.id, st_t, st)
                    # print("CSEA 1", t, ft_a, ft_i, ft_j, st_t, st)
                elif self.is_successor(t, task):
                    ft_j = st + self._RT[task.id] + self.PT[task.id]
                    ft_a = ft_i = max(ft_j, self.FT(t) + self.PT[t.id])
                else:
                    if st_t <= st:
                        ft_a, ft_i, ft_j = self.ef_XYAB(
                            t.id, task.id, st_t, st)
                    else:
                        ft_a, ft_j, ft_i = self.ef_XYAB(
                            t.id, task.id, st, st_t)
                    # print("CSEA 3", t, ft_a, ft_i, ft_j, st_t, st)
                all_ft.append(ft_i)
                cur_ft = max(cur_ft, ft_j)
                wrk_ft = max(wrk_ft, ft_a)
                # print(t, ft_a, ft_i, ft_j)
        all_ft.append(cur_ft)
        return wrk_ft, task_ft, sorted(all_ft, reverse=True)


class CAN8_6_5(CAN8_6_4):
    @memo
    def merge_gain(self, ti, tj):
        ts = {}
        mi = self.PL_m(self.problem.tasks[ti]) if self._placed[ti] else None
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i, ft_j = 0, self._RT[tj]
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        jc_started = False
        scti = 0
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            if self._placed[tx.id]:
                stx = self.ST(tx) - self.FT(self.problem.tasks[ti])
                if self.PL_m(tx) == mi:
                    d = sum(self._RT[t.id]
                            for t in (self._ctasks[tx.id] | {tx}) - ctasks)
                    tst = max(tst, stx) + self._RT[tx.id]
                    lft = max(lft, stx) + d
                    if self.PT_r[tx.id] > 0:
                        r_i = max(
                            r_i, stx + self._RT[tx.id] + self.PT_r[tx.id])
                    ctasks.add(tx)
                    ctasks.update(self._ctasks[tx.id])
                    ctasks_i.add(tx)
                    ctasks_i.update(self._ctasks[tx.id])
                else:
                    r_i = stx + self._RT[tx.id] + self.PT[tx.id]
                    if not jc_started and not ct_j:
                        scti += ct_i
            else:
                d = sum(self._RT[t.id]
                        for t in (self._ctasks[tx.id] | {tx}) - ctasks)
                fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
                if hlc > 0:
                    if ct_i >= hlc:
                        frt = cst + ct_i + ct_j - hlc
                    else:
                        if ct_j > 0:
                            frt = cst + ct_j
                        else:
                            frt = cst - hlc + ct_i
                else:
                    frt = cst + ct_i + ct_j
                fr = frt + self.RP[tx.id]
                # fr = self.ef_XA_YB(ti, tx.id, 0, frt)[2]
                # fr = max(fr, self.ef_XA_YB(tj, tx.id, self._RT[tj], frt)[2])
                # print(tx, fl, fr)
                if (fl, tst) <= (fr, fr - self.RP[tx.id]):
                    # print("fl", tx, fl, tst)
                    tst += self._RT[tx.id]
                    lft += d
                    ctasks.add(tx)
                    ctasks.update(self._ctasks[tx.id])
                    if ct_i:
                        if self.PT_r[tx.id] > 0:
                            r_i = max(r_i, tst + self.PT_r[tx.id])
                        ctasks_i.add(tx)
                        ctasks_i.update(self._ctasks[tx.id])
                    if ct_j:
                        if self.PT_r[tx.id] > 0:
                            r_j = max(r_j, tst + self.PT_r[tx.id])
                        ctasks_j.add(tx)
                        ctasks_j.update(self._ctasks[tx.id])
                else:
                    # print("fr", tx, fr)
                    if ct_i:
                        r_i = max(r_i, fr)
                        if not jc_started and not ct_j:
                            scti += ct_i
                    if ct_j:
                        r_j = max(r_j, fr)
                    hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        # print(">", self.problem.tasks[ti], self.problem.tasks[tj], ctasks_i, ctasks_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j, scti


class CAN8_7(CAN8_5):
    # def sort_succs_2(self, ts):
        # return sorted(ts.items(),
                      # key=lambda t: - self.RP[t[0].id])

    @memo
    def merge_gain(self, ti, tj):
        ft_i, ft_j = 0, self._RT[tj]
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        ts = {}
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task.id == tj:
                continue
            if self.is_successor(c.to_task, self.problem.tasks[tj]):
                ft_j += self._RT[c.to_task.id]
                ctasks.add(c.to_task)
                # ctasks.update(self._ctasks[c.to_task.id] - {self.problem.tasks[tj]})
                ctasks_i.add(c.to_task)
                # ctasks_i.update(self._ctasks[c.to_task.id] - {self.problem.tasks[tj]})
                continue
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            d = sum(self._RT[t.id]
                    for t in (self._ctasks[tx.id] | {tx}) - ctasks)
            fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
            if hlc > 0:
                if ct_i >= hlc:
                    frt = cst + ct_i + ct_j - hlc
                else:
                    if ct_j > 0:
                        frt = cst + ct_j
                    else:
                        frt = cst - hlc + ct_i
            else:
                frt = cst + ct_i + ct_j
            fr = frt + self.RP[tx.id]
            # fr = self.ef_XA_YB(ti, tx.id, 0, frt)[2]
            # fr = max(fr, self.ef_XA_YB(tj, tx.id, self._RT[tj], frt)[2])
            # print("!", tx, fl, fr, ct_i, ct_j, frt)
            if (fl, tst) <= (fr, fr - self.RP[tx.id]):
                # print("fl", tx, fl, tst)
                tst += self._RT[tx.id]
                lft += d
                ctasks.add(tx)
                ctasks.update(self._ctasks[tx.id])
                if ct_i:
                    if self.PT_r[tx.id] > 0:
                        r_i = max(r_i, tst + self.PT_r[tx.id])
                    ctasks_i.add(tx)
                    ctasks_i.update(self._ctasks[tx.id])
                if ct_j:
                    if self.PT_r[tx.id] > 0:
                        r_j = max(r_j, tst + self.PT_r[tx.id])
                    ctasks_j.add(tx)
                    ctasks_j.update(self._ctasks[tx.id])
            else:
                # print("fr", tx, fr)
                if ct_i:
                    r_i = max(r_i, fr)
                if ct_j:
                    r_j = max(r_j, fr)
                hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        # print(">", self.problem.tasks[ti], self.problem.tasks[tj],
        # ctasks_i, ctasks_j, r_i - ft_i, r_j - ft_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j


class CAN8_8(CAN8_5):
    @memo
    def merge_gain(self, ti, tj):
        ts = {}
        for c in self.problem.tasks[ti].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].communications(COMM_OUTPUT):
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i, ft_j = 0, self._RT[tj]
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        lts = []
        for tx, [ct_i, ct_j] in self.sort_succs_2(ts):
            d = sum(self._RT[t.id]
                    for t in (self._ctasks[tx.id] | {tx}) - ctasks)
            fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
            if hlc > 0:
                if ct_i >= hlc:
                    frt = cst + ct_i + ct_j - hlc
                else:
                    if ct_j > 0:
                        frt = cst + ct_j
                    else:
                        frt = cst - hlc + ct_i
            else:
                frt = cst + ct_i + ct_j
            fr = max([self.ef_XA_YB(lt.id, tx.id, ls, frt)[2] for ls, lt in lts],
                     default=frt + self.RP[tx.id])
            if (fl, tst) <= (fr, frt):
                # print("fl", tx, fl, tst)
                lts.append((tst, tx))
                tst += self._RT[tx.id]
                lft += d
                ctasks.add(tx)
                ctasks.update(self._ctasks[tx.id])
                if ct_i:
                    if self.PT_r[tx.id] > 0:
                        r_i = max(r_i, tst + self.PT_r[tx.id])
                    ctasks_i.add(tx)
                    ctasks_i.update(self._ctasks[tx.id])
                if ct_j:
                    if self.PT_r[tx.id] > 0:
                        r_j = max(r_j, tst + self.PT_r[tx.id])
                    ctasks_j.add(tx)
                    ctasks_j.update(self._ctasks[tx.id])
            else:
                # print("fr", tx, fr)
                if ct_i:
                    r_i = max(r_i, fr)
                if ct_j:
                    r_j = max(r_j, fr)
                hlc, cst = self.update_cst_hlc(hlc, cst, ct_i, ct_j)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        # print(">", self.problem.tasks[ti], self.problem.tasks[tj], ctasks_i, ctasks_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j


class CAN8_9(CAN8_6):
    def ef_XYAB(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = st_j + self._RT[tj]
        if ft_j > st_i and st_j < ft_i:
            ft_j = ft_i + self._RT[tj]
        pt_l_i, pt_l_ij, pt_l_j, r_i, r_j = self.merge_gain(ti, tj)
        if ft_i + pt_l_i - pt_l_ij <= ft_j - self._RT[tj]:
            if pt_l_ij:
                ft_ai = max(ft_j + pt_l_ij, ft_i + r_i)
            else:
                ft_ai = ft_i + max(pt_l_i, r_i)
            ft_aj = ft_j + max(pt_l_j, r_j)
        else:
            ft_ai = ft_i + max(pt_l_i + self._RT[tj], r_i)
            ft_aj = max(ft_i + pt_l_i + pt_l_j + self._RT[tj] - pt_l_ij,
                        ft_j + r_j)
        # print(self.problem.tasks[ti], self.problem.tasks[tj],
            # ft_i, ft_j, self.merge_gain(ti, tj), ft_ai, ft_aj)
        return max(ft_ai, ft_aj), ft_ai, ft_aj


class CAN8_10(CAN8_7):
    @memo
    def ft_ft(self, tx, task, machine, st):
        if tx == task:
            ft_a = ft_j = ft_i = st + self.RP[tx.id]
        elif not self._placed[task.id]:
            if id(machine) in self.BM[task.id]:
                if task in tx.succs():
                    ft_a0 = ft_j0 = ft_i0 = st + self.RP[tx.id]
                else:
                    ft_a0, ft_j0, ft_i0 = self.ef_XYAB(
                        tx.id, task.id, st, self.AT[task.id])
            else:
                ft_a0, ft_j0, ft_i0 = self.ef_XYAB(
                    tx.id, task.id, st, self.ATO[task.id])
            if id(machine) in self.BM[task.id] and len(self.BM[task.id]) == 1:
                ft_a1, ft_j1, ft_i1 = self.ef_XA_YB(
                    tx.id, task.id, st, self.ATO[task.id])
            else:
                ft_a1, ft_j1, ft_i1 = self.ef_XA_YB(
                    tx.id, task.id, st, self.AT[task.id])
            if ft_a0 < ft_a1:
                ft_a, ft_i, ft_j = ft_a0, ft_i0, ft_j0
            else:
                ft_a, ft_i, ft_j = ft_a1, ft_i1, ft_j1
        elif not any(self._placed[t.id] for t in task.succs()) and not self.is_successor(task, tx):
            # elif not self.is_successor(task, tx):
            # else:
            # if any(not self._placed[t.id] for t in task.succs() if t != tx):
            if self.PL_m(task) == machine:
                ft_a, ft_i, ft_j = self.ef_XYAB(
                    task.id, tx.id, self.ST(task), st)
                # print("A", task, tx, self.ST(task), st)
            else:
                ft_a, ft_i, ft_j = self.ef_XA_YB(
                    task.id, tx.id, self.ST(task), st)
            # else:
            # ft_a = ft_i = ft_j = st + self.RP[tx.id]
        else:
            ft_a, ft_i, ft_j = 0, 0, 0
            # for t in [t for t in task.succs() if self._placed[t.id]]:
            for t in task.succs():
                ft_a0, ft_j0, ft_i0 = self.ft_ft(tx, t, machine, st)
                ft_a = max(ft_a, ft_a0)
                ft_i = max(ft_i, ft_i0)
                ft_j = max(ft_j, ft_j0)
        # print(tx, task, st, ft_a, ft_i, ft_j)
        return ft_a, ft_i, ft_j

    def fitness(self, task, machine, comm_pls, st):
        all_ft = []
        task_ft = st + self._RT[task.id] + self.PT[task.id]
        wrk_ft = task_ft
        cur_ft = task_ft

        for t in self.problem.tasks:
            if self._placed[t.id] and any(not self._placed[_t.id] for _t in t.succs()):
                ft_a, ft_i, ft_j = self.ft_ft(task, t, machine, st)
                all_ft.append(ft_i)
                cur_ft = max(cur_ft, ft_j)
                wrk_ft = max(wrk_ft, ft_a)
                # print(t, st, ft_a


class PTv1(Heuristic):
    def sort_succs(self, comms):
        return sorted(comms, key=lambda c: self.CT(c) + self.RP[c.to_task.id], reverse=True)

    @memo
    def calculate_PT(self, task):
        comms = task.communications(COMM_OUTPUT)
        pto = 0
        tc = 0
        for c in sorted(comms,
                        key=lambda c: self.RP[c.to_task.id],
                        reverse=True):
            tc += self.CT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])

        local_ft = 0
        local_st = 0
        comm_ft = 0
        remote_ft = 0
        self._cdeps[task.id] = Counter()
        self._ctasks[task.id] = set()
        # for c in sorted(comms,
        # key=lambda c: self.CT(c) + self.RP[c.to_task.id],
        # reverse=True):
        for c in self.sort_succs(comms):
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


class PTv2(PTv1):
    def sort_succs(self, comms):
        w = {c: 0 for c in comms}
        for c0 in comms:
            for c1 in comms:
                if c0 == c1:
                    continue
                x, y = self.CT(c0), self.CT(c1)
                A, B = self.RP[c0.to_task.id], self.RP[c1.to_task.id]
                if min(A, y) + B < A + x:
                    w[c1] += 1
                if A + min(B, x) < B + y:
                    w[c0] += 1
        return sorted(comms, key=lambda c: (w[c], -self.CT(c) - self.RP[c.to_task.id]))


class PTv3(Heuristic):
    def choose_succ(self, comms, t, c):
        w = {c: 0 for c in comms}
        for c0 in comms:
            for c1 in comms:
                if c0 == c1:
                    continue
                x, y = self.CT(c0), self.CT(c1)
                A, B = self.RP[c0.to_task.id], self.RP[c1.to_task.id]
                if c + x < t:
                    t0 = max(c + x + A, t + B)
                else:
                    t0 = min(t + A + B, max(t + A, c + y + B))
                if c + y < t:
                    t1 = max(c + y + B, t + A)
                else:
                    t1 = min(t + A + B, max(t + B, c + x + A))
                if t0 > t1:
                    w[c0] += 1
        task = min(comms, key=lambda c: (w[c], -self.CT(c)))
        return task

    @memo
    def calculate_PT(self, task):
        comms = copy(task.communications(COMM_OUTPUT))
        pto = 0
        tc = 0
        for c in sorted(comms,
                        key=lambda c: self.RP[c.to_task.id],
                        reverse=True):
            tc += self.CT(c)
            pto = max(pto, tc + self.RP[c.to_task.id])

        local_ft = 0
        local_st = 0
        comm_ft = 0
        remote_ft = 0
        self._cdeps[task.id] = Counter()
        self._ctasks[task.id] = set()
        while comms:
            c = self.choose_succ(comms, local_st, comm_ft)
            comms.remove(c)
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
        # print(task, self.PT[task.id])


class CSLTest(Heuristic):
    @memo
    def task_distance(self, tx, ty):
        return self.PT[tx.id] - self.RP[ty.id]

    def csl_sort(self, tasks):
        return sorted(tasks, key=lambda _t: -self.RP[_t.id])

    def ef_XA_YB(self, ti, tj, st_i, st_j):
        pti, ptj = self.comm_succ_len(ti, tj)
        ft_ai = st_i + self._RT[ti] + pti
        ft_aj = st_j + self._RT[tj] + ptj
        # print("XA_YB", self.problem.tasks[ti], self.problem.tasks[tj], st_i + self._RT[ti], st_j + self._RT[tj], pti, ptj, ft_ai, ft_aj)
        return max(ft_ai, ft_aj), ft_ai, ft_aj

    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = ctasks_i & ctasks_j
        rft_i = self.PT_r[ti]
        rft_j = self.PT_r[tj]
        dt_i, dt_j = 0, 0

        for t in self.csl_sort(cts):
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

        pti = max(rft_i, sum(self._RT[t.id] for t in ctasks_i))
        ptj = max(rft_j, sum(self._RT[t.id] for t in ctasks_j))
        # print(self.problem.tasks[ti], self.problem.tasks[tj], pti, ptj, rft_i, rft_j)
        return pti, ptj


class CSLTest2(CSLTest):
    @memo
    def csl_st(self, c, ti, ft_i):
        return ft_i + self.task_distance(self.problem.tasks[ti], c.from_task) + self._RT[c.from_task.id]

    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = ctasks_i & ctasks_j
        ft_i, ft_j = 0, 0
        dt_i, dt_j = ft_i, ft_j
        rft_i = ft_i + self.PT_r[ti]
        rft_j = ft_j + self.PT_r[tj]

        for t in self.csl_sort(cts):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self.csl_st(c, ti, ft_i)):
                dt_i = max(dt_i, self.csl_st(c, ti, ft_i)) + self.CT(c)
            for c in sorted(ds_j, key=lambda c: self.csl_st(c, tj, ft_j)):
                dt_j = max(dt_j, self.csl_st(c, tj, ft_j)) + self.CT(c)
            if dt_i <= dt_j:
                dt_j = dt_j0
                rft_i = max(rft_i, dt_i + self.RP[t.id])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j, dt_j + self.RP[t.id])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        pti = max(rft_i - ft_i, sum(self._RT[t.id] for t in ctasks_i))
        ptj = max(rft_j - ft_j, sum(self._RT[t.id] for t in ctasks_j))
        # print(self.problem.tasks[ti], self.problem.tasks[tj], pti, ptj, rft_i, rft_j)
        return pti, ptj


class CSLTest3(CSLTest):
    @memo
    def csl_st(self, c, ti, ft_i):
        return ft_i + self.task_distance(self.problem.tasks[ti], c.from_task) + self._RT[c.from_task.id]

    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = ctasks_i & ctasks_j
        ft_i, ft_j = self.RA[ti], self.RA[tj]
        dt_i, dt_j = ft_i, ft_j
        rft_i = ft_i + self.PT_r[ti]
        rft_j = ft_j + self.PT_r[tj]

        for t in self.csl_sort(cts):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self.csl_st(c, ti, ft_i)):
                dt_i = max(dt_i, self.csl_st(c, ti, ft_i)) + self.CT(c)
            for c in sorted(ds_j, key=lambda c: self.csl_st(c, tj, ft_j)):
                dt_j = max(dt_j, self.csl_st(c, tj, ft_j)) + self.CT(c)
            if dt_i <= dt_j:
                dt_j = dt_j0
                rft_i = max(rft_i, dt_i + self.RP[t.id])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j, dt_j + self.RP[t.id])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        pti = max(rft_i - ft_i, sum(self._RT[t.id] for t in ctasks_i))
        ptj = max(rft_j - ft_j, sum(self._RT[t.id] for t in ctasks_j))
        return pti, ptj


class CSLTest4(CSLTest):
    @memo
    def csl_st(self, c, ti, ft_i):
        return ft_i + self.task_distance(self.problem.tasks[ti], c.from_task) + self._RT[c.from_task.id]

    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = ctasks_i & ctasks_j
        ft_i, ft_j = 0, 0
        dt_i, dt_j = ft_i, ft_j
        rft_i = ft_i + self.PT_r[ti]
        rft_j = ft_j + self.PT_r[tj]
        delayed = False

        for t in self.csl_sort(cts):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self.csl_st(c, ti, ft_i)):
                dt_i = max(dt_i, self.csl_st(c, ti, ft_i)) + self.CT(c)
            for c in sorted(ds_j, key=lambda c: self.csl_st(c, tj, ft_j)):
                dt_j = max(dt_j, self.csl_st(c, tj, ft_j)) + self.CT(c)
            if dt_i <= dt_j:
                dt_j = dt_j0
                rft_i = max(rft_i, dt_i + self.RP[t.id])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j, dt_j + self.RP[t.id])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)
                if self.RP[t.id] > 0:
                    delayed = True

        pti = sum(self._RT[t.id] for t in ctasks_i)
        ptj = sum(self._RT[t.id] for t in ctasks_j)
        return pti, ptj, rft_i - ft_i, rft_j - ft_j, delayed

    def ef_XA_YB(self, ti, tj, st_i, st_j):
        ptl_i, ptl_j, ptr_i, ptr_j, delayed = self.comm_succ_len(ti, tj)
        ft_i = st_i + self._RT[ti]
        ft_j = st_j + self._RT[tj]
        if delayed:
            ft_ai = max(ft_j + ptl_i, ft_i + ptr_i)
        else:
            ft_ai = ft_i + max(ptl_i, ptr_i)
        ft_aj = ft_j + max(ptl_j, ptr_j)
        return max(ft_ai, ft_aj), ft_ai, ft_aj


class CSLTest5(CSLTest):
    @memo
    def csl_st(self, c, ti, ft_i):
        return ft_i + self.task_distance(self.problem.tasks[ti], c.from_task) + self._RT[c.from_task.id]

    @memo
    def comm_succ_len(self, ti, tj, ft_i, ft_j):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        cts = ctasks_i & ctasks_j
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        dt_i, dt_j = ft_i, ft_j
        rft_i = ft_i + self.PT_r[ti]
        rft_j = ft_j + self.PT_r[tj]

        for t in self.csl_sort(cts):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self.csl_st(c, ti, ft_i)):
                dt_i = max(dt_i, self.csl_st(c, ti, ft_i)) + self.CT(c)
            for c in sorted(ds_j, key=lambda c: self.csl_st(c, tj, ft_j)):
                dt_j = max(dt_j, self.csl_st(c, tj, ft_j)) + self.CT(c)
            # print(self.problem.tasks[ti], self.problem.tasks[tj], t, ds_i, ds_j, dt_i, dt_j, self.PT[t.id], self.task_distance(self.problem.tasks[ti], t))
            if dt_i <= dt_j:
                dt_j = dt_j0
                rft_i = max(rft_i, dt_i + self.RP[t.id])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j, dt_j + self.RP[t.id])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        pti = sum(self._RT[t.id] for t in ctasks_i)
        ptj = sum(self._RT[t.id] for t in ctasks_j)
        # print(self.problem.tasks[ti], self.problem.tasks[tj], ctasks_i, ctasks_j, pti, ptj, rft_i - ft_i, rft_j - ft_j)
        return max(pti, rft_i - ft_i), max(ptj, rft_j - ft_j)

    def ef_XA_YB(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = st_j + self._RT[tj]
        pt_i, pt_j = self.comm_succ_len(ti, tj, ft_i, ft_j)
        ft_ai = ft_i + pt_i
        ft_aj = ft_j + pt_j
        # print(self.problem.tasks[ti], self.problem.tasks[tj], ft_ai, ft_aj, pt_i, pt_j, st_i, st_j)
        return max(ft_ai, ft_aj), ft_ai, ft_aj


class CSLTest6(CSLTest2):
    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = ctasks_i & ctasks_j
        ft_i, ft_j = 0, 0
        dt_i, dt_j = ft_i, ft_j
        rft_i = ft_i + self.PT_r[ti]
        rft_j = ft_j + self.PT_r[tj]
        lft_i, lft_j = ft_i, ft_j

        for t in self.csl_sort(cts):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self.csl_st(c, ti, ft_i)):
                dt_i = max(dt_i, self.csl_st(c, ti, ft_i)) + self.CT(c)
            for c in sorted(ds_j, key=lambda c: self.csl_st(c, tj, ft_j)):
                dt_j = max(dt_j, self.csl_st(c, tj, ft_j)) + self.CT(c)
            if dt_i <= dt_j:
                dt_j = dt_j0
                rft_i = max(rft_i, dt_i + self.RP[t.id])
                lft_j = max(lft_j, dt_i + self._RT[t.id])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j, dt_j + self.RP[t.id])
                lft_i = max(lft_i, dt_j + self._RT[t.id])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        pti = max(rft_i, lft_i, ft_i +
                  sum(self._RT[t.id] for t in ctasks_i)) - ft_i
        ptj = max(rft_j, lft_j, ft_j +
                  sum(self._RT[t.id] for t in ctasks_j)) - ft_j
        # print(self.problem.tasks[ti], self.problem.tasks[tj], pti, ptj, rft_i, rft_j)
        return pti, ptj


class CSLTest7(CSLTest2):
    def default_fts_in_sl(self, ti, tj):
        return self.RA[ti], self.RA[tj]

    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = ctasks_i & ctasks_j
        ft_i, ft_j = self.default_fts_in_sl(ti, tj)
        dt_i, dt_j = ft_i, ft_j
        rft_i = ft_i + self.PT_r[ti]
        rft_j = ft_j + self.PT_r[tj]
        lft_i, lft_j = ft_i, ft_j

        for t in self.csl_sort(cts):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self.csl_st(c, ti, ft_i)):
                dt_i = max(dt_i, self.csl_st(c, ti, ft_i)) + self.CT(c)
            for c in sorted(ds_j, key=lambda c: self.csl_st(c, tj, ft_j)):
                dt_j = max(dt_j, self.csl_st(c, tj, ft_j)) + self.CT(c)
            if dt_i <= dt_j:
                dt_j = dt_j0
                rft_i = max(rft_i, dt_i + self.RP[t.id])
                lft_j = max(lft_j, dt_i + self._RT[t.id])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j, dt_j + self.RP[t.id])
                lft_i = max(lft_i, dt_j + self._RT[t.id])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        pti = max(rft_i, lft_i, ft_i +
                  sum(self._RT[t.id] for t in ctasks_i)) - ft_i
        ptj = max(rft_j, lft_j, ft_j +
                  sum(self._RT[t.id] for t in ctasks_j)) - ft_j
        # print(self.problem.tasks[ti], self.problem.tasks[tj], pti, ptj, rft_i, rft_j)
        return pti, ptj


class CSLTest7_4(CSLTest2):
    def default_fts_in_sl(self, ti, tj):
        return self.RA[ti], self.RA[tj]

    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = ctasks_i & ctasks_j
        ft_i, ft_j = self.default_fts_in_sl(ti, tj)
        dt_i, dt_j = ft_i, ft_j
        rft_i = ft_i + self.PT_r[ti]
        rft_j = ft_j + self.PT_r[tj]
        lft_i, lft_j = ft_i, ft_j

        for t in self.csl_sort(cts):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self.csl_st(c, ti, ft_i)):
                dt_i = max(dt_i, self.csl_st(c, ti, ft_i)) + self.CT(c)
            for c in sorted(ds_j, key=lambda c: self.csl_st(c, tj, ft_j)):
                dt_j = max(dt_j, self.csl_st(c, tj, ft_j)) + self.CT(c)
            if dt_i <= dt_j:
                dt_j = dt_j0
                rft_i = max(rft_i, dt_i + self.RP[t.id])
                lft_j = max(lft_j, dt_i + self._RT[t.id])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j, dt_j + self.RP[t.id])
                lft_i = max(lft_i, dt_j + self._RT[t.id])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        pti = max(rft_i, ft_i +
                  sum(self._RT[t.id] for t in ctasks_i)) - ft_i
        ptj = max(rft_j, ft_j +
                  sum(self._RT[t.id] for t in ctasks_j)) - ft_j
        # print(self.problem.tasks[ti], self.problem.tasks[tj], pti, ptj, rft_i, rft_j)
        return pti, ptj


class CSLTest7_1(CSLTest7):
    def default_fts_in_sl(self, ti, tj):
        return 0, 0


class CSLTest7_2(CSLTest7):
    def default_fts_in_sl(self, ti, tj):
        return self.PT[tj], self.PT[ti]


class CSLTest7_3(CSLTest7):
    def default_fts_in_sl(self, ti, tj):
        return self.AT[ti], self.ATO[tj]


class CSLTest8(CSLTest2):
    @memo
    def comm_succ_len(self, ti, tj):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = ctasks_i & ctasks_j
        ft_i, ft_j = 0, 0
        dt_i, dt_j = ft_i, ft_j
        rft_i = ft_i + self.PT_r[ti]
        rft_j = ft_j + self.PT_r[tj]
        lft_i, lft_j = ft_i, ft_j

        for t in self.csl_sort(cts):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self.csl_st(c, ti, ft_i)):
                dt_i = max(dt_i, self.csl_st(c, ti, ft_i)) + self.CT(c)
            for c in sorted(ds_j, key=lambda c: self.csl_st(c, tj, ft_j)):
                dt_j = max(dt_j, self.csl_st(c, tj, ft_j)) + self.CT(c)
            if dt_i <= dt_j:
                dt_j = dt_j0
                rft_i = max(rft_i, dt_i + self.RP[t.id])
                lft_j = max(lft_j, dt_i + self._RT[t.id])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j, dt_j + self.RP[t.id])
                lft_i = max(lft_i, dt_j + self._RT[t.id])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        pti = sum(self._RT[t.id] for t in ctasks_i)
        ptj = sum(self._RT[t.id] for t in ctasks_j)
        # print(self.problem.tasks[ti], self.problem.tasks[tj], pti, ptj, rft_i, rft_j)
        return pti, lft_i - ft_i, rft_i - ft_i, ptj, lft_j - ft_j, rft_j - ft_j

    def ef_XA_YB(self, ti, tj, st_i, st_j):
        pti, lti, rti, ptj, ltj, rtj = self.comm_succ_len(ti, tj)
        ft_i = st_i + self._RT[ti]
        ft_j = st_j + self._RT[tj]
        if ft_j > ft_i:
            ft_ai = ft_i + max(pti, lti, rti)
            if ft_j - ft_i <= ltj - ptj:
                ft_aj = max(ft_i + ltj, ft_j + rtj)
            else:
                ft_aj = ft_j + max(ptj, ltj, rtj)
        else:
            if ft_i - ft_j <= lti - pti:
                ft_ai = max(ft_j + lti, ft_i + rti)
            else:
                ft_ai = ft_i + max(pti, lti, rti)
            ft_aj = ft_j + max(ptj, ltj, rtj)
        # print(self.problem.tasks[ti],
        # self.problem.tasks[tj], pti, ptj, ft_ai, ft_aj)
        return max(ft_ai, ft_aj), ft_ai, ft_aj


class SAT(Heuristic):
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
            at, ato, bmt = self._min2(
                at, ato, bmt, st, -id(task) if st == 0 else id(m))
        for t, d in St:
            st = max(self.RA[t.id], at_none - d)
            at, ato, bmt = self._min2(at, ato, bmt, st, id(t))
        if self.L > len(self.platform) or len(Sm) < len(self.platform):
            at, ato, bmt = self._min2(at, ato, bmt, at_none, -id(task))

        self.AT[task.id] = at
        self.ATO[task.id] = ato
        self.BM[task.id] = set(bmt)
        # print("AT", task, self.AT[task.id], self.ATO[task.id], self.BM[task.id])
        if not hasattr(self, "ATCS"):
            self.ATCS = [0] * self.problem.num_tasks
        self.ATCS[task.id] = 0
        for c in task.communications(COMM_INPUT):
            t = c.from_task
            if not (self._placed[t.id] and id(self.PL_m(t)) in bmt) or id(t) in bmt:
                self.ATCS[task.id] += self.CT(c)
        # print("!!!!ATCS", task, self.ATCS[task.id])
