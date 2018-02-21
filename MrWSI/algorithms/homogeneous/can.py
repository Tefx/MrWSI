from math import inf
from collections import Counter
from copy import copy
from functools import wraps
import networkx as nx

from .base import Heuristic
from MrWSI.core.problem import COMM_INPUT, COMM_OUTPUT


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


def _min2(at, ato, bmt, x, m):
    if x < at:
        return x, at, [m]
    if x == at:
        bmt.append(m)
        return x, at, bmt
    if x < ato:
        return at, x, bmt
    return at, ato, bmt


def _upd_cst_hlc(hlc, cst, ct_i, ct_j):
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


def _rsts(cti, ctj, ati, atj):
    if cti > 0 and ctj > 0 and -cti < atj - ati < ctj:
        atj = min(ati + ctj, atj + cti)
    return ati, atj


class CAWS(Heuristic):
    def _ti2t(self, i):
        return self.problem.tasks[i]

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
        self.PT = [None] * self.problem.num_tasks
        self.PT_l = [None] * self.problem.num_tasks
        self.PT_r = [None] * self.problem.num_tasks
        self.RA = [None] * self.problem.num_tasks
        self.RP = [None] * self.problem.num_tasks
        self.BM = [None] * self.problem.num_tasks
        self._placed = [False] * self.problem.num_tasks
        self._ctasks = [None] * self.problem.num_tasks
        self._cdeps = [None] * self.problem.num_tasks
        self.toporder = self._topsort()

    @memo
    def _is_successor(self, task_i, task_j):
        return task_j in task_i.succs() or \
            any(self._is_successor(t, task_j) for t in task_i.succs())

    def _ready_graph(self):
        self._edges.sort(key=lambda i: i[-1], reverse=True)
        rg = nx.DiGraph()
        for tx, ty, w in self._edges:
            rg.add_edge(tx, ty)
            try:
                rg.remove_edge(tx, ty)
            except nx.exception.NetworkXNoCycle:
                pass
        for t in self.ready_tasks:
            if t.id not in rg:
                rg.add_node(t.id)
        for t in rg.nodes():
            if rg.in_degree(t) == 0:
                yield self.problem.tasks[t]

    def _comm_st_in_SP(self, c, ti, ft_i):
        return ft_i + self.PT[ti] - self.PT[c.from_task.id]

    def _sort_succs_in_MP(self, ts):
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
        # print([(t, w[t], ct) for t,ct in ts.items()])
        return sorted(ts.items(),
                      key=lambda t: (w[t[0]], -sum(t[1]) - self.RP[t[0].id]))

    def sort_tasks(self):
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
        self._edges = []
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and \
                        not self.rids[tx] and not self.rids[ty]:
                    ftx = self.est_ft(tx, ty)
                    fty = self.est_ft(ty, tx)
                    if ftx < fty:
                        self._edges.append((tx, ty, fty))
                    elif ftx > fty:
                        self._edges.append((ty, tx, ftx))
        task = max(self._ready_graph(), key=lambda t: self.RP[t.id])
        # print(task)
        return task

    def default_fitness(self):
        return inf, inf, [inf]

    def fitness(self, task, machine, comm_pls, st):
        all_ft = []
        task_ft = st + self._RT[task.id] + self.PT[task.id]
        wrk_ft = task_ft
        cur_ft = task_ft

        for t in self.problem.tasks:
            if self._placed[t.id] and \
                    any(not self._placed[c.to_task.id]
                        for c in t.out_comms
                        if c.to_task != task):
                st_t = self.ST(t)
                if self.PL_m(t) != machine:
                    ft_a, ft_i, ft_j = self.split_FT(t.id, task.id, st_t, st)
                    # print("FT_1", task, t, ft_a)
                elif self._is_successor(t, task):
                    ft_j = st + self._RT[task.id] + self.PT[task.id]
                    ft_a = ft_i = max(ft_j, self.FT(t) + self.PT[t.id])
                    # print("FT_2", task, t, ft_a)
                else:
                    ft_a, ft_i, ft_j = self.merge_FT(t.id, task.id, st_t, st)
                    # print("FT_3", task, t, ft_a)
                all_ft.append(ft_i)
                cur_ft = max(cur_ft, ft_j)
                wrk_ft = max(wrk_ft, ft_a)
        all_ft.append(cur_ft)
        return wrk_ft, task_ft, sorted(all_ft, reverse=True)

    def update_AT_and_PT(self):
        for t in self.toporder:
            if self._placed[t.id]:
                m = self.PL_m(t)
                self.AT[t.id] = self.ST(t)
                self.RA[t.id] = m.earliest_idle_time_for_communication(
                    self.bandwidth, COMM_OUTPUT, self.FT(t))
            else:
                self.calculate_AT(t)
                self.RA[t.id] = self.AT[t.id] + self._RT[t.id]
        for t in reversed(self.toporder):
            if not self._placed[t.id]:
                self.calculate_PT(t)
                self.RP[t.id] = self.PT[t.id] + self._RT[t.id]

    def calculate_AT(self, task):
        comms = sorted(task.in_comms,
                       key=lambda c: self.RA[c.from_task.id])
        A = [None] * self.problem.num_tasks
        B = [None] * self.problem.num_tasks

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
            at, ato, bmt = _min2(
                at, ato, bmt, st, -id(task) if st == 0 else id(m))
        for t, d in St:
            st = max(self.RA[t.id], at_none - d)
            at, ato, bmt = _min2(at, ato, bmt, st, id(t))
        if self.L > len(self.platform) or len(Sm) < len(self.platform):
            at, ato, bmt = _min2(at, ato, bmt, at_none, -id(task))

        self.AT[task.id] = at
        self.ATO[task.id] = ato
        self.BM[task.id] = set(bmt)
        if not hasattr(self, "ATCS"):
            self.ATCS = [0] * self.problem.num_tasks
        self.ATCS[task.id] = 0
        for c in task.in_comms:
            t = c.from_task
            if not (self._placed[t.id] and id(self.PL_m(t)) in bmt) or id(t) in bmt:
                self.ATCS[task.id] += self.CT(c)

    @memo
    def calculate_PT(self, task):
        comms = sorted(task.out_comms,
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
            t_local = max(
                remote_ft,
                local_ft + local_delta,
                local_st + self._RT[c.to_task.id] + self.PT_r[c.to_task.id])
            t_remote = max(
                remote_ft,
                local_ft, comm_ft +
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
        self.PT_r[task.id] = remote_ft

    def est_ft(self, ti, tj):
        p0, p1 = self.BM[ti], self.BM[tj]
        cti, ctj = self.ATCS[ti], self.ATCS[tj]
        ft_m = inf
        if p0 & p1:
            sti, stj = _rsts(cti, ctj, self.AT[ti], self.AT[tj])
            ft_m = min(ft_m, self.merge_FT(ti, tj, sti, stj)[0])
        else:
            sti, stj = _rsts(cti, ctj, self.AT[ti], self.ATO[tj])
            ft_m = min(ft_m, self.merge_FT(ti, tj, sti, stj)[0])
        if (p0 & p1) and len(p1) == 1:
            ft_m = min(ft_m, self.split_FT(
                ti, tj, self.AT[ti], self.ATO[tj])[0])
        else:
            ft_m = min(ft_m,
                       self.split_FT(ti, tj, self.AT[ti], self.AT[tj])[0])
        return ft_m

    def merge_FT(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = max(st_j, ft_i) + self._RT[tj]
        ptl_i, ptl_ij, ptl_j, ptr_i, ptr_j, scti = self.merge_PT(ti, tj)
        # print(">>>", self._ti2t(ti), self._ti2t(tj), st_i, st_j, ptr_i, ptr_j)
        if ft_i + ptl_i - ptl_ij <= ft_j - self._RT[tj]:
            if ptl_ij:
                ft_ai = max(ft_j + ptl_ij, ft_i + ptr_i)
            else:
                ft_ai = ft_i + max(ptl_i, ptr_i)
            ft_aj = ft_j + ptl_j
        else:
            ft_ai = ft_i + max(ptl_i + self._RT[tj], ptr_i)
            ft_aj = ft_i + ptl_i + ptl_j + self._RT[tj] - ptl_ij
        ft_aj = max(ft_aj, ft_j + ptr_j)
        # ft_aj = max(ft_aj,
                    # ptr_j + min(self._RT[tj], scti) + max(ft_i, ft_j - scti))
        return max(ft_ai, ft_aj), ft_ai, ft_aj

    @memo
    def merge_PT(self, ti, tj, fts=None):
        ts = {}
        for c in self.problem.tasks[ti].out_comms:
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].out_comms:
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i, ft_j = fts or (0, self._RT[tj])
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        jc_started = False
        scti = 0
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        for tx, [ct_i, ct_j] in self._sort_succs_in_MP(ts):
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
                    if not jc_started and not ct_j:
                        scti += ct_i
                if ct_j:
                    r_j = max(r_j, fr)
                hlc, cst = _upd_cst_hlc(hlc, cst, ct_i, ct_j)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j, scti

    def split_FT(self, ti, tj, st_i, st_j):
        pti, ptj = self.split_PT(ti, tj)
        ft_ai = st_i + self._RT[ti] + pti
        ft_aj = st_j + self._RT[tj] + ptj
        # print(self._ti2t(ti), self._ti2t(tj),
        # ft_ai, ft_aj, st_i, st_j, pti, ptj)
        return max(ft_ai, ft_aj), ft_ai, ft_aj

    @memo
    def split_PT(self, ti, tj, fts=None):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = ctasks_i & ctasks_j
        ft_i, ft_j = fts or (self.RA[ti], self.RA[tj])
        dt_i, dt_j = ft_i, ft_j
        rft_i = ft_i + self.PT_r[ti]
        rft_j = ft_j + self.PT_r[tj]
        lft_i, lft_j = ft_i, ft_j

        # print("SPLIT_PT", self._ti2t(ti), self._ti2t(tj), ft_i, ft_j, cts)

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            # if self._RT[t.id] == 0:
                # continue
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            # print(t, ds_i, ds_j, self.AT[t.id], self.AT[ti], self.AT[tj])
            dt_i0, dt_j0 = dt_i, dt_j
            cst_i, cst_j = ft_i, ft_j
            for c in sorted(ds_i, key=lambda c: self._comm_st_in_SP(c, ti, ft_i)):
                if self.CT(c) > 0:
                    dt_i = max(dt_i, self._comm_st_in_SP(
                        c, ti, ft_i)) + self.CT(c)
                    cst_i = max(cst_i, dt_i)
                else:
                    cst_i = max(cst_i, self._comm_st_in_SP(c, ti, ft_i))
            for c in sorted(ds_j, key=lambda c: self._comm_st_in_SP(c, tj, ft_j)):
                if self.CT(c) > 0:
                    dt_j = max(dt_j, self._comm_st_in_SP(
                        c, tj, ft_j)) + self.CT(c)
                    cst_j = max(cst_j, dt_j)
                else:
                    cst_j = max(cst_j, self._comm_st_in_SP(c, tj, ft_j))
            # print(">>", cst_i, cst_j)
            if cst_i <= cst_j:
                dt_j = dt_j0
                rft_i = max(rft_i, cst_i + self.RP[t.id])
                lft_j = max(lft_j, cst_i + self._RT[t.id])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j, cst_j + self.RP[t.id])
                lft_i = max(lft_i, cst_j + self._RT[t.id])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        pti = max(rft_i, lft_i, ft_i +
                  sum(self._RT[t.id] for t in ctasks_i)) - ft_i
        ptj = max(rft_j, lft_j, ft_j +
                  sum(self._RT[t.id] for t in ctasks_j)) - ft_j
        # print("i", rft_i, lft_i, ft_i, ctasks_i)
        # print("j", rft_j, lft_j, ft_j, ctasks_j)
        return pti, ptj


class CAWSv1_1(CAWS):
    def _comm_st_in_SP(self, c, ti, ft_i):
        return ft_i + self.AT[c.from_task.id] - self.RA[ti]


class CAWSv1_1_1(CAWSv1_1):
    @memo
    def split_PT(self, ti, tj, fts=None):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = ctasks_i & ctasks_j
        ft_i, ft_j = fts or (self.RA[ti], self.RA[tj])
        dt_i, dt_j = ft_i, ft_j
        if self._ti2t(ti) in self._ti2t(tj).prevs():
            dt_i += self._CT[ti, tj]
        rft_i = ft_i + self.PT_r[ti]
        rft_j = ft_j + self.PT_r[tj]
        lft_i, lft_j = ft_i, ft_j

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            cst_i, cst_j = ft_i, ft_j
            for c in sorted(ds_i, key=lambda c: self._comm_st_in_SP(c, ti, ft_i)):
                if self.CT(c) > 0:
                    dt_i = max(dt_i, self._comm_st_in_SP(
                        c, ti, ft_i)) + self.CT(c)
                    cst_i = max(cst_i, dt_i)
                else:
                    cst_i = max(cst_i, self._comm_st_in_SP(c, ti, ft_i))
            for c in sorted(ds_j, key=lambda c: self._comm_st_in_SP(c, tj, ft_j)):
                if self.CT(c) > 0:
                    dt_j = max(dt_j, self._comm_st_in_SP(
                        c, tj, ft_j)) + self.CT(c)
                    cst_j = max(cst_j, dt_j)
                else:
                    cst_j = max(cst_j, self._comm_st_in_SP(c, tj, ft_j))
            if cst_i <= cst_j:
                dt_j = dt_j0
                rft_i = max(rft_i, cst_i + self.RP[t.id])
                lft_j = max(lft_j, cst_i + self._RT[t.id])
                deps_i -= self._cdeps[t.id]
                ctasks_i.remove(t)
            else:
                dt_i = dt_i0
                rft_j = max(rft_j, cst_j + self.RP[t.id])
                lft_i = max(lft_i, cst_j + self._RT[t.id])
                deps_j -= self._cdeps[t.id]
                ctasks_j.remove(t)

        pti = max(rft_i, lft_i, ft_i +
                  sum(self._RT[t.id] for t in ctasks_i)) - ft_i
        ptj = max(rft_j, lft_j, ft_j +
                  sum(self._RT[t.id] for t in ctasks_j)) - ft_j
        return pti, ptj


class CAWSv1_2(CAWSv1_1):
    @memo
    def split_PT(self, ti, tj, fts=None):
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = self._ctasks[ti] & self._ctasks[tj]
        ft_i, ft_j = fts or (self.RA[ti], self.RA[tj])
        dt_i, dt_j = ft_i, ft_j
        rft_i = ft_i + self.PT_r[ti]
        rft_j = ft_j + self.PT_r[tj]
        lft_i, lft_j = ft_i, ft_j

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            cst_i, cst_j = ft_i, ft_j
            for c in sorted(ds_i, key=lambda c: self._comm_st_in_SP(c, ti, ft_i)):
                if self.CT(c) > 0:
                    dt_i = max(dt_i, self._comm_st_in_SP(
                        c, ti, ft_i)) + self.CT(c)
                    cst_i = max(cst_i, dt_i)
                else:
                    cst_i = max(cst_i, self._comm_st_in_SP(c, ti, ft_i))
            for c in sorted(ds_j, key=lambda c: self._comm_st_in_SP(c, tj, ft_j)):
                if self.CT(c) > 0:
                    dt_j = max(dt_j, self._comm_st_in_SP(
                        c, tj, ft_j)) + self.CT(c)
                    cst_j = max(cst_j, dt_j)
                else:
                    cst_j = max(cst_j, self._comm_st_in_SP(c, tj, ft_j))
            # print(">>", cst_i, cst_j)
            if cst_i <= cst_j:
                dt_j = dt_j0
                rft_i = max(rft_i, cst_i + self.RP[t.id])
                lft_j = max(lft_j, cst_i + self._RT[t.id])
                deps_i -= self._cdeps[t.id]
            else:
                dt_i = dt_i0
                rft_j = max(rft_j, cst_j + self.RP[t.id])
                lft_i = max(lft_i, cst_j + self._RT[t.id])
                deps_j -= self._cdeps[t.id]

        ctasks_i = set(c.to_task for c in +deps_i)
        ctasks_j = set(c.to_task for c in +deps_j)
        pti = max(rft_i, lft_i, ft_i +
                  sum(self._RT[t.id] for t in ctasks_i)) - ft_i
        ptj = max(rft_j, lft_j, ft_j +
                  sum(self._RT[t.id] for t in ctasks_j)) - ft_j
        # print(self._ti2t(ti), self._ti2t(tj), pti, ptj, lft_i, lft_j, rft_i, rft_j, ft_i, ft_j)
        return pti, ptj


class CAWSv1_2_1(CAWSv1_2):
    def _comm_st_in_SP(self, c, ti, ft_i):
        return ft_i + self.RA[c.from_task.id] - self.RA[ti]



class CAWSv1_3(CAWSv1_2):
    def fitness(self, task, machine, comm_pls, st):
        all_ft = []
        task_ft = st + self._RT[task.id] + self.PT[task.id]
        wrk_ft = task_ft
        cur_ft = task_ft

        for t in self.problem.tasks:
            if self._placed[t.id] and \
                    any(not self._placed[c.to_task.id]
                        for c in t.out_comms
                        if c.to_task != task):
                st_t = self.ST(t)
                if self._is_successor(t, task):
                    ft_j = st + self._RT[task.id] + self.PT[task.id]
                    ft_a = ft_i = max(ft_j, self.FT(t) + self.PT[t.id])
                elif self.PL_m(t) != machine:
                    ft_a, ft_i, ft_j = self.split_FT(t.id, task.id, st_t, st)
                else:
                    ft_a, ft_i, ft_j = self.merge_FT(t.id, task.id, st_t, st)
                all_ft.append(ft_i)
                cur_ft = max(cur_ft, ft_j)
                wrk_ft = max(wrk_ft, ft_a)
        all_ft.append(cur_ft)
        return wrk_ft, task_ft, sorted(all_ft, reverse=True)


class CAWSv1_4(CAWSv1_2):
    @memo
    def merge_PT(self, ti, tj, fts=None):
        ts = {}
        for c in self.problem.tasks[ti].out_comms:
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].out_comms:
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i, ft_j = fts or (0, self._RT[tj])
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        jc_started = False
        scti = 0
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        for tx, [ct_i, ct_j] in self._sort_succs_in_MP(ts):
            d = sum(self._RT[t.id]
                    for t in (self._ctasks[tx.id] | {tx}) - ctasks)
            fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
            if ct_j > 0:
                frt = cst + ct_j + max(ct_i - hlc, 0)
            elif ct_i > hlc:
                frt = cst + ct_i - hlc
            else:
                frt = ft_j + ct_i - hlc
            fr = frt + self.RP[tx.id]
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
                    if not jc_started and not ct_j:
                        scti += ct_i
                if ct_j:
                    r_j = max(r_j, fr)
                cst += ct_j + max(ct_i - hlc, 0)
                hlc = max(hlc - ct_i, 0)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j, scti


class CAWSv1_5(CAWSv1_2):
    def est_ft(self, ti, tj):
        sti, stj = _rsts(self.ATCS[ti], self.ATCS[tj],
                         self.AT[ti], self.AT[tj])
        ft_0 = self.merge_FT(ti, tj, sti, stj)[0]
        ft_1 = self.split_FT(ti, tj, self.AT[ti], self.AT[tj])[0]
        # print(self._ti2t(ti), self._ti2t(tj), ft_0, ft_1)
        return min(ft_0, ft_1)


class CAWSv1_6(CAWSv1_5):
    def update_AT_and_PT(self):
        for t in self.toporder:
            if self._placed[t.id]:
                self.AT[t.id] = self.ST(t)
            else:
                self.calculate_AT(t)
            self.RA[t.id] = self.AT[t.id] + self._RT[t.id]
        for t in reversed(self.toporder):
            if not self._placed[t.id]:
                self.calculate_PT(t)
                self.RP[t.id] = self.PT[t.id] + self._RT[t.id]


class CAWSv1_7(CAWSv1_5):
    def calculate_ATs_and_PTs(self):
        for t in self.toporder:
            self.calculate_AT(t)
            self.RA[t.id] = self.AT[t.id] + self._RT[t.id]
        for t in reversed(self.toporder):
            self.calculate_PT(t)
            self.RP[t.id] = self.PT[t.id] + self._RT[t.id]

    def update_ATs(self, task):
        self.AT[task.id] = self.ST(task)
        self.RA[task.id] = self.PL_m(task).earliest_idle_time_for_communication(
            self.bandwidth, COMM_OUTPUT, self.FT(task))
        for t in task.succs():
            if self._placed[t.id]:
                m = self.PL_m(t)
                self.RA[t.id] = m.earliest_idle_time_for_communication(
                    self.bandwidth, COMM_OUTPUT, self.FT(t))
            else:
                self.calculate_AT(t)
                self.RA[t.id] = self.AT[t.id] + self._RT[t.id]

    def sort_tasks(self):
        self._prepare_arrays()
        self.calculate_ATs_and_PTs()
        self.ready_tasks = set(
            t for t in self.problem.tasks if not t.in_degree)
        self.rids = [t.in_degree for t in self.problem.tasks]
        while self.ready_tasks:
            task = self.select_task()
            yield task
            self.ready_tasks.remove(task)
            self._placed[task.id] = True
            self.update_ATs(task)
            for t in task.succs():
                self.rids[t.id] -= 1
                if not self.rids[t.id]:
                    self.ready_tasks.add(t)

    def select_task(self):
        self._edges = []
        for tx in range(self.problem.num_tasks):
            for ty in range(tx + 1, self.problem.num_tasks):
                if not self._placed[tx] and not self._placed[ty] and \
                        not self.rids[tx] and not self.rids[ty]:
                    ftx = self.est_ft(tx, ty)
                    fty = self.est_ft(ty, tx)
                    if ftx < fty:
                        self._edges.append((tx, ty, fty))
                    elif ftx > fty:
                        self._edges.append((ty, tx, ftx))
        task = max(self._ready_graph(), key=lambda t: self.RP[t.id])
        return task


class CAWSv1_8(CAWSv1_5):
    @memo
    def split_PT(self, ti, tj):
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = self._ctasks[ti] & self._ctasks[tj]
        ft_i = self.AT[ti] + self._RT[ti]
        ft_j = self.AT[tj] + self._RT[tj]
        l_i, l_j = ft_i, ft_j
        r_i = ft_i + self.PT_r[ti]
        r_j = ft_j + self.PT_r[tj]
        st_i, st_j = self.RA[ti], self.RA[tj]

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            if self._RT[t.id] == 0:
                continue
            ds_i = set(c for c in t.in_comms if c in +
                       (deps_i - deps_j))
            ds_j = set(c for c in t.in_comms if c in +
                       (deps_j - deps_i))
            if not ds_i or not ds_j:
                continue
            si, sj = st_i, st_j
            cst_i, cst_j = ft_i, ft_j
            for c in sorted(ds_i, key=lambda c: self.RA[c.from_task.id]):
                if self.CT(c) > 0:
                    si = max(si, self.RA[c.from_task.id]) + self.CT(c)
                    cst_i = max(cst_i, si)
                else:
                    cst_i = max(
                        cst_i, self.AT[c.from_task.id] + self._RT[c.from_task.id])
            for c in sorted(ds_j, key=lambda c: self.RA[c.from_task.id]):
                if self.CT(c) > 0:
                    sj = max(sj, self.RA[c.from_task.id]) + self.CT(c)
                    cst_j = max(cst_j, sj)
                else:
                    cst_j = max(
                        cst_j, self.AT[c.from_task.id] + self._RT[c.from_task.id])
            # print(">>", cst_i, cst_j, si, sj, ds_i, ds_j, r_i, r_j)
            if cst_i <= cst_j:
                st_i = si
                r_i = max(r_i, cst_i + self.RP[t.id])
                l_j = max(l_j, cst_i + self._RT[t.id])
                deps_i -= self._cdeps[t.id]
            else:
                st_j = sj
                r_j = max(r_j, cst_j + self.RP[t.id])
                l_i = max(l_i, cst_j + self._RT[t.id])
                deps_j -= self._cdeps[t.id]

        l_i = max(l_i,
                  ft_i + sum(self._RT[t]
                             for t in set(c.to_task.id
                                          for c in +deps_i)))
        l_j = max(l_j,
                  ft_j + sum(self._RT[t]
                             for t in set(c.to_task.id
                                          for c in +deps_j)))
        pti = max(r_i, l_i) - ft_i
        ptj = max(r_j, l_j) - ft_j
        # print(self._ti2t(ti), self._ti2t(tj), pti, ptj, l_i, l_j, r_i, r_j, ft_i, ft_j)
        return pti, ptj


class CAWSv1_9(CAWSv1_5):
    @memo
    def split_PT(self, ti, tj):
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = self._ctasks[ti] & self._ctasks[tj]
        ft_i = self.AT[ti] + self._RT[ti]
        ft_j = self.AT[tj] + self._RT[tj]
        l_i, l_j = ft_i, ft_j
        r_i = ft_i + self.PT_r[ti]
        r_j = ft_j + self.PT_r[tj]
        st_i, st_j = self.RA[ti], self.RA[tj]

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            if self._RT[t.id] == 0:
                continue
            ds_i = set(c for c in t.in_comms if c in +
                       (deps_i - deps_j))
            ds_j = set(c for c in t.in_comms if c in +
                       (deps_j - deps_i))
            if not ds_i or not ds_j:
                continue
            si, sj = st_i, st_j
            cst_i, cst_j = ft_i, ft_j
            for c in sorted(ds_i, key=lambda c: self.RA[c.from_task.id]):
                if self.CT(c) > 0:
                    si = max(si, self.RA[c.from_task.id]) + self.CT(c)
                    cst_i = max(cst_i, si)
                else:
                    cst_i = max(
                        cst_i, self.AT[c.from_task.id] + self._RT[c.from_task.id])
                cst_j = max(
                    cst_j, self.AT[c.from_task.id] + self._RT[c.from_task.id])
            for c in sorted(ds_j, key=lambda c: self.RA[c.from_task.id]):
                if self.CT(c) > 0:
                    sj = max(sj, self.RA[c.from_task.id]) + self.CT(c)
                    cst_j = max(cst_j, sj)
                else:
                    cst_j = max(
                        cst_j, self.AT[c.from_task.id] + self._RT[c.from_task.id])
                cst_i = max(
                    cst_i, self.AT[c.from_task.id] + self._RT[c.from_task.id])
            # print(">>", cst_i, cst_j, si, sj, ds_i, ds_j, r_i, r_j)
            if cst_i <= cst_j:
                st_i = si
                r_i = max(r_i, cst_i + self.RP[t.id])
                l_j = max(l_j, cst_i + self._RT[t.id])
                deps_i -= self._cdeps[t.id]
            else:
                st_j = sj
                r_j = max(r_j, cst_j + self.RP[t.id])
                l_i = max(l_i, cst_j + self._RT[t.id])
                deps_j -= self._cdeps[t.id]

        l_i = max(l_i,
                  ft_i + sum(self._RT[t]
                             for t in set(c.to_task.id
                                          for c in +deps_i)))
        l_j = max(l_j,
                  ft_j + sum(self._RT[t]
                             for t in set(c.to_task.id
                                          for c in +deps_j)))
        pti = max(r_i, l_i) - ft_i
        ptj = max(r_j, l_j) - ft_j
        # print(self._ti2t(ti), self._ti2t(tj), pti, ptj, l_i, l_j, r_i, r_j, ft_i, ft_j)
        return pti, ptj


class CAWSv1_10(CAWSv1_5):
    @memo
    def split_PT(self, ti, tj):
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = self._ctasks[ti] & self._ctasks[tj]
        ft_i = self.AT[ti] + self._RT[ti]
        ft_j = self.AT[tj] + self._RT[tj]
        l_i, l_j = ft_i, ft_j
        r_i = ft_i + self.PT_r[ti]
        r_j = ft_j + self.PT_r[tj]
        st_i, st_j = self.RA[ti], self.RA[tj]

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            if self._RT[t.id] == 0:
                continue
            ds_i = set(c for c in t.in_comms if c in +
                       (deps_i - deps_j))
            ds_j = set(c for c in t.in_comms if c in +
                       (deps_j - deps_i))
            if not ds_i or not ds_j:
                continue
            si, sj = st_i, st_j
            cst_i, cst_j = ft_i, ft_j
            for c in sorted(ds_i, key=lambda c: self.RA[c.from_task.id]):
                if self.CT(c) > 0:
                    si = max(si, self.RA[c.from_task.id]) + self.CT(c)
                    cst_i = max(cst_i, si)
                else:
                    cst_i = max(
                        cst_i, self.AT[c.from_task.id] + self._RT[c.from_task.id])
                cst_j = max(
                    cst_j, self.AT[c.from_task.id] + self._RT[c.from_task.id])
            for c in sorted(ds_j, key=lambda c: self.RA[c.from_task.id]):
                if self.CT(c) > 0:
                    sj = max(sj, self.RA[c.from_task.id]) + self.CT(c)
                    cst_j = max(cst_j, sj)
                else:
                    cst_j = max(
                        cst_j, self.AT[c.from_task.id] + self._RT[c.from_task.id])
                cst_i = max(
                    cst_i, self.AT[c.from_task.id] + self._RT[c.from_task.id])
            # print(">>", cst_i, cst_j, si, sj, ds_i, ds_j, r_i, r_j)
            if cst_i <= cst_j:
                st_i = si
                r_i = max(r_i, cst_i + self.RP[t.id])
                l_j = max(l_j, cst_i + self.RP[t.id])
                deps_i -= self._cdeps[t.id]
            else:
                st_j = sj
                r_j = max(r_j, cst_j + self.RP[t.id])
                l_i = max(l_i, cst_j + self.RP[t.id])
                deps_j -= self._cdeps[t.id]

        l_i = max(l_i,
                  ft_i + sum(self._RT[t]
                             for t in set(c.to_task.id
                                          for c in +deps_i)))
        l_j = max(l_j,
                  ft_j + sum(self._RT[t]
                             for t in set(c.to_task.id
                                          for c in +deps_j)))
        pti = max(r_i, l_i) - ft_i
        ptj = max(r_j, l_j) - ft_j
        # print(self._ti2t(ti), self._ti2t(tj), pti, ptj, l_i, l_j, r_i, r_j, ft_i, ft_j)
        return pti, ptj


class CAWSv1_11(CAWSv1_5):
    @memo
    def split_PT(self, ti, tj):
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = self._ctasks[ti] & self._ctasks[tj]
        ft_i = self.AT[ti] + self._RT[ti]
        ft_j = self.AT[tj] + self._RT[tj]
        l_i, l_j = ft_i, ft_j
        r_i = ft_i + self.PT_r[ti]
        r_j = ft_j + self.PT_r[tj]
        st_i, st_j = self.RA[ti], self.RA[tj]

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            if self._RT[t.id] == 0:
                continue
            ds_i = set(c for c in t.in_comms if c in +
                       (deps_i - deps_j))
            ds_j = set(c for c in t.in_comms if c in +
                       (deps_j - deps_i))
            if not ds_i or not ds_j:
                continue
            si, sj = st_i, st_j
            cst_i, cst_j = ft_i, ft_j
            for c in sorted(ds_i, key=lambda c: self.RA[c.from_task.id]):
                if self.CT(c) > 0:
                    si = max(si, self.RA[c.from_task.id]) + self.CT(c)
                    cst_i = max(cst_i, si)
                else:
                    cst_i = max(
                        cst_i, self.AT[c.from_task.id] + self._RT[c.from_task.id])
            for c in sorted(ds_j, key=lambda c: self.RA[c.from_task.id]):
                if self.CT(c) > 0:
                    sj = max(sj, self.RA[c.from_task.id]) + self.CT(c)
                    cst_j = max(cst_j, sj)
                else:
                    cst_j = max(
                        cst_j, self.AT[c.from_task.id] + self._RT[c.from_task.id])
            # print(">>", cst_i, cst_j, si, sj, ds_i, ds_j, r_i, r_j)
            if cst_i <= cst_j:
                st_i = si
                r_i = max(r_i, cst_i + self.RP[t.id])
                l_j = max(l_j, cst_i + self.RP[t.id])
                deps_i -= self._cdeps[t.id]
            else:
                st_j = sj
                r_j = max(r_j, cst_j + self.RP[t.id])
                l_i = max(l_i, cst_j + self.RP[t.id])
                deps_j -= self._cdeps[t.id]

        l_i = max(l_i,
                  ft_i + sum(self._RT[t]
                             for t in set(c.to_task.id
                                          for c in +deps_i)))
        l_j = max(l_j,
                  ft_j + sum(self._RT[t]
                             for t in set(c.to_task.id
                                          for c in +deps_j)))
        pti = max(r_i, l_i) - ft_i
        ptj = max(r_j, l_j) - ft_j
        # print(self._ti2t(ti), self._ti2t(tj), pti, ptj, l_i, l_j, r_i, r_j, ft_i, ft_j)
        return pti, ptj


class CAWSv1_12(CAWSv1_11):
    @memo
    def calculate_PT(self, task):
        comms = sorted(task.out_comms,
                       key=lambda c: -self.RP[c.to_task.id])

        self._cdeps[task.id] = Counter()
        self._ctasks[task.id] = set()
        pt_l = pt_r = 0
        st_c = st_t = 0
        for c in comms:
            t = c.to_task
            l_d = sum(self._RT[_t.id] for _t in (
                self._ctasks[t.id] | {t}) - self._ctasks[task.id])
            t_l = max(pt_l + l_d, st_t + self._RT[t.id] + self.PT_r[t.id])
            t_r = st_c + self.CT(c) + self.RP[t.id]
            if (t_l, st_t) <= (t_r, t_r - self.RP[t.id]):
                st_t += self._RT[t.id]
                pt_l += l_d
                if self.PT_r[t.id] > 0:
                    pt_r = max(pt_r, st_t + self.PT_r[t.id])
                self._cdeps[task.id][c] += 1
                self._cdeps[task.id] += self._cdeps[c.to_task.id]
                self._ctasks[task.id].add(c.to_task)
                self._ctasks[task.id].update(self._ctasks[c.to_task.id])
            else:
                st_c += self.CT(c)
                pt_r = max(pt_r, t_r)
        self.PT_l[task.id] = pt_l
        self.PT_r[task.id] = pt_r
        self.PT[task.id] = max(pt_l, pt_r)

    @memo
    def merge_PT(self, ti, tj, fts=None):
        ts = {}
        for c in self.problem.tasks[ti].out_comms:
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].out_comms:
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i, ft_j = fts or (0, self._RT[tj])
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        jc_started = False
        scti = 0
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        for tx, [ct_i, ct_j] in self._sort_succs_in_MP(ts):
            d = sum(self._RT[t.id]
                    for t in (self._ctasks[tx.id] | {tx}) - ctasks)
            fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
            if ct_j > 0:
                frt = cst + ct_j + max(ct_i - hlc, 0)
            elif ct_i > hlc:
                frt = cst + ct_i - hlc
            else:
                frt = ft_j + ct_i - hlc
            fr = frt + self.RP[tx.id]
            # print(self._ti2t(ti), self._ti2t(tj), tx, fl, fr)
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
                    # print("Ri")
                    r_i = max(r_i, fr)
                    if not jc_started and not ct_j:
                        scti += ct_i
                if ct_j:
                    # print("Rj")
                    r_j = max(r_j, fr)
                cst += ct_j + max(ct_i - hlc, 0)
                hlc = max(hlc - ct_i, 0)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        # print(self._ti2t(ti), self._ti2t(tj), ctasks_i, ctasks_j, r_i - ft_i, r_j - ft_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j, scti


class CAWSv1_13(CAWSv1_12):
    @memo
    def calculate_PT(self, task):
        comms = sorted(task.out_comms,
                       key=lambda c: -self.RP[c.to_task.id])
        self._cdeps[task.id] = set()
        self._ctasks[task.id] = set()
        pt_l = pt_r = 0
        st_c = st_t = 0
        for c in comms:
            t = c.to_task
            l_d = sum(self._RT[_t.id] for _t in (
                self._ctasks[t.id] | {t}) - self._ctasks[task.id])
            t_l = max(pt_l + l_d, st_t + self._RT[t.id] + self.PT_r[t.id])
            t_r = st_c + self.CT(c) + self.RP[t.id]
            if (t_l, st_t) <= (t_r, t_r - self.RP[t.id]):
                st_t += self._RT[t.id]
                pt_l += l_d
                if self.PT_r[t.id] > 0:
                    pt_r = max(pt_r, st_t + self.PT_r[t.id])
                self._cdeps[task.id].add(c)
                self._cdeps[task.id].update(self._cdeps[c.to_task.id])
                self._ctasks[task.id].add(c.to_task)
                self._ctasks[task.id].update(self._ctasks[c.to_task.id])
            else:
                st_c += self.CT(c)
                pt_r = max(pt_r, t_r)
        self.PT_l[task.id] = pt_l
        self.PT_r[task.id] = pt_r
        self.PT[task.id] = max(pt_l, pt_r)

    @memo
    def split_PT(self, ti, tj):
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = self._ctasks[ti] & self._ctasks[tj]
        ft_i = self.AT[ti] + self._RT[ti]
        ft_j = self.AT[tj] + self._RT[tj]
        l_i, l_j = ft_i, ft_j
        r_i = ft_i + self.PT_r[ti]
        r_j = ft_j + self.PT_r[tj]
        st_i, st_j = self.RA[ti], self.RA[tj]

        # print(">>", self._ti2t(ti), self._ti2t(tj), "!", self._cdeps[ti], self._cdeps[tj], cts)

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            if self._RT[t.id] == 0:
                continue
            ds_i = [c for c in t.in_comms if c in deps_i and c not in deps_j]
            ds_j = [c for c in t.in_comms if c in deps_j and c not in deps_i]
            if not ds_i or not ds_j:
                continue
            si, sj = st_i, st_j
            cst_i, cst_j = ft_i, ft_j
            for c in sorted(ds_i, key=lambda c: self.RA[c.from_task.id]):
                if self.CT(c) > 0:
                    si = max(si, self.RA[c.from_task.id]) + self.CT(c)
                    cst_i = max(cst_i, si)
                else:
                    cst_i = max(
                        cst_i, self.AT[c.from_task.id] + self._RT[c.from_task.id])
            for c in sorted(ds_j, key=lambda c: self.RA[c.from_task.id]):
                if self.CT(c) > 0:
                    sj = max(sj, self.RA[c.from_task.id]) + self.CT(c)
                    cst_j = max(cst_j, sj)
                else:
                    cst_j = max(
                        cst_j, self.AT[c.from_task.id] + self._RT[c.from_task.id])
            # print(">>", cst_i, cst_j, si, sj, ds_i, ds_j, r_i, r_j)
            if cst_i <= cst_j:
                st_i = si
                r_i = max(r_i, cst_i + self.RP[t.id])
                l_j = max(l_j, cst_i + self.RP[t.id])
                deps_i -= self._cdeps[t.id]
            else:
                st_j = sj
                r_j = max(r_j, cst_j + self.RP[t.id])
                l_i = max(l_i, cst_j + self.RP[t.id])
                deps_j -= self._cdeps[t.id]

        # print(">>>", set(c.to_task for c in deps_i), set(c.to_task for c in deps_j), r_i, r_j, r_i, l_j)

        l_i = max(l_i,
                  ft_i + sum(self._RT[t.id] for t in set(c.to_task for c in deps_i)))
        l_j = max(l_j,
                  ft_j + sum(self._RT[t.id] for t in set(c.to_task for c in deps_j)))
        pti = max(r_i, l_i) - ft_i
        ptj = max(r_j, l_j) - ft_j
        # print(self._ti2t(ti), self._ti2t(tj), pti, ptj, l_i, l_j, r_i, r_j, ft_i, ft_j)
        return pti, ptj


class CAWSv1_14(CAWSv1_12):
    def _ready_graph(self):
        self._edges.sort(key=lambda i: i[-1], reverse=True)
        rg = nx.DiGraph()
        for t in self.ready_tasks:
            if t.id not in rg:
                rg.add_node(t.id)
        for tx, ty, w in self._edges:
            rg.add_edge(tx, ty)
            try:
                nx.find_cycle(rg, source=tx)
                rg.remove_edge(tx, ty)
            except nx.exception.NetworkXNoCycle:
                pass
        for t in rg.nodes():
            if rg.in_degree(t) == 0:
                yield self.problem.tasks[t]


class CAWSv1_15(CAWSv1_13):
    def select_task(self):
        self.update_AT_and_PT()
        task = max(self.ready_tasks, key=lambda t: self.RP[t.id])
        # print(task)
        return task


class CAWSv1_16(CAWSv1_13):
    def sort_tasks(self):
        self._prepare_arrays()
        self.update_AT_and_PT()
        for tx in range(self.problem.num_tasks):
            for ty in range(self.problem.num_tasks):
                if tx != ty:
                    self.merge_PT(tx, ty)
                    self.split_PT(tx, ty)
        self.ready_tasks = set(
            t for t in self.problem.tasks if not t.in_degree)
        self.rids = [t.in_degree for t in self.problem.tasks]
        while self.ready_tasks:
            task = max(self.ready_tasks, key=lambda t: self.RP[t.id])
            yield task
            self.ready_tasks.remove(task)
            self._placed[task.id] = True
            for t in task.succs():
                self.rids[t.id] -= 1
                if not self.rids[t.id]:
                    self.ready_tasks.add(t)
            self.update_AT_and_PT()


class CAWSv1_16_1(CAWSv1_12):
    def sort_tasks(self):
        self._prepare_arrays()
        self.update_AT_and_PT()
        for tx in range(self.problem.num_tasks):
            for ty in range(self.problem.num_tasks):
                if tx != ty:
                    self.merge_PT(tx, ty)
                    self.split_PT(tx, ty)
        self.ready_tasks = set(
            t for t in self.problem.tasks if not t.in_degree)
        self.rids = [t.in_degree for t in self.problem.tasks]
        while self.ready_tasks:
            task = max(self.ready_tasks, key=lambda t: self.RP[t.id])
            yield task
            self.ready_tasks.remove(task)
            self._placed[task.id] = True
            for t in task.succs():
                self.rids[t.id] -= 1
                if not self.rids[t.id]:
                    self.ready_tasks.add(t)
            self.update_AT_and_PT()


class CAWSv1_16_2(CAWSv1_12):
    def sort_tasks(self):
        self._prepare_arrays()
        self.update_AT_and_PT()
        # for tx in range(self.problem.num_tasks):
            # for ty in range(self.problem.num_tasks):
                # if tx != ty:
                    # self.merge_PT(tx, ty)
                    # self.split_PT(tx, ty)
        self.ready_tasks = set(
            t for t in self.problem.tasks if not t.in_degree)
        self.rids = [t.in_degree for t in self.problem.tasks]
        while self.ready_tasks:
            task = max(self.ready_tasks, key=lambda t: self.RP[t.id])
            for t in self.problem.tasks:
                if self._placed[t.id] and \
                        any(not self._placed[c.to_task.id]
                            for c in t.out_comms
                            if c.to_task != task):
                    self.merge_PT(t.id, task.id)
                    self.split_PT(t.id, task.id)
            yield task
            self.ready_tasks.remove(task)
            self._placed[task.id] = True
            for t in task.succs():
                self.rids[t.id] -= 1
                if not self.rids[t.id]:
                    self.ready_tasks.add(t)
            # self.update_AT_and_PT()


class CAWSv1_17(CAWSv1_16):
    @memo
    def merge_PT(self, ti, tj):
        ts = {}
        for c in self.problem.tasks[ti].out_comms:
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].out_comms:
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i, ft_j = self.RA[ti], self.RA[tj]
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        jc_started = False
        scti = 0
        ctasks = set()
        ctasks_i = set()
        ctasks_j = set()
        for tx, [ct_i, ct_j] in self._sort_succs_in_MP(ts):
            d = sum(self._RT[t.id]
                    for t in (self._ctasks[tx.id] | {tx}) - ctasks)
            fl = max(lft + d, tst + self._RT[tx.id] + self.PT_r[tx.id])
            if ct_j > 0:
                frt = cst + ct_j + max(ct_i - hlc, 0)
            elif ct_i > hlc:
                frt = cst + ct_i - hlc
            else:
                frt = ft_j + ct_i - hlc
            fr = frt + self.RP[tx.id]
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
                    if not jc_started and not ct_j:
                        scti += ct_i
                if ct_j:
                    r_j = max(r_j, fr)
                cst += ct_j + max(ct_i - hlc, 0)
                hlc = max(hlc - ct_i, 0)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j, scti


class CAWS_r(CAWS):
    def merge_FT(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = max(st_j, ft_i) + self._RT[tj]
        ptl_i, ptl_ij, ptl_j, ptr_i, ptr_j, scti = self.merge_PT(
            ti, tj, (ft_i, ft_j))
        if ft_i + ptl_i - ptl_ij <= ft_j - self._RT[tj]:
            if ptl_ij:
                ft_ai = max(ft_j + ptl_ij, ft_i + ptr_i)
            else:
                ft_ai = ft_i + max(ptl_i, ptr_i)
            ft_aj = ft_j + ptl_j
        else:
            ft_ai = ft_i + max(ptl_i + self._RT[tj], ptr_i)
            ft_aj = ft_i + ptl_i + ptl_j + self._RT[tj] - ptl_ij
        ft_aj = max(ft_aj,
                    ptr_j + min(self._RT[tj], scti) + max(ft_i, ft_j - scti))
        # print(self._ti2t(ti), self._ti2t(tj), ft_aj, st_i, st_j,
        # ft_i, ft_j, ptl_i, ptl_j, ptl_ij, ptr_i, ptr_j, scti)
        return max(ft_ai, ft_aj), ft_ai, ft_aj

    def merge_PT(self, ti, tj, fts=None):
        ts = {}
        mi = self.PL_m(self._ti2t(ti)) if self._placed[ti] else None
        for c in self.problem.tasks[ti].out_comms:
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][0] += self.CT(c)

        for c in self.problem.tasks[tj].out_comms:
            if c.to_task not in ts:
                ts[c.to_task] = [0, 0]
            ts[c.to_task][1] += self.CT(c)

        ft_i, ft_j = fts or (0, self._RT[tj])
        r_i, r_j = ft_i, ft_j
        tst, cst, lft = ft_j, ft_j, ft_j
        hlc = ft_j - ft_i
        jc_started = False
        scti = 0
        ctasks_i = set()
        ctasks_j = set()
        for tx, [ct_i, ct_j] in self._sort_succs_in_MP(ts):
            d = 0
            for _t in self._ctasks[tx.id] | {tx}:
                if _t not in ctasks_i and _t not in ctasks_j and \
                        (not self._placed[_t.id] or self.FT(_t) > ft_j):
                    d += self._RT[_t.id]
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
            # print("IN", self._ti2t(ti), self._ti2t(tj), tx, fl, fr)
            if self._placed[tx.id]:
                if self.PL_m(tx) == mi:
                    tst = max(tst, self.FT(tx))
                    lft += d
                    ctasks_i.add(tx)
                    ctasks_i.update(self._ctasks[tx.id])
                    if self.PT_r[tx.id] > 0:
                        r_i = max(r_i, self.FT(tx) + self.PT_r[tx.id])
                else:
                    r_i = max(r_i, self.FT(tx) + self.PT[tx.id])
                    if not jc_started and not ct_j:
                        scti += ct_i
                    hlc, cst = _upd_cst_hlc(hlc, cst, ct_i, ct_j)
            else:
                if (fl, tst) <= (fr, fr - self.RP[tx.id]):
                    tst += self._RT[tx.id]
                    lft += d
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
                        if not jc_started and not ct_j:
                            scti += ct_i
                    if ct_j:
                        r_j = max(r_j, fr)
                    hlc, cst = _upd_cst_hlc(hlc, cst, ct_i, ct_j)
        pt_l_i = sum(self._RT[t.id] for t in ctasks_i)
        pt_l_ij = sum(self._RT[t.id] for t in ctasks_i & ctasks_j)
        pt_l_j = sum(self._RT[t.id] for t in ctasks_j)
        return pt_l_i, pt_l_ij, pt_l_j, r_i - ft_i, r_j - ft_j, scti

    def split_FT(self, ti, tj, st_i, st_j):
        ft_i = st_i + self._RT[ti]
        ft_j = st_j + self._RT[tj]
        pti, ptj = self.split_PT(ti, tj, (ft_i, ft_j))
        ft_ai = ft_i + pti
        ft_aj = ft_j + ptj
        # print(self._ti2t(ti), self._ti2t(tj),
        # ft_i, ft_j, ft_ai, ft_aj, pti, ptj)
        return max(ft_ai, ft_aj), ft_ai, ft_aj

    def split_PT(self, ti, tj, fts=None):
        ctasks_i = copy(self._ctasks[ti])
        ctasks_j = copy(self._ctasks[tj])
        deps_i = copy(self._cdeps[ti])
        deps_j = copy(self._cdeps[tj])
        cts = ctasks_i & ctasks_j
        ft_i, ft_j = fts or (self.RA[ti], self.RA[tj])
        dt_i, dt_j = ft_i, ft_j
        m_i = self.PL_m(self._ti2t(ti)) if self._placed[ti] else None
        # if self._placed[ti]:
        # dt_i = max(dt_i, m_i.earliest_idle_time_for_communication(
        # self.bandwidth, COMM_OUTPUT, self.FT(self._ti2t(ti))))
        for c in self._ti2t(tj).in_comms:
            if c.from_task.id == ti or \
                    (self._placed[c.from_task.id] and self.PL_m(c.from_task) == m_i):
                if self.FT(c.from_task) >= ft_i:
                    dt_i += self.CT(c)
                # if self.CT(c) > 0:
                    # dt_i = ft_j - self._RT[tj]
                    # break
        rft_i = ft_i + self.PT_r[ti]
        rft_j = ft_j + self.PT_r[tj]
        lft_i, lft_j = ft_i, ft_j
        # print(dt_i, dt_j)

        for t in sorted(cts, key=lambda _t: self.RP[_t.id], reverse=True):
            ds_i = Counter(c for c in +deps_i if c.to_task ==
                           t and c not in deps_j)
            ds_j = Counter(c for c in +deps_j if c.to_task ==
                           t and c not in deps_i)
            if not ds_i or not ds_j:
                continue
            dt_i0, dt_j0 = dt_i, dt_j
            for c in sorted(ds_i, key=lambda c: self._comm_st_in_SP(c, ti, ft_i)):
                dt_i = max(dt_i, self._comm_st_in_SP(c, ti, ft_i)) + self.CT(c)
                # print("i", dt_i)
            for c in sorted(ds_j, key=lambda c: self._comm_st_in_SP(c, tj, ft_j)):
                dt_j = max(dt_j, self._comm_st_in_SP(c, tj, ft_j)) + self.CT(c)
                # print("j", dt_j, c, self._comm_st_in_SP(c, tj, ft_j))
            # print("IN", t, ds_i, ds_j, dt_i, dt_j)
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
        return pti, ptj
