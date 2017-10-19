from MrWSI.core.problem import Task
from MrWSI.core.platform import Machine, COMM_INPUT, COMM_OUTPUT
from .base import Heuristic
from .sorting import memo, memo2, memo_clean, memo_delete, tryFT
import numpy as np


class CSort(Heuristic):
    def sort_tasks(self):
        self.critical_parent = {}
        self.best_pl_gauss = {}
        self.rem_in_deps = {t: t.in_degree for t in self.problem.tasks}
        self.ready_tasks = set(t for t in self.problem.tasks
                               if not t.in_degree)
        task = None
        while self.ready_tasks:
            self.prepare_selection(task)
            task = self.elect_task()
            self.ready_tasks.remove(task)
            for succ in task.succs():
                self.rem_in_deps[succ] -= 1
                if not self.rem_in_deps[succ]:
                    self.ready_tasks.add(succ)
            # print("Selected", task)
            yield task

    def prepare_selection(self, task):
        if task:
            memo_clean(self.rt_before)
            memo_clean(self.rt_after)
            memo_clean(self.ft_est)
            memo_clean(self.at_est)
            memo_clean(self.should_place_before)

    def elect_task(self):
        dominated_count = {t: 0 for t in self.ready_tasks}
        for task in self.ready_tasks:
            for t in self.ready_tasks:
                if t is not task and self.should_place_before(t, task):
                    dominated_count[task] += 1
        min_dc = min(dominated_count.values())
        # print(dominated_count)
        # print([(task, self.default_rank(task)) for task in self.ready_tasks])
        return max(
            [t for t, c in dominated_count.items() if c == min_dc],
            key=self.default_rank)

    def may_conflict(self, p0, p1):
        return len(p0) == 1 and len(p1) == 1 and min(p0) and min(p0) in p1

    @memo2
    def should_place_before(self, tx, ty):
        t0, t0a, p0 = self.rt_before(tx)
        x0 = self.RT(tx)
        y0, y0a = self.rt_after(tx)
        t1, t1a, p1 = self.rt_before(ty)
        x1 = self.RT(ty)
        y1, y1a = self.rt_after(ty)
        if self.may_conflict(p0, p1):
            # print(tx, t0, t0a, x0, y0, y0a)
            s0 = self.span_est_2(tx, ty, t0, t0a, x0, y0,
                                 y0a, t1, t1a, x1, y1, y1a)
            s1 = self.span_est_2(tx, ty, t1, t1a, x1, y1,
                                 y1a, t0, t0a, x0, y0, y0a)
            # print(tx, t0, t0a, x0, y0, y0a)
            # print(ty, t1, t1a, x1, y1, y1a)
            # print(p0, p1)
            # print(s0, s1, self.default_rank(tx), self.default_rank(ty), "\n")
            # print(tx, ty, s0, s1)
            return s0 < s1
        else:
            return False

    def span_est_2(self, tx, ty, t0, t0a, x0, y0, y0a, t1, t1a, x1, y1, y1a):
        return min(
            max(t0 + x0 + y0, t1a + x1 + y1),
            max(t0 + x0 + y0, max(t1, t0 + x0) + x1 + y1a),
            max(t0 + x0 + y0a, max(t1, t0 + x0) + x1 + y1),
            max(t1, t0 + x0 + y0) + x1 + y1)

    def default_rank(self, task):
        return self.at_est(task)

    @memo
    def rt_before(self, task):
        pls = set()
        for c in task.communications(COMM_INPUT):
            if self.is_placed(c.from_task):
                m = self.PL_m(c.from_task)
                if m not in pls:
                    pls.add(m)
            else:
                pls.add(c.from_task)
        comms = self.sorted_in_comms(task)
        fts = [(p, self.start_time_est(task, comms, p)) for p in pls]
        fts.append((None, self.start_time_est(task, comms, None)))
        ft_bst = min(ft for _, ft in fts)
        pl_bst = set(p for p, ft in fts if ft == ft_bst)
        if len(pl_bst) > 1:
            ft_2nd = ft_bst
        elif len(fts) > 1:
            ft_2nd = min([ft for _, ft in fts if ft != ft_bst])
        else:
            ft_2nd = None
        return ft_bst, ft_2nd, pl_bst

    def sorted_in_comms(self, task):
        return sorted(
            task.communications(COMM_INPUT),
            key=lambda c: self.ft_est(c.from_task))

    def start_time_est(self, task, comms, pl):
        if isinstance(pl, Task):
            ft = 0
            for c in comms:
                if c.from_task != pl:
                    ft = max(ft, self.ft_est(c.from_task)) + self.RT(c)
            return max(ft, self.ft_est(pl))
        elif isinstance(pl, Machine):
            ft = 0
            ft_m = 0
            for c in comms:
                if not self.is_placed(
                        c.from_task) or self.PL_m(c.from_task) != pl:
                    ft = max(ft, self.ft_est(c.from_task)) + self.RT(c)
                else:
                    ft_m = max(ft_m, self.ft_est(c.from_task))
            return max(ft, ft_m)
        elif pl is None:
            ft = 0
            for c in comms:
                ft = max(ft, self.ft_est(c.from_task)) + self.RT(c)
            return ft

    @tryFT
    @memo
    def ft_est(self, task):
        ft_bst, _, _ = self.rt_before(task)
        return ft_bst + self.RT(task)

    @memo
    def rt_after(self, task):
        comms = self.sorted_out_comms(task)
        if comms:
            lc = len(comms)
            X = [0 for _ in range(lc)]
            Hu = [0 for _ in range(lc)]
            Hd = [0 for _ in range(lc)]
            for i, c in enumerate(comms):
                if i == 0:
                    X[i] = self.RT(c)
                    Hu[i] = self.RT(c) + self.at_est(c.to_task)
                else:
                    X[i] = X[i - 1] + self.RT(c)
                    Hu[i] = max(Hu[i - 1], X[i] + self.at_est(c.to_task))
            for i, c in enumerate(reversed(comms)):
                if i == 0:
                    Hd[lc - i - 1] = self.RT(c) + self.at_est(c.to_task)
                else:
                    Hd[lc - i - 1] = self.RT(c) + max(Hd[lc - i],
                                                      self.at_est(c.to_task))
            r = Hu[-1]
            rn = r
            for i, c in enumerate(comms):
                if self.comm_can_follow(task, c):
                    hu = 0 if i == 0 else Hu[i - 1]
                    x = 0 if i == 0 else X[i - 1]
                    hd = 0 if i == len(comms) - 1 else Hd[i + 1]
                    r1 = max(hu, self.at_est(c.to_task), x + hd)
                    r = min(r, r1)
        else:
            r = rn = 0
        return r, rn

    def sorted_out_comms(self, task):
        return sorted(
            task.communications(COMM_OUTPUT),
            key=lambda c: self.at_est(c.to_task),
            reverse=True)

    def comm_can_follow(self, task, c):
        return any(task is t for t in self.rt_before(c.to_task)[2])

    @memo
    def at_est(self, task):
        rt, _ = self.rt_after(task)
        return rt + self.RT(task)


class NConflict(CSort):
    def mc_helper(self, p0, p1):
        if len(p0) == 1:
            pl = min(p0)
            if pl and any(pl is p for p in p1):
                return True
        return False

    def may_conflict(self, p0, p1):
        return self.mc_helper(p0, p1) or self.mc_helper(p1, p0)


class N2Conflict(CSort):
    def may_conflict(self, p0, p1):
        return p0 & p1


class StrictCommFollowTest(CSort):
    def comm_can_follow(self, task, c):
        return any(task is t for t in self.rt_before(c.to_task)[2]) and not any(self.is_placed(t) for t in c.to_task.prevs())


class NSpanComparer(CSort):
    def span_est_2(self, tx, ty, t0, t0a, x0, y0, y0a, t1, t1a, x1, y1, y1a):
        if -x1 < t1 - t0 < x0:
            return min(
                t0 + x0 + x1 + y0 + y1,
                max(t0 + x0 + x1 + y0, t0 + x0 + x1 + y1a),
                max(t0 + x0 + y0a, t0 + x0 + x1 + y1),
                max(t0 + x0 + y0, t1a + x1 + y1)
            )
        return 0


class MRNSComparer(CSort):
    def span_est_2(self, tx, ty, t0, t0a, x0, y0, y0a, t1, t1a, x1, y1, y1a):
        if -x1 < t1 - t0 < x0:
            if tx.demands() + ty.demands() >= self.vm_type.capacities:
                return min(
                    t0 + x0 + x1 + y0 + y1,
                    max(t0 + x0 + x1 + y0, t0 + x0 + x1 + y1a),
                    max(t0 + x0 + y0a, t0 + x0 + x1 + y1),
                    max(t0 + x0 + y0, t1a + x1 + y1)
                )
            else:
                xx = max(t0 + x0, t1 + x1)
                return min(
                    xx + y0 + y1,
                    max(xx + y0, t1 + x1 + y1a),
                    max(t0 + x0 + y0a, xx + y1),
                    max(t0 + x0 + y0, t1a + x1 + y1)
                )
        return 0


class NS2Comparer(CSort):
    def span_est_2(self, tx, ty,  t0, t0a, x0, y0, y0a, t1, t1a, x1, y1, y1a):
        if -x1 < t1 - t0 < x0:
            return min(
                t0 + x0 + x1 + y0 + y1,
                max(t0 + x0 + x1 + y0, t0 + x0 + x1 + y1a),
                max(t0 + x0 + y0a, t0 + x0 + x1 + y1),
                max(t0 + x0 + y0, t1a + x1 + y1)
            )
        elif -x0 - x1 < t1 - t0 <= -x1:
            return min(
                t1 + x0 + x1 + y0 + y1,
                max(t0 + x0 + y0, t1 + x1 + y1a),
                max(t0 + x0 + y0a, t0 + x0 + y1),
                max(t0 + x0 + y0, t1a + x1 + y1)
            )
        elif x0 <= t1 - t0 < x0 + x1:
            return min(
                t0 + x0 + x1 + y0 + y1,
                max(t1 + x1 + y0, t1 + x1 + y1a),
                max(t0 + x0 + y0a, t1 + x1 + y1),
                max(t0 + x0 + y0, t1a + x1 + y1)
            )
        return 0


class OutCommSorter(CSort):
    def sorted_out_comms(self, task):
        dcs = {c: 0 for c in task.communications(COMM_OUTPUT)}
        for c0 in dcs.keys():
            for c1 in dcs.keys():
                if c0 != c1 and self.should_place_before(c0.to_task, c1.to_task):
                    dcs[c1] += 1
        return sorted(dcs.keys(), key=lambda c: (dcs[c], -self.default_rank(c.to_task)))


class RTEstimater(CSort):
    def rt_after(self, task):
        comms = self.sorted_out_comms(task)
        ffs = [self.comm_can_follow(task, c) for c in comms]
        rts = [self.rem_time_est(task, comms, ffs, c)
               for c in comms if self.comm_can_follow(task, c)]
        rt_2nd = self.rem_time_est(task, comms, ffs, None)
        rt_1st = min(rts, default=rt_2nd)
        return rt_1st, rt_2nd

    def rem_time_est(self, task, comms, ffs, xc):
        flag = False
        cst = 0
        tst = 0
        ft = 0
        for c, f in zip(comms, ffs):
            if xc and c == xc:
                flag = True
                tst += self.at_est(c.to_task)
                ft = max(ft, tst)
            elif flag and f and tst <= cst + self.RT(c):
                tst += self.at_est(c.to_task)
                ft = max(ft, tst)
            else:
                cst += self.RT(c)
                ft = max(ft, cst + self.at_est(c.to_task))
        return ft


class RTEstimater2(CSort):
    def rt_after(self, task):
        comms = self.sorted_out_comms(task)
        ato = 0
        tc = 0
        for c in comms:
            tc += self.RT(c)
            ato = max(ato, tc + self.at_est(c.to_task))
        at = 0
        tc = 0
        tt = 0
        for c in comms:
            if self.comm_can_follow(task, c) and tc + self.RT(c) > tt:
                tt += self.at_est(c.to_task)
                at = max(at, tt)
            else:
                tc += self.RT(c)
                at = max(at, tc + self.at_est(c.to_task))
        return at, ato


class C2Sort(CSort):
    def start_time_est(self, task, comms, pl):
        if isinstance(pl, Task):
            ft = 0
            ft_m = 0
            for c in comms:
                if c.from_task != pl:
                    ft = max(ft, self.ft_est(c.from_task)) + self.RT(c)
                else:
                    ft_m = max(ft_m, self.ft_est(c.from_task))
            return max(ft, ft_m)
        elif isinstance(pl, Machine):
            ft = 0
            ft_m = 0
            for c in comms:
                if not self.is_placed(
                        c.from_task) or self.PL_m(c.from_task) != pl:
                    ft = max(ft, self.ft_est(c.from_task)) + self.RT(c)
                else:
                    ft_m = max(ft_m, self.ft_est(c.from_task))
            ft, _ = pl.earliest_slot_for_task(self.vm_type, task, max(
                ft, ft_m))
            return ft
        elif pl is None:
            ft = 0
            for c in comms:
                ft = max(ft, self.ft_est(c.from_task)) + self.RT(c)
            return ft


class C3Sort(C2Sort):
    def prepare_selection(self, task):
        super().prepare_selection(task)
        memo_clean(self.ft_w_comm_est)

    @memo
    def ft_w_comm_est(self, task):
        if not self.is_placed(task):
            return self.ft_est(task)
        machine = self.PL_m(task)
        ft = self.ft_est(task)
        for c in task.communications(COMM_OUTPUT):
            if self.is_placed(c.to_task) and self.PL_m(c.to_task) != machine:
                ft += self.RT(c)
        return ft

    def sorted_in_comms(self, task):
        return sorted(
            task.communications(COMM_INPUT),
            key=lambda c: self.ft_w_comm_est(c.from_task))

    def start_time_est(self, task, comms, pl):
        if isinstance(pl, Task):
            ft = 0
            ft_m = 0
            for c in comms:
                if c.from_task != pl:
                    ft = max(ft, self.ft_w_comm_est(c.from_task)) + self.RT(c)
                else:
                    ft_m = max(ft_m, self.ft_est(c.from_task))
            return max(ft, ft_m)
        elif isinstance(pl, Machine):
            ft = 0
            ft_m = 0
            for c in comms:
                if not self.is_placed(
                        c.from_task) or self.PL_m(c.from_task) != pl:
                    ft = max(ft, self.ft_w_comm_est(c.from_task)) + self.RT(c)
                else:
                    ft_m = max(ft_m, self.ft_est(c.from_task))
            ft, _ = pl.earliest_slot_for_task(self.vm_type, task, max(
                ft, ft_m))
            return ft
        elif pl is None:
            ft = 0
            for c in comms:
                ft = max(ft, self.ft_w_comm_est(c.from_task)) + self.RT(c)
            return ft


class RTEstimater3(C3Sort):
    # @profile
    def rt_before(self, task):
        comms = self.sorted_in_comms(task)
        pls = {}
        # print("                pppp", task, comms, [self.ft_w_comm_est(c.from_task) for c in comms])
        for i, c in enumerate(comms):
            if self.is_placed(c.from_task):
                m = self.PL_m(c.from_task)
                if m not in pls:
                    pls[m] = [i]
                else:
                    pls[m].append(i)
            else:
                pls[c.from_task] = [i]

        A = np.zeros(len(comms), dtype=int)
        B = np.zeros(len(comms), dtype=int)
        bt_none = 0
        for i, c in enumerate(comms):
            rt_c = self.RT(c)
            cst = self.ft_w_comm_est(c.from_task)
            rst = max(bt_none, cst)
            A[i] = rst + rt_c - bt_none
            B[i] = rst - cst
            bt_none = rst + rt_c
            # print(c, cst, rst)

        k = None
        ma = None
        M = np.zeros(len(comms), dtype=int)
        for i in range(len(comms) - 1, -1, -1):
            if k is None or k >= B[i]:
                k = B[i]
                ma = i
            M[i] = ma

        bt = bt_none
        bto = None
        pl_bst = set([None])
        for k, v in pls.items():
            ft = self.adv_time(task, comms, bt_none, A, B, M, k, v)
            if ft < bt:
                bto = bt
                bt = ft
                pl_bst = set([k])
            elif ft == bt:
                bto = ft
                pl_bst.add(k)
            elif bto is None or ft < bto:
                bto = ft
        return bt, bto, pl_bst

    def adv_time(self, task, comms, bt_none, A, B, M, pl, pli):
        ma = 0
        d = 0
        ft_m = 0
        for i in pli:
            ft_m = max(ft_m, self.ft_est(comms[i].from_task))
            if i < ma:
                continue
            elif i == len(comms) - 1:
                d += A[i]
            else:
                ma = M[i + 1]
                d += min(A[i], B[ma] - d)
        # print(">>", task, ft_m, bt_none, d)
        ft = max(ft_m, bt_none - d)
        if isinstance(pl, Machine):
            return pl.earliest_slot_for_task(self.vm_type, task, ft)[0]
        else:
            return ft
