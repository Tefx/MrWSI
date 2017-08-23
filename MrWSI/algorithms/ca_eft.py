from MrWSI.core.platform import Context, Machine, Platform, bandwidth2capacities, COMM_INPUT, COMM_OUTPUT
from MrWSI.core.resource import MultiRes
from MrWSI.algorithms.heft import sort_by_rank_u
from math import ceil
from MrWSI.utils.plot import plot_usage


class NCMachine(Machine):
    def __init__(self, problem):
        self.context = Context(problem)
        super().__init__(None, self.context)

    def __del__(self):
        del self.context


class CA_EFT(object):
    def __init__(self, problem):
        self.problem = problem
        self.context = Context(problem)
        self.platform = Platform(self.context)
        self.start_times = {}
        self.finish_times = {}
        self.placements = {}
        self.comm_rates = {}

    def candidate_types(self, min_mr, max_mr):
        typs = [typ for typ in self.problem.types if typ.capacities >= min_mr]
        for typ in typs:
            if not any(
                    t for t in typs
                    if t != typ and max_mr <= t.capacities <= typ.capacities):
                yield typ

    def snapshot_for_prevs(self, task, task_machine):
        state = {}
        for prev_task in task.prevs():
            prev_machine = self.placements[prev_task.task_id]
            if prev_machine != task_machine:
                state[prev_machine] = (prev_machine.vm_type,
                                       prev_machine.cost(), [])
        return state

    def comm_by_ranks(self, task, task_machine):
        est = 0
        for comm in task.in_communications:
            if self.placements[comm.from_task_id] == task_machine:
                est = max(est, self.finish_times[comm.from_task_id])
            else:
                est = max(
                    est,
                    self.finish_times[comm.from_task_id] + comm.mean_runtime())
        ranks = []
        for comm in task.in_communications:
            if self.placements[comm.from_task_id] != task_machine \
               and est != self.finish_times[comm.from_task_id]:
                ranks.append([
                    float(comm.mean_runtime()) /
                    (est - self.finish_times[comm.from_task_id]), comm
                ])
        return ranks

    def best_comm_pls_on(self, comm, est, from_machine, from_type, to_machine,
                         to_type):
        cr = min(from_type.bandwidth, to_type.bandwidth)
        runtime = comm.runtime(cr)
        st, _, _ = from_machine.earliest_slot_for_communication(
            to_machine, from_type, to_type, comm, cr, est)
        ci = from_machine.cost_increase(st, runtime,
                                        from_type) + to_machine.cost_increase(
                                            st, runtime, to_type)
        crs = [(runtime, cr)]
        return st, st + runtime, crs, ci

    def best_in_comm_pls(self, task, task_machine, tm_type):
        machine_snapshot = self.snapshot_for_prevs(task, task_machine)
        comm_pls = {}
        latest_comm_finish_time = 0
        earliest_comm_start_time = float("inf")
        # comm_n_ranks = [(c.mean_runtime() + self.finish_times[c.from_task_id],
        # c) for c in task.in_communications]
        comm_n_ranks = self.comm_by_ranks(task, task_machine)
        for _, comm in sorted(comm_n_ranks, key=lambda x: x[0], reverse=True):
            from_machine = self.placements[comm.from_task_id]
            if not comm.data_size or from_machine == task_machine: continue
            st_bst, ft_bst, ci_bst, crs_bst, fm_type_bst = (float("inf"),
                                                            float("inf"),
                                                            float("inf"), None,
                                                            None)
            est = self.finish_times[comm.from_task_id]
            for fm_type in self.candidate_types(
                    from_machine.peak_usage(),
                    from_machine.peak_usage() + bandwidth2capacities(
                        tm_type.bandwidth,
                        self.problem.multiresource_dimension, COMM_OUTPUT)):
                st, ft, crs, ci = self.best_comm_pls_on(
                    comm, est, from_machine, fm_type, task_machine, tm_type)
                if (ft, ci) < (ft_bst, ci_bst):
                    st_bst, ft_bst, ci_bst, crs_bst, fm_type_bst = (st, ft, ci,
                                                                    crs,
                                                                    fm_type)
            from_machine.vm_type = fm_type_bst
            from_machine.place_communication_2(comm, st_bst, crs_bst,
                                               COMM_OUTPUT)
            task_machine.place_communication_2(comm, st_bst, crs_bst,
                                               COMM_INPUT)
            comm_pls[comm] = (st_bst, crs_bst)
            earliest_comm_start_time = min(earliest_comm_start_time, st_bst)
            latest_comm_finish_time = max(latest_comm_finish_time, ft_bst)
            machine_snapshot[from_machine][-1].append(comm)

        ci = 0
        machine_type_changes = {}
        for machine, (orig_type, orig_cost, comms) in machine_snapshot.items():
            ci += machine.cost() - orig_cost
            for comm in comms:
                machine.remove_communication_2(comm, COMM_OUTPUT)
                task_machine.remove_communication_2(comm, COMM_INPUT)
            machine_type_changes[machine] = machine.vm_type
            machine.vm_type = orig_type
        return comm_pls, machine_type_changes, earliest_comm_start_time, latest_comm_finish_time, ci

    def plan_task(self, task):
        est = max(
            [self.finish_times[pt.task_id] for pt in task.prevs()], default=0)
        max_bandwidth_cap = bandwidth2capacities(
            self.problem.type_max_bandwidth(),
            self.problem.multiresource_dimension, COMM_INPUT)
        st_bst, ft_bst, ci_bst, comm_pls_bst, comm_mtc_bst, machine_bst, type_bst = (
            float("inf"), float("inf"), float("inf"), None, None, None, None)
        for machine in list(self.platform.machines) + [
                Machine(None, self.context)
        ]:
            for vm_type in self.candidate_types(
                    MultiRes.max(machine.peak_usage(), task.demands()),
                    machine.peak_usage() + task.demands() + max_bandwidth_cap):
                comm_pls, comm_mtc, comm_est, comm_lft, ci = self.best_in_comm_pls(
                    task, machine, vm_type)
                real_est = max(est, comm_lft)
                st, _ = machine.earliest_slot_for_task(vm_type, task, real_est)
                ft = st + task.runtime(vm_type)
                comm_n_task_length = ft - min(st, comm_est)
                ci += machine.cost_increase(
                    min(st, comm_est), comm_n_task_length, vm_type)
                if (ft, ci) < (ft_bst, ci_bst):
                    st_bst, ft_bst, ci_bst, comm_pls_bst, comm_mtc_bst, machine_bst, type_bst = (
                        st, ft, ci, comm_pls, comm_mtc, machine, vm_type)

        for machine, vm_type in comm_mtc_bst.items():
            machine.vm_type = vm_type
        for comm, (st, crs) in comm_pls_bst.items():
            self.placements[comm.from_task_id].place_communication_2(
                comm, st, crs, COMM_OUTPUT)
            machine_bst.place_communication_2(comm, st, crs, COMM_INPUT)
            self.start_times[comm.pair_id] = st
            self.comm_rates[comm.pair_id] = crs
        machine_bst.vm_type = type_bst
        machine_bst.place_task(task, st_bst)
        self.placements[task.task_id] = machine_bst
        self.start_times[task.task_id] = st_bst
        self.finish_times[task.task_id] = st_bst + task.runtime(type_bst)
        self.platform.update_machine(machine_bst)

    def solve(self):
        for task in sort_by_rank_u(self.problem):
            self.plan_task(task)
        # plot_usage(self.platform, 0, "{}.test".format(self.__class__.__name__))
        # plot_usage(self.platform, 2, "{}.test".format(self.__class__.__name__))
        # plot_usage(self.platform, 3, "{}.test".format(self.__class__.__name__))
        return self.platform.span(), self.platform.cost()


class CA_EFT2(CA_EFT):
    def best_comm_pls_on(self, comm, est, from_machine, from_type, to_machine,
                         to_type):
        remaining_data_size = comm.data_size
        st = est
        crs = []
        while remaining_data_size > 0:
            cr_0, len_0 = from_machine.current_available_cr(
                st, from_type, COMM_OUTPUT)
            cr_1, len_1 = to_machine.current_available_cr(
                st, to_type, COMM_INPUT)
            cr = min(cr_0, cr_1)
            length = min(len_0, len_1)
            if cr * length >= remaining_data_size:
                length = ceil(remaining_data_size / cr)
                remaining_data_size = 0
            else:
                remaining_data_size -= cr * length
            crs.append([length, cr])
            st += length
        while not crs[0][1]:
            crs.pop(0)
        runtime = sum(length for length, _ in crs)
        st -= runtime
        ci = from_machine.cost_increase(st, runtime,
                                        from_type) + to_machine.cost_increase(
                                            st, runtime, to_type)
        return st, st + runtime, crs, ci
