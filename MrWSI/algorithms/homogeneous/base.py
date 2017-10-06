from MrWSI.core.problem import Problem, Task, Communication, COMM_INPUT, COMM_OUTPUT
from MrWSI.core.platform import Context, Platform, Machine
from MrWSI.core.schedule import Schedule


class HomoProblem(Problem):
    @classmethod
    def load(cls,
             wrk_file,
             plt_file,
             vm_type,
             charge_unit=3600,
             platform_limit=1000):
        return super().load(wrk_file, plt_file, vm_type, charge_unit,
                            [platform_limit])

    @property
    def vm_type(self):
        return self.types[0]


class Heuristic(object):
    log = [
        # "alg",
        # "sort",
    ]
    allow_share = False
    allow_preemptive = False

    def __init__(self, problem):
        self.problem = problem
        self.context = Context(problem)
        self.platform = Platform(self.context)
        self.vm_type = problem.vm_type
        self.bandwidth = self.vm_type.bandwidth

        self.start_times = {}
        self.finish_times = {}
        self.placements = {}
        self.have_solved = False
        self._pls = None
        self._order = []

    def ST(self, obj):
        return self.start_times[obj]

    def FT(self, obj):
        return self.finish_times[obj]

    def PL_m(self, task):
        return self.placements[task]

    def TYP(self, x):
        return self.vm_type

    def RT(self, x):
        if isinstance(x, Task):
            return x.runtime(self.vm_type)
        else:
            return x.runtime(self.bandwidth)

    def need_communication(self, comm, to_machine=None):
        if not to_machine:
            if comm.to_task in self.placements:
                to_machine = self.PL_m(comm.to_task)
            else:
                return True
        else:
            return comm.data_size and \
                self.placements[comm.from_task] != to_machine

    def sort_tasks(self):
        pass

    def default_fitness(self):
        return float("inf"), float("inf")

    def compare_fitness(self, f0, f1):
        return f0 < f1

    def place_task(self, task, machine, st):
        machine.place_task(task, st)
        self.platform.update_machine(machine)
        self.placements[task] = machine
        self.start_times[task] = st
        self.finish_times[task] = st + task.runtime(self.vm_type)

    def perform_placement(self, task, placement):
        pass

    def plan_task_on(self, task, machine):
        pass

    def available_machines(self):
        for machine in self.platform.machines:
            yield machine
        if self.problem.platform_limits[0] > len(self.platform):
            yield Machine(self.problem.vm_type, self.context)

    def solve(self):
        for task in self.sort_tasks():
            self._order.append(task)
            placement_bst, fitness_bst = None, self.default_fitness()
            for machine in self.available_machines():
                assert machine.vm_type.capacities >= task.demands()
                placement, fitness = self.plan_task_on(task, machine)
                if self.compare_fitness(fitness, fitness_bst):
                    placement_bst, fitness_bst = placement, fitness
            self.perform_placement(task, placement_bst)
        self.have_solved = True

        if "alg" in self.log:
            self.log_alg("./")

    @property
    def cost(self):
        if not self.have_solved: self.solve()
        return self.platform.cost()

    @property
    def span(self):
        if not self.have_solved: self.solve()
        return self.platform.span()

    @property
    def machine_number(self):
        if not self.have_solved: self.solve()
        return len(self.platform)

    @property
    def schedule(self):
        if not self.have_solved: self.solve()
        pls = {}
        for i, machine in enumerate(self.platform):
            for task in machine.tasks:
                pls[task] = i
        return Schedule(self.problem, lambda x: pls[x], self.TYP, self.ST,
                        len(self.platform))

    def log_alg(self, path):
        from MrWSI.utils.plot import draw_dag
        import os.path

        draw_dag(self.problem, os.path.join(path, "dag.png"))
        print("By {}, Order: {}".format(self.alg_name, self._order))
        for machine in sorted(
                self.platform, key=lambda m: (m.open_time(), m.close_time())):
            print("Machine {}:".format(str(id(machine))[-4:]))
            # machine.print_list()
            for task in sorted(
                    machine.tasks, key=lambda t: (self.ST(t), self.FT(t))):
                print("  Task {}: [{}, {})".format(
                    task, self.ST(task), self.FT(task)))
                for comm in sorted(
                    [
                        c for c in task.communications(COMM_OUTPUT)
                        if c in self.start_times
                    ],
                        key=self.ST):
                    print("    COMM {}: [{}, {}) <Machine {} => Machine {}>".
                          format(comm,
                                 self.ST(comm),
                                 self.ST(comm) + comm.runtime(self.bandwidth),
                                 str(id(machine))[-4:],
                                 str(id(self.PL_m(comm.to_task)))[-4:]))
        print()

    def export(self, path, attrs={}):
        import json

        if not self.have_solved: self.solve()
        machines_list = list(self.platform)
        machines = []
        for machine in machines_list:
            tasks = []
            for task in sorted(machine.tasks, key=self.ST):
                demands = task.demands()
                comms = []
                for comm in task.communications(COMM_OUTPUT):
                    if comm not in self.start_times: continue
                    to_machine = machines_list.index(self.PL_m(comm.to_task))
                    comms.append({
                        "to_task": str(comm.to_task),
                        "start_time": self.ST(comm) / 100,
                        "finish_time": self.FT(comm) / 100,
                        "data_size": int(comm.data_size * 1024 / 100),
                    })
                tasks.append({
                    "id": str(task),
                    "start_time": self.ST(task) / 100,
                    "runtime": self.RT(task) / 100,
                    "resources": (demands[0] / 1000, demands[1]),
                    "prevs": [str(t) for t in task.prevs()],
                    "succs": [str(t) for t in task.succs()],
                    "output": comms,
                })
            machines.append(tasks)
        capacities = self.vm_type.capacities
        schedule = {
            "vm_capacities": [int(capacities[0] / 1000), capacities[1]],
            "num_tasks": self.problem.num_tasks,
            "allow_share": self.allow_share,
            "allow_preemptive": self.allow_preemptive,
            "machines": machines
        }
        schedule.update(attrs)
        with open(path, "w") as f:
            json.dump(schedule, f, indent=2)
