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

        # self.log("./")

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

    def log(self, path):
        from MrWSI.utils.plot import draw_dag
        import os.path

        draw_dag(self.problem, os.path.join(path, "dag.png"))
        print("By {}, Order: {}".format(self.__class__.__name__, self._order))
        for machine in sorted(
                self.platform, key=lambda m: (m.open_time(), m.close_time())):
            print("Machine {}:".format(str(id(machine))[-4:]))
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
