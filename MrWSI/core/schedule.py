class Schedule(object):
    def __init__(self, problem, PL, TYP, ST, num_vms):
        self.problem = problem
        self.PL = PL
        self.TYP = TYP
        self.ST = ST
        self.num_vms = num_vms

    @classmethod
    def from_arrays(cls, problem, pls, typs, sts):
        return cls(problem, lambda x: pls[x.task_id], lambda x: typs[x],
                   lambda x: sts[x.task_id], len(typs))

    def FT(self, task):
        return self.ST(task) + task.runtime(self.TYP_PL(task))

    def TYP_PL(self, task):
        return self.TYP(self.PL(task))

    def span(self):
        return max(self.FT(task) for task in self.problem.tasks) - min(
            self.ST(task) for task in self.problem.tasks)
