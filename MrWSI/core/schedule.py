class Schedule(object):
    def __init__(self, problem, PL, TYP, ST, CST, num_vms):
        self.problem = problem
        self.PL = PL
        self.TYP = TYP
        self.ST = ST
        self.CST = CST
        self.num_vms = num_vms

    @classmethod
    def from_arrays(cls, problem, pls, typs, sts, csts):
        return cls(problem, pls
                   if callable(pls) else lambda x: pls[x.task_id], typs
                   if callable(typs) else lambda x: typs[x], sts
                   if callable(sts) else lambda x: sts[x.task_id], csts
                   if callable(csts) else
                   lambda x, y: csts[(x.task_id, y.task_id)], len(typs))

    def FT(self, task):
        return self.ST(task) + task.runtime(self.TYP_PL(task))

    def TYP_PL(self, task):
        return self.TYP(self.PL(task))

    def span(self):
        return max(self.FT(task) for task in self.problem.tasks) - min(
            self.ST(task) for task in self.problem.tasks)
