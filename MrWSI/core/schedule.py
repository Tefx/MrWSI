class Schedule(object):
    def __init__(self, problem, PL, TYP, ST, num_vms):
        self.problem = problem
        self.PL = PL
        self.TYP = TYP
        self.ST = ST
        self.num_vms = num_vms

    def TYP_PL(self, task):
        return self.TYP(self.PL(task))

    def need_communication(self, comm):
        return comm.data_size and \
                self.PL(comm.from_task) != self.PL(comm.to_task)
