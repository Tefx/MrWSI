class Schedule(object):
    def __init__(self, PL, TYP, ST, num_vms):
        self.PL = PL
        self.TYP = TYP
        self.ST = ST
        self.num_vms = num_vms

    @classmethod
    def from_arrays(cls, pls, typs, sts):
        return cls(lambda x: pls[x.task_id], lambda x: typs[x],
                   lambda x: sts[x.task_id], len(typs))
