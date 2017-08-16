#!/usr/bin/env python

from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context
from MrWSI.algorithms.heft import heft
from MrWSI.simulation.fair import FairEnvironment
from MrWSI.simulation.fcfs import FCFSEnvironment, FCFS2Environment

if __name__ == "__main__":
    import os

    ec2_file = "./resources/platforms/EC2_small.plt"
    wrk_dir = "./resources/workflows/"
    for wrk in os.listdir(wrk_dir):
        # for wrk in ["Sipht_60.wrk"]:
        if wrk.endswith(".wrk"):
            problem = Problem.load(
                os.path.join(wrk_dir, wrk),
                ec2_file,
                type_family="t2",
                charge_unit=1)
            schedule, scheduled_cost = heft(problem)
            scheduled_span = schedule.span()
            fcfs_makespan, fcfs_cost = FCFSEnvironment(problem, schedule).run()
            fcfs2_makespan, fcfs2_cost = FCFS2Environment(problem,
                                                          schedule).run()
            fair_makespan, fair_cost = FairEnvironment(problem, schedule).run()
            print(
                "{:<16} Scheduled: {:6}s/${:<6.2f} FCFS: {:6}s/${:<6.2f}  FCFS2: {:6}s/${:<6.2f} Fair: {:6}s/${:<6.2f}".
                format(wrk[:-4], scheduled_span, scheduled_cost, fcfs_makespan,
                       fcfs_cost, fcfs2_makespan, fcfs2_cost, fair_makespan,
                       fair_cost))
