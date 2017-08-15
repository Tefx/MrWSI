#!/usr/bin/env python

from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context
from MrWSI.algorithms.heft import heft
from MrWSI.simulation.fair import FairEnvironment
from MrWSI.simulation.fcfs import FCFSEnvironment

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
                charge_unit=3600)
            platform, schedule = heft(problem)
            fair_env = FairEnvironment(problem, schedule)
            fcfs_env = FCFSEnvironment(problem, schedule)
            makespan_fair = fair_env.run()
            makespan_fcfs = fcfs_env.run()
            print(
                "{:<16} Scheduled: {:<8} Fair: {:<8} FCFS: {:<8} on {}/{} VMs".
                format(wrk[:-4],
                       platform.span(), makespan_fair, makespan_fcfs,
                       len(platform.machines), platform.peak_usage()))
