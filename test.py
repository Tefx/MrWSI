#!/usr/bin/env python

from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context
from MrWSI.algorithms.heft import heft
from MrWSI.simulation import SimulationEnvironment

if __name__ == "__main__":
    import os

    ec2_file = "./resources/platforms/EC2.plt"
    wrk_dir = "./resources/workflows/"
    for wrk in os.listdir(wrk_dir):
    # for wrk in ["Sipht_60.wrk"]:
        if wrk.endswith(".wrk"):
            problem = Problem.load(
                os.path.join(wrk_dir, wrk),
                ec2_file,
                type_family="t2",
                charge_unit=3600)
            platform, schedule = heft(problem, 20)
            print("{:<24} Makespan:{:8}\tCost:{:8.2f}\ton {}/{} VMs".format(
                wrk[:-4],
                platform.span(), platform.cost(), len(platform.machines), platform.peak_usage()))
            sim_env = SimulationEnvironment(problem, schedule)
            print(sim_env.run())
