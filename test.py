#!/usr/bin/env python

from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context
from MrWSI.algorithms.heft import heft

if __name__ == "__main__":
    import os

    ec2_file = "./resources/platforms/EC2.plt"
    wrk_dir = "./resources/workflows/"
    for wrk in os.listdir(wrk_dir):
    # for wrk in ["Sipht_30.wrk"]:
        if wrk.endswith(".wrk"):
            problem = Problem.load(
                os.path.join(wrk_dir, wrk),
                ec2_file,
                type_family="c4",
                charge_unit=3600,
                platform_limits=20)
            platform = heft(problem)
            print("{:<24} Makespan:{:<8} Cost:{:8.2f}".format(wrk[:-4], platform.span(), platform.cost()))
