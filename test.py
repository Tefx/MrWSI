#!/usr/bin/env python

from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context
from MrWSI.algorithms.heft import heft
from MrWSI.algorithms.ca_eft import CA_EFT, CA_EFT2
from MrWSI.simulation.base import FCFSEnv
from MrWSI.simulation.fair import FairEnv


def str_result(alg_name, res):
    return "{}: {:6}s/${:<6.2f}".format(alg_name, res[0], res[1])


if __name__ == "__main__":
    import os

    ec2_file = "./resources/platforms/EC2_small.plt"
    wrk_dir = "./resources/workflows/"
    for wrk in sorted(
            os.listdir(wrk_dir), key=lambda x: int(x[:-4].split("_")[1])):
    # for wrk in ["Sipht_100.wrk"]:
        if wrk.endswith(".wrk"):
            problem = Problem.load(
                os.path.join(wrk_dir, wrk),
                ec2_file,
                type_family="t2",
                charge_unit=1)
            schedule, scheduled_cost = heft(problem)
            scheduled_span = schedule.span()
            results = [
                ("HEFT (scheduled)", (scheduled_span, scheduled_cost)),
                ("HEFT (fcfs)", FCFSEnv(problem, schedule).run()),
                ("HEFT (fair)", FairEnv(problem, schedule).run()),
                ("CA_EFT(C)", (CA_EFT(problem).solve())),
                ("CA_EFT(V)", CA_EFT2(problem).solve()),
            ]
            print("{:<16} ".format(wrk[:-4]) + " ".join(
                str_result(*res) for res in results))
