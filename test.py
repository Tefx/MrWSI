#!/usr/bin/env python

from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context
from MrWSI.simulation.base import FCFSEnv
from MrWSI.simulation.fair import FairEnv
from MrWSI.algorithms.homogeneous import *

from math import ceil


def str_result(alg_name, res):
    return "{}:<{:<2}> {:6}s/${:<6.2f}".format(alg_name, res.machine_number,
                                               res.span / 100, res.cost)


def pegasus_wrks(wrk_dir, start=""):
    for wrk in sorted(
            os.listdir(wrk_dir), key=lambda x: int(x[:-4].split("_")[1])):
        if wrk.endswith(".wrk") and wrk.startswith(start):
            yield os.path.join(wrk_dir, wrk), wrk[:-4]


def random_wrks(wrk_dir, start=""):
    for wrk in sorted(os.listdir(wrk_dir), key=lambda x: int(x.split("_")[0])):
        if wrk.endswith(".wrk") and wrk.startswith(start):
            yield os.path.join(wrk_dir, wrk), wrk[:-4]


def log_record(log, results):
    if not log:
        for alg, _ in results:
            log[alg] = {"span": [], "cost": []}

    for field in ["span", "cost"]:
        res = [getattr(res, field) for _, res in results]
        v_min = min(res)
        v_max = max(res)

        for alg, res in results:
            value = (getattr(res, field) - v_min) / (
                v_max - v_min) if v_max != v_min else 0
            log[alg][field].append(value)


if __name__ == "__main__":
    import os
    from MrWSI.utils.plot import plot_cmp_results

    pegasus_wrk_path = "./resources/workflows/pegasus"
    random_wrk_path = "./resources/workflows/random_tiny"

    ec2_file = "./resources/platforms/EC2_small.plt"
    result_log = {}
    for wrk_path, wrk_name in random_wrks(random_wrk_path, ""):
    # for wrk_path, wrk_name in pegasus_wrks(pegasus_wrk_path):
        problem = HomoProblem.load(wrk_path, ec2_file, "t2.xlarge", 3600, 1000)
        # if problem.num_tasks > 90: continue
        eft = EFT(problem)
        results = [
            # ("EFT(s)", eft),
            # ("EFT(fcfs)", FCFSEnv(eft)),
            # ("EFT(fair)", FairEnv(eft)),
            ("CAEFT(PU)", CAEFT_PU(problem)),
            ("CAEFT(PT)", CAEFT_PT(problem)),
            ("CAEFT(PS)", CAEFT_PS(problem)),
            ("CAEFT(PM)", CAEFT_PM(problem)),
            ("CAEFT(PL)", CAEFT_PL(problem)),
            ("CAEFT(PL2)", CAEFT_PL2(problem)),
            ("CAEFT(PL3)", CAEFT_PL3(problem)),
        ]
        log_record(result_log, results)
        if results[-1][1].span != min(x.span for _, x in results):
            print("{:<16} ".format(wrk_name) + " ".join(
                str_result(*res) for res in results))
    for alg, res in result_log.items():
        rs = res["span"]
        print(alg, sum(rs) / len(rs))
    plot_cmp_results(result_log, "span")
