#!/usr/bin/env python

from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context
from MrWSI.simulation.base import FCFSEnv, SimEnv
from MrWSI.simulation.fair import FairEnv
from MrWSI.algorithms.homogeneous import *

from math import ceil
from statistics import mean, median
from collections import namedtuple


def str_result(res):
    return "{}:<{:<2}> {:6}s/${:<6.2f}".format(
        res.alg_name, res.machine_number, res.span / 100, res.cost)


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
        for res in results:
            log[res.alg_name] = {"span": [], "cost": []}

    for field in ["span", "cost"]:
        res = [getattr(res, field) for res in results]
        v_min = min(res)
        v_max = max(res)

        for res in results:
            value = (getattr(res, field) - v_min) / (
                v_max - v_min) if v_max != v_min else 0
            log[res.alg_name][field].append(value)


def log_record_r(log, results):
    if not log:
        for res in results:
            log[res.alg_name] = {"span": [], "cost": []}

    for field in ["span", "cost"]:
        base = getattr(results[0], field)
        if all(getattr(res, field) == base for res in results):
            break
        for res in results:
            log[res.alg_name][field].append(base / getattr(res, field))


AlgRes = namedtuple("AlgRes", ["alg_name", "span", "cost"])


def run_alg_on(wrk):
    ec2_file = "./resources/platforms/EC2.plt"
    wrk_path, wrk_name = wrk
    problem = HomoProblem.load(wrk_path, ec2_file, "c4.xlarge", 1, 1000)
    eft = EFT(problem)
    algs = [
        # eft,
        # FairEnv(eft),
        FCFSEnv(eft),
        # mkalg("CAEFT(U)", UpwardRanking, CAEFT)(problem),
        # mkalg("CAEFT(C3.5)", NConflict, NSpanComparer, RTEstimater, C3Sort, CAEFT)(problem),
        mkalg("CAEFT(PU2)", UpwardRanking, CAEFT_P2)(problem),
        # mkalg("CA", CASort, CAEFT_P)(problem),
        mkalg("CA2", CASort, CAEFT_P2)(problem),
        mkalg("CA3", CA3Sort, CAEFT_P2)(problem),
        mkalg("CA4", CA4Sort, CAEFT_P2)(problem),
    ]
    # for alg in algs:
        # alg.export("results/{}.{}.schedule".format(wrk_name, alg.alg_name))
    for alg in algs:
        alg.span
    # if algs[-1].span != min(x.span for x in algs):
    print("{:<16} ".format(wrk_name) + " ".join(str_result(alg)
                                                for alg in algs))
    return [AlgRes(alg.alg_name, alg.span, alg.cost) for alg in algs]


if __name__ == "__main__":
    import os
    from MrWSI.utils.plot import plot_cmp_results
    from multiprocessing.pool import Pool

    pegasus_wrk_path = "./resources/workflows/pegasus"
    random_wrk_path = "./resources/workflows/random_tiny"

    wrks = random_wrks(random_wrk_path, "")
    all_results = list(Pool().map(run_alg_on, wrks))

    result_log = {}
    for results in all_results:
        log_record(result_log, results)
    for alg, res in result_log.items():
        rs = res["span"]
        print(alg, mean(rs), median(rs))
    plot_cmp_results(result_log, "span", "box")
