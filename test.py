#!/usr/bin/env python

from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context
from MrWSI.simulation.base import FCFSEnv, SimEnv
from MrWSI.simulation.fair import FairEnv
from MrWSI.algorithms.homogeneous import *
from MrWSI.utils.plot import plot_cmp_results

import os
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
        if v_min == v_max:
            break

        for res in results:
            value = (getattr(res, field) - v_min) / (
                v_max - v_min) if v_max != v_min else 0
            log[res.alg_name][field].append(value)


def log_record_r(log, results):
    if not log:
        for res in results[1:]:
            log[res.alg_name] = {"span": [], "cost": []}

    for field in ["span", "cost"]:
        base = getattr(results[0], field)
        if all(getattr(res, field) == base for res in results):
            break
        for res in results[1:]:
            log[res.alg_name][field].append(base / getattr(res, field))
            # log[res.alg_name][field].append(getattr(res, field) / base)


AlgRes = namedtuple("AlgRes", ["alg_name", "span", "cost"])


def two_algs(alg0, alg1, problem):
    alg0 = alg0(problem)
    alg1 = alg1(problem)
    if alg0.span < alg1.span:
        alg0.alg_name = "two"
        return alg0
    else:
        alg1.alg_name = "two"
        return alg1


def run_alg_on(wrk):
    ec2_file = "./resources/platforms/EC2.plt"
    wrk_path, wrk_name = wrk
    problem = HomoProblem.load(wrk_path, ec2_file, "c4.xlarge", 1, 2000)
    eft = EFT(problem)
    algs = [
        mkalg("Seq", SeqPlan)(problem),
        # FairEnv(eft),
        FCFSEnv(eft),
        # mkalg("CAEFT(U)", EFT_RankU, CAEFT)(problem),
        mkalg("CAEFT(PU)", EFT_RankU, CAEFT_P)(problem),
        mkalg("CA2Fit5(PU)", CANewRA, CAFit5, CAMoreCompare, CAEFT_P)(problem),
        # mkalg("CA3.3(PU)", CA3_3, CAEFT_P)(problem),
        # mkalg("CAN3(PU)", CAN3, CAEFT_P)(problem),
        # mkalg("CAN3.1(PU)", CAN3_1, CAEFT_P)(problem),
        # mkalg("CAN3.2(PU)", CAN3_2, CAEFT_P)(problem),
        # mkalg("CAN3.2.1(PU)", CAN3_2_1, CAEFT_P)(problem),
        # mkalg("CAN5(PU)", CAN5, CAEFT_P)(problem),
        mkalg("CAN6(PU)", CAN6, CAEFT_P)(problem),
        # mkalg("CAN6.1(PU)", CAN6_1, CAEFT_P)(problem),
        mkalg("CAN6.2(PU)", CAN6_2, CAEFT_P)(problem),
        # mkalg("CAN6.3(PU)", CAN6_3, CAEFT_P)(problem),
        # mkalg("CAN6.1.1(PU)", CAN6_1_1, CAEFT_P)(problem),
        # two_algs(mkalg("_", CANewRA, CAFit5, CAMoreCompare, CAEFT_P),
        # mkalg("_", CA3_2, CAEFT_P),
        # problem),
    ]
    # for alg in algs:
    # alg.export("results/{}.{}.schedule".format(wrk_name, alg.alg_name))
    for alg in algs:
        alg.span
    if algs[-1].span != min(x.span for x in algs):
        print("{:<16}(CCR={:<.1f}) ".format(wrk_name, problem.ccr) +
        " ".join(str_result(alg) for alg in algs))
    return [AlgRes(alg.alg_name, alg.span, alg.cost) for alg in algs]


def stat_n_plot(all_results, plot_type="box", std_type="MM"):
    result_log = {}
    for results in all_results:
        if std_type == "MM":
            log_record(result_log, results)
        elif std_type == "AS":
            log_record_r(result_log, results)
    for alg, res in result_log.items():
        rs = res["span"]
        print(alg, mean(rs), median(rs))
    if std_type == "MM":
        plot_cmp_results(result_log, "span", plot_type, 0)
    elif std_type == "AS":
        plot_cmp_results(result_log, "span", plot_type, 1)


if __name__ == "__main__":
    from multiprocessing.pool import Pool

    pegasus_wrk_path = "./resources/workflows/pegasus"
    random_wrk_path = "./resources/workflows/random_tiny"

    wrks = list(random_wrks(random_wrk_path, ""))
    # all_results = list(map(run_alg_on, wrks))
    all_results = list(Pool().map(run_alg_on, wrks))
    stat_n_plot(all_results, "hist", "AS")
