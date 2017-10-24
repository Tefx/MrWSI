#!/usr/bin/env python

from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context
from MrWSI.simulation.base import FCFSEnv, SimEnv
from MrWSI.simulation.fair import FairEnv
from MrWSI.algorithms.homogeneous import *

from math import ceil
from statistics import mean, median


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

        # base = getattr(results[0], field)
        # if all(getattr(res, field) == base for res in results):
        # break
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


def solve_alg(alg):
    return alg.span


if __name__ == "__main__":
    import os
    from MrWSI.utils.plot import plot_cmp_results
    from multiprocessing import Pool

    pegasus_wrk_path = "./resources/workflows/pegasus"
    random_wrk_path = "./resources/workflows/random_tiny"

    ec2_file = "./resources/platforms/EC2.plt"
    result_log = {}
    for wrk_path, wrk_name in random_wrks(random_wrk_path, ""):
        # for wrk_path, wrk_name in pegasus_wrks(pegasus_wrk_path, ""):
        problem = HomoProblem.load(wrk_path, ec2_file, "c4.xlarge", 1, 1000)
        # if problem.num_tasks > 90: continue
        eft = EFT(problem)
        results = [
            # eft,
            # FairEnv(eft),
            FCFSEnv(eft),
            # mkalg("CAEFT(U)", UpwardRanking, CAEFT)(problem),
            # mkalg("CAEFT(C3.5)", NConflict, NSpanComparer, RTEstimater, C3Sort, CAEFT)(problem),
            # mkalg("CAEFT(M3)", M3Ranking, CAEFT)(problem),
            # mkalg("CAEFT(PU)", UpwardRanking, CAEFT_P)(problem),
            mkalg("CAEFT(PU2)", UpwardRanking, CAEFT_P2)(problem),
            # mkalg("CAEFT(PM3)", M3Ranking, CAEFT_P)(problem),
            # mkalg("CAEFT(PM5)", M5Ranking, CAEFT_P)(problem),
            # mkalg("CAEFT(PL4)", LLT4_3Ranking, CAEFT_P)(problem),
            # mkalg("CAEFT(PP)", PSort, CAEFT_P)(problem),
            # mkalg("CAEFT(PP1.1)", P1_1Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PP2)", P2Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC)", CSort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC1.1)", NConflict, CSort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC1.3)", NConflict, NSpanComparer, CSort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC2)", C2Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC2.1)", NConflict, C2Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC2.2)", NConflict, StrictCommFollowTest, C2Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC2.3)", NConflict, NSpanComparer, C2Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC2.4)", NConflict, NS2Comparer, C2Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC2.5)", NConflict, NSpanComparer, RTEstimater, C2Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC3)", C3Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC3.1)", NConflict, C3Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC3.3)", NConflict, NSpanComparer, C3Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PCS3.3)", NConflict, NSpanComparer, CSortStatic, CAEFT_P)(problem),
            # mkalg("CAEFT(PC3.5)", NConflict, NSpanComparer, RTEstimater, C3Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC3.5.m)", NConflict, MRNSComparer, RTEstimater, C3Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC3.6)", NConflict, NSpanComparer, RTEstimater, OutCommSorter, C3Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC3.7)", NConflict, NSpanComparer, RTEstimater2, C3Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC3.8)", NConflict, NSpanComparer, RTEstimater2, OutCommSorter, C3Sort, CAEFT_P)(problem),
            # mkalg("CAEFT(PC3.9)", NConflict, NSpanComparer, RTEstimater2, RTEstimater3, C3Sort, CAEFT_P)(problem),
            # mkalg("CA", CASort, CAEFT_P)(problem),
            # mkalg("CA2", CASort, CAEFT_P2)(problem),
            mkalg("CA3", CA3Sort, CAEFT_P2)(problem),
            mkalg("CA4", CA4Sort, CAEFT_P2)(problem),
        ]
        for alg in results:
            alg.export("results/{}.{}.schedule".format(wrk_name, alg.alg_name))
        log_record(result_log, results)
        # if results[-1].span != min(x.span for x in results):
        print("{:<16} ".format(wrk_name) + " ".join(
            str_result(res) for res in results))
    for alg, res in result_log.items():
        rs = res["span"]
        print(alg, mean(rs), median(rs))
    plot_cmp_results(result_log, "span", "box")
