#!/usr/bin/env python

from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context
from MrWSI.simulation.base import FCFSEnv, SimEnv
from MrWSI.simulation.fair import FairEnv
from MrWSI.algorithms.homogeneous import *
from MrWSI.utils.plot import plot_cmp_results

import os
import sys
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
        # if all(getattr(res, field) == base for res in results): break
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
    problem = HomoProblem.load(wrk_path, ec2_file, "c4.xlarge", 1, 20)
    eft = EFT(problem)
    algs = [
        # mkalg("Seq", SeqPlan)(problem),
        # FairEnv(eft),
        # FCFSEnv(eft),
        # mkalg("CAEFT(U)", EFT_RankU, CAEFT)(problem),
        mkalg("CAEFT(PU)", EFT_RankU, CAEFT_P)(problem),
        # mkalg("PEFT(PU)", PEFT, CAEFT_P)(problem),
        # mkalg("CAEFT(PU2)", EFT_RankU2, CAEFT_P)(problem),
        # mkalg("CA2Fit5(U)", CANewRA, CAFit5, CAMoreCompare, CAEFT)(problem),
        # mkalg("CA2Fit5(PU)", CANewRA, CAFit5, CAMoreCompare, CAEFT_P)(problem),
        # mkalg("CAN6.2(PU)", CAN6_2, CAEFT_P)(problem),
        # mkalg("CAN6.2.2(N2PU)", NeighborFirstSort2, CAN6_2_2, CAEFT_P)(problem),
        # mkalg("CAN8.6(PU)", CAN8_6, CAEFT_P)(problem),
        # mkalg("CAN8.6.t0(MCPU)", CSLTest, MoreContention, CAN8_6, CAEFT_P)(problem),
        # mkalg("CAN8.6.t1(MCPU)", CSLTest2, MoreContention, CAN8_6, CAEFT_P)(problem),
        # mkalg("CAN8.6.t2(MCPU)", CSLTest3, MoreContention, CAN8_6, CAEFT_P)(problem),
        # mkalg("CAN8.6.t3(MCPU)", CSLTest6, MoreContention, CAN8_6, CAEFT_P)(problem),
        # mkalg("CAN8.6.t4(MCPU)", CSLTest7, MoreContention, CAN8_6, CAEFT_P)(problem),
        # mkalg("CAN8.6.t4(EMCPU)", ESTFTv1, CSLTest7, MoreContention, CAN8_6, CAEFT_P)(problem),
        # mkalg("CAN8.6.t4(RSPU)", CSLTest7, Sort2, MoreContention, CAN8_6, CAEFT_P)(problem),
        # mkalg("CAN8.6.t4.1(RSPU)", ESTFTv1, CSLTest7, Sort2, MoreContention, CAN8_6, CAEFT_P)(problem),
        # mkalg("CAN8.6.1.t4.1(RSPU)", ESTFTv1, CSLTest7, Sort2, MoreContention, CAN8_6_1, CAEFT_P)(problem),
        # mkalg("CAN8.6.t4.2(RSPU)", ESTFTv1, CSLTest7, Sort3, MoreContention, CAN8_6, CAEFT_P)(problem),
        # mkalg("CAN8.6.t4.2(RSPU)", ESTFTv1, CSLTest7, Sort2, MoreContention, CAN8_6, CAEFT_P)(problem),
        # mkalg("CAN8.6.t4.3(RSPU)", ESTFTv2, CSLTest7, Sort3, MoreContention, CAN8_6, CAEFT_P)(problem),
        # mkalg("CAN8.6.1.t4.4(RSPU)", ESTFTv2, CSLTest7, Sort3, MoreContention, CAN8_6_1, CAEFT_P)(problem),
        # mkalg("CAN8.6.1.1.t4.4(RSPU)", ESTFTv2, CSLTest7, Sort3, MoreContention, CAN8_6_1_1, CAEFT_P)(problem),
        # mkalg("CAN8.6.2.t4.4(RSPU)", ESTFTv2, CSLTest7, Sort3, MoreContention, CAN8_6_2, CAEFT_P)(problem)
        # mkalg("CAN8.6.3.t4.3(RSPU)", ESTFTv2, CSLTest7, Sort3, MoreContention, CAN8_6_3, CAEFT_P)(problem),
        # mkalg("CAN8.6.3.t4.3.1(RSPU)", SAT, ESTFTv2, CSLTest7, Sort3, MoreContention, CAN8_6_3, CAEFT_P)(problem),
        # mkalg("CAN8.6.3.t4.3.2(RSU)", SAT, ESTFTv3, CSLTest7, Sort3, MoreContention, CAN8_6_3, CAEFT)(problem),
        # mkalg("CAN8.6.3.t4.3.2(RSPU)", SAT, ESTFTv3, CSLTest7, Sort3, MoreContention, CAN8_6_3, CAEFT_P)(problem), #RES
        # mkalg("CAN8.6.3.t4.3.7(RSPU)", SAT, ESTFTv4, CSLTest7, Sort7, MoreContention, CAN8_6_3, CAEFT_P)(problem), #RES
        # mkalg("CAWS(PU)", CAWS, CAEFT_P)(problem),
        # mkalg("CAWS1.1(PU)", CAWSv1_1, CAEFT_P)(problem),
        # mkalg("CAWS1.1.1(PU)", CAWSv1_1_1, CAEFT_P)(problem),
        # mkalg("CAWS1.2(U)", CAWSv1_2, CAEFT)(problem),
        # mkalg("CAWS1.2(PU)", CAWSv1_2, CAEFT_P)(problem),  # RES
        # mkalg("CAWS1.2.1(PU)", CAWSv1_2_1, CAEFT_P)(problem), #RES
        # mkalg("CAWS1.3(PU)", CAWSv1_3, CAEFT_P)(problem),
        # mkalg("CAWS1.4(PU)", CAWSv1_4, CAEFT_P)(problem),
        # mkalg("CAWS1.5(PU)", CAWSv1_5, CAEFT_P)(problem),
        # mkalg("CAWS1.6(PU)", CAWSv1_6, CAEFT_P)(problem),
        # mkalg("CAWS1.7(PU)", CAWSv1_7, CAEFT_P)(problem),
        # mkalg("CAWS1.8(PU)", CAWSv1_8, CAEFT_P)(problem),
        # mkalg("CAWS1.9(PU)", CAWSv1_9, CAEFT_P)(problem),
        # mkalg("CAWS1.10(PU)", CAWSv1_10, CAEFT_P)(problem),
        # mkalg("CAWS1.11(PU)", CAWSv1_11, CAEFT_P)(problem),
        # mkalg("CAWS1.12(PU)", CAWSv1_12, CAEFT_P)(problem),
        # mkalg("CAWS1.13(PU)", CAWSv1_13, CAEFT_P)(problem), #RES
        # mkalg("CAWS1.14(PU)", CAWSv1_14, CAEFT_P)(problem),
        # mkalg("CAWS1.15(PU)", CAWSv1_15, CAEFT_P)(problem),
        # mkalg("CAWS1.16(PU)", CAWSv1_16, CAEFT_P)(problem),
        # mkalg("CAWS1.16.1(PU)", CAWSv1_16_1, CAEFT_P)(problem),
        # mkalg("CAWS1.16.2(PU)", CAWSv1_16_2, CAEFT_P)(problem),
        # mkalg("CAWS1.17(PU)", CAWSv1_17, CAEFT_P)(problem),
        # mkalg("CAS(PU)", CAS, CAEFT_P)(problem),
        # mkalg("CAS5(PU)", CAS5, CAEFT_P)(problem),
        # mkalg("CAS6(PU)", CAS6, CAEFT_P)(problem),
        # mkalg("CAS7(PU)", CAS7, CAEFT_P)(problem),
        # mkalg("CAS8(PU)", CAS8, CAEFT_P)(problem),
        # mkalg("CAS9(U)", CAS9, CAEFT)(problem),
        # mkalg("CAS9(PU)", CAS9, CAEFT_P)(problem),
        # mkalg("CAS_nr(PU)", CAS_nr, CAEFT_P)(problem),
        # mkalg("CAS_n1(PU)", CAS_n1, CAEFT_P)(problem),
        # mkalg("CAS_n2(PU)", CAS_n2, CAEFT_P)(problem),
        # mkalg("CAS_n3(PU)", CAS_n3, CAEFT_P)(problem),
        # mkalg("CAS_n4(PU)", CAS_n4, CAEFT_P)(problem),
        # mkalg("CAS_n5(PU)", CAS_n5, CAEFT_P)(problem),
        # mkalg("CAS_n6(PU)", CAS_n6, CAEFT_P)(problem),
        # mkalg("CAS_n7(PU)", CAS_n7, CAEFT_P)(problem),
        # mkalg("CAS_n8(PU)", CAS_n8, CAEFT_P)(problem),
        # mkalg("CAS_n9(PU)", CAS_n9, CAEFT_P)(problem),
        # mkalg("CAS_n10(PU)", CAS_n10, CAEFT_P)(problem),
        # mkalg("CAWS_FRA(PU)", ForwardRA, CAEFT_P)(problem),
        # mkalg("CAWS(PU)", CAWS, CAEFT_P)(problem),
        # mkalg("CAWS2(PU)", RAEst, CAEFT_P)(problem),
        # mkalg("CAWS4(PU)", NoRPr, CAEFT_P)(problem),
        # mkalg("CAWS5(PU)", RAEst, NoRPr, CAEFT_P)(problem),
        # mkalg("CAWS6(PU)", C1, CAEFT_P)(problem),
        # mkalg("CAWS7(PU)", C2, CAEFT_P)(problem),
        # mkalg("CAWS8(PU)", C3, CAEFT_P)(problem),
        # mkalg("CAWS9(PU)", C4, CAEFT_P)(problem),
        # mkalg("CAWS10(PU)", C5, CAEFT_P)(problem), #BEST
        # mkalg("CAWS11(PU)", C6, CAEFT_P)(problem),
        # mkalg("CAWS12(PU)", C7, CAEFT_P)(problem),
        # mkalg("CAWS13(PU)", C8, CAEFT_P)(problem),
        # mkalg("CAWS14(PU)", C9, CAEFT_P)(problem),
        mkalg("NCAS(PU)", NewCAS, CAEFT_P)(problem),
        mkalg("NCAS_RA(PU)", NewCAS_RA, CAEFT_P)(problem),
        mkalg("NCAS_RA2(PU)", NewCAS_RA2, CAEFT_P)(problem),
        # mkalg("NCAS_RA3(PU)", NewCAS_RA3, CAEFT_P)(problem),
        # mkalg("NCAS_m2(PU)", NewCAS_m2, CAEFT_P)(problem),
        # mkalg("NCAS_m2RA(PU)", NewCAS_m2RA, CAEFT_P)(problem),
        # mkalg("NCAS_m3RA(PU)", NewCAS_m3RA, CAEFT_P)(problem),
        # mkalg("NCAS_RP2(PU)", NewCAS_RP2, CAEFT_P)(problem),
        # mkalg("NCASc(PU)", NewCAS_Counter, CAEFT_P)(problem),
        # mkalg("CN(PU)", CN, CAEFT_P)(problem),
        # mkalg("CN2(PU)", CN2, CAEFT_P)(problem),
        # mkalg("CAS10(PU)", CAS10, CAEFT_P)(problem),
        # mkalg("CAS11(PU)", CAS11, CAEFT_P)(problem),
        # mkalg("CAS12(PU)", CAS12, CAEFT_P)(problem),
        # mkalg("CAS13(PU)", CAS13, CAEFT_P)(problem),
        # mkalg("CAS15(PU)", CAS15, CAEFT_P)(problem),
        # mkalg("CAS14(PU)", CAS14, CAEFT_P)(problem),
        # mkalg("CAWS1.9(PU)", CAWSv1_9, CAEFT_P)(problem),
        # mkalg("CAWS_r(PU)", CAWS_r, CAEFT_P)(problem),
        # mkalg("CAN8.6.3.t4.3.3(RSPU)", SAT, ESTFTv3, CSLTest7, NeighborFirstSort2, MoreContention, CAN8_6_3, CAEFT_P)(problem),
        # mkalg("CAN8.6.3.t4.3.4(RSPU)", SAT, ESTFTv2, CSLTest7, Sort6, MoreContention, CAN8_6_3, CAEFT_P)(problem),
        # mkalg("CAN8.6.3.t4.4(RSPU)", SAT, ESTFTv2, CSLTest7, Sort5, MoreContention, CAN8_6_3, CAEFT_P)(problem),
        # mkalg("CAN8.6.3.t4.3(RSU)", ESTFTv2, CSLTest7, Sort3, MoreContention, CAN8_6_3, CAEFT)(problem),
        # mkalg("CAN8.6.3.t4.4(RSPU)", ESTFTv3, CSLTest7, Sort3, MoreContention, CAN8_6_3, CAEFT_P)(problem),
        # mkalg("CAN8.6.4.t4.3(RSPU)", ESTFTv2, CSLTest7, Sort3, MoreContention, CAN8_6_4, CAEFT_P)(problem),
        # mkalg("CAN8.6.3.t5.3(RSPU)", ESTFTv2, CSLTest7_4, Sort3, MoreContention, CAN8_6_3, CAEFT_P)(problem),
        # mkalg("CAN8.6.4.t5.3(RSPU)", ESTFTv2, CSLTest7_4, Sort3, MoreContention, CAN8_6_4, CAEFT_P)(problem),
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


def stat_n_plot(all_results, plot_type="box", std_type="MM", outfile=sys.stdout):
    result_log = {}
    for results in all_results:
        if std_type == "MM":
            log_record(result_log, results)
        elif std_type == "AS":
            log_record_r(result_log, results)
    for alg, res in result_log.items():
        rs = res["span"]
        print(alg, mean(rs), median(rs), file=outfile)
    if std_type == "MM":
        plot_cmp_results(result_log, "span", plot_type, 0)
    elif std_type == "AS":
        plot_cmp_results(result_log, "span", plot_type, 1)


if __name__ == "__main__":
    from multiprocessing.pool import Pool
    from sys import argv

    if len(argv) > 1:
        wrk_path = argv[1]
    else:
        wrk_path = "./resources/workflows/random"
        # wrk_path = "./resources/workflows/pegasus"
    wrks = list(random_wrks(wrk_path, ""))
    # wrks = list(pegasus_wrks(wrk_path, ""))
    # all_results = list(Pool().map(run_alg_on, wrks))
    all_results = list(map(run_alg_on, wrks))
    stat_n_plot(all_results, "box", "AS")
