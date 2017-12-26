#!/usr/bin/env python3
import os
import os.path
from subprocess import run
from multiprocessing import Pool
from glob import glob

from dot2wrk import convert_dot

DOT_GEN = "resources/daggen/gen.py"
DOT_PATH = "resources/daggen/gen"
WRK_PATH = "resources/workflows/random_tiny"
CCR_SET = [8]


def rm_existings():
    for item in glob(os.path.join(DOT_PATH, "*")):
        os.remove(item)
    for item in glob(os.path.join(WRK_PATH, "*")):
        os.remove(item)


def convert(ccr, dot):
    print("DOT => WRK (CCR={})".format(ccr), dot)
    convert_dot(os.path.join(DOT_PATH, dot), WRK_PATH, ccr)


if __name__ == "__main__":
    from sys import argv
    from functools import partial

    if len(argv) > 1:
        ccr_set = [float(ccr) for ccr in argv[1:]]
    else:
        ccr_set = CCR_SET

    rm_existings()
    run([DOT_GEN])
    dots = [item for item in os.listdir(DOT_PATH) if item.endswith(".dot")]
    Pool().map(partial(convert, ccr_set), dots)
