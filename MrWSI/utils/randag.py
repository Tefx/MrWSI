#!/usr/bin/env python3
import os
import os.path
from subprocess import run
from multiprocessing import Pool
from glob import glob
from sys import argv
from functools import partial
import fire

from dot2wrk import convert_dot

DOT_GEN = "resources/daggen/gen.py"
DOT_PATH = "resources/daggen/gen"
WRK_PATH = "resources/workflows/random"
CCR_SET = [8]


def rm_existings():
    for item in glob(os.path.join(DOT_PATH, "*")):
        os.remove(item)
    for item in glob(os.path.join(WRK_PATH, "*")):
        os.remove(item)


def convert(ccr, path, dot):
    print("DOT => WRK (CCR={})".format(ccr), dot)
    convert_dot(os.path.join(DOT_PATH, dot), path, ccr)


def gen_dot():
    rm_existings()
    run([DOT_GEN])


def dot2dax(ccr_set=CCR_SET, main_dir=False):
    if not isinstance(ccr_set, list):
        ccr_set = [ccr_set]
    dots = [item for item in os.listdir(DOT_PATH) if item.endswith(".dot")]
    if not main_dir:
        path = "{}_{}".format(WRK_PATH, "_".join(map(str, ccr_set)))
    else:
        path=WRK_PATH
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        for item in glob(os.path.join(path, "*")):
            os.remove(item)
    Pool().map(partial(convert, ccr_set, path), dots)

def gen_dax(ccr_set=CCR_SET):
    gen_dot()
    dot2dax(ccr_set, True)

if __name__ == "__main__":
    fire.Fire()
