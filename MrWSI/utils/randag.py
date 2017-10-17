#!/usr/bin/env python3
import os
import os.path
from subprocess import run
from multiprocessing import Pool
from glob import glob

from dot2wrk import convert_dot

DOT_PATH = "resources/daggen/gen"
WRK_PATH = "resources/workflows/random_tiny"
DOT_GEN = "resources/daggen/gen.py"


def rm_existings():
    for item in glob(os.path.join(DOT_PATH, "*")):
        os.remove(item)
    for item in glob(os.path.join(WRK_PATH, "*")):
        os.remove(item)


def convert(dot):
    print("DOT => WRK", dot)
    convert_dot(os.path.join(DOT_PATH, dot), WRK_PATH)


if __name__ == "__main__":
    rm_existings()
    run([DOT_GEN])
    dots = [item for item in os.listdir(DOT_PATH) if item.endswith(".dot")]
    Pool().map(convert, dots)
