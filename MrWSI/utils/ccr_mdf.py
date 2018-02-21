#!/usr/bin/env python

import json
from statistics import mean

BW = 76546048 / 1024


def ccr_modify(wrk, new_path, ccr):
    print("Change {} ccr to {}".format(wrk, ccr))
    with open(wrk) as f:
        tasks = json.load(f)

    rts = []
    cts = []
    for task in tasks.values():
        rts.append(task["runtime"])
        cts.extend(task["prevs"].values())
    crt_ccr = mean(cts) / BW / mean(rts)
    factor = ccr / crt_ccr
    for task in tasks.values():
        for prev in task["prevs"]:
            task["prevs"][prev] *= factor

    with open(new_path, "w") as f:
        json.dump(tasks, f)


def dir_mdf(wrk_dir, new_dir, ccr):
    import os
    import os.path

    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    for wrk in os.listdir(wrk_dir):
        if wrk.endswith(".wrk"):
            ccr_modify(os.path.join(wrk_dir, wrk),
                       os.path.join(new_dir, wrk),
                       float(ccr))


if __name__ == '__main__':
    from sys import argv
    wrk_dir, new_dir, ccr = argv[1:]
    dir_mdf(wrk_dir, new_dir, ccr)
