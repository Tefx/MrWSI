#!/usr/bin/env python3

import networkx as nx
import os.path
from math import ceil
from random import gauss, uniform, choice
import json

# CPU_GAUSS_MU = 0
# CPU_GAUSS_SIGMA = 1
# CPU_LIMIT = 1

# MEM_GAUSS_MU = 200
# MEM_GAUSS_SIGMA = 500
# MEM_LIMIT = 1024

CPU_GAUSS_MU = 0
CPU_GAUSS_SIGMA = 2
CPU_LIMIT = 4

MEM_GAUSS_MU = 1024
MEM_GAUSS_SIGMA = 4096
MEM_LIMIT = 7680


def generate_task_demand():
    cores = abs(round(gauss(CPU_GAUSS_MU, CPU_GAUSS_SIGMA), 2))
    memory = ceil(abs(gauss(MEM_GAUSS_MU, MEM_GAUSS_SIGMA)))
    return min(cores, CPU_LIMIT), min(memory, MEM_LIMIT), 0


def convert_dot(dot_path, out_dir):
    with open(dot_path) as f:
        dag_id = os.path.basename(dot_path)[:-4]
        dag = nx.DiGraph(nx.nx_agraph.read_dot(dot_path))

    tasks = {}
    # ccr = choice([0.1, 1, 10])
    ccr = 10
    for task_id in dag:
        tasks[task_id] = {
            "runtime": ceil(int(dag.node[task_id]["size"]) / 10e8),
            "demands": generate_task_demand(),
            "prevs": {
                p: int(int(dag[p][task_id]["size"]) / 309.6 * ccr)
                for p in dag.predecessors(task_id)
            }
        }

    with open(os.path.join(out_dir, dag_id + ".wrk"), "w") as f:
        json.dump(tasks, f, indent=2)


if __name__ == "__main__":
    from sys import argv
    convert_dot(argv[1], argv[2])
