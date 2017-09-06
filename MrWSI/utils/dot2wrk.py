#!/usr/bin/env python3

import networkx as nx
import os.path
from math import ceil
from random import gauss
import json

MAX_CPU = 4
MAX_MEM = 16384


def generate_task_demand(max_cpu, max_memory):
    cores = abs(round(gauss(0, 2), 2))
    if cores > max_cpu: cores = max_cpu

    memory = ceil(abs(gauss(1024, 10240)))
    if memory > max_memory: memory = max_memory

    return cores, memory, 0


def read_dot(dot_path, out_dir):
    with open(dot_path) as f:
        dag_id = os.path.basename(dot_path)[:-4]
        dag = nx.DiGraph(nx.nx_agraph.read_dot(dot_path))

    tasks = {}
    for task_id in dag:
        tasks[task_id] = {
            "runtime": ceil(int(dag.node[task_id]["size"]) / 10e9),
            "demands": generate_task_demand(MAX_CPU, MAX_MEM),
            "prevs":
            {p: int(dag[p][task_id]["size"])
             for p in dag.predecessors(task_id)}
        }

    with open(os.path.join(out_dir, dag_id + ".wrk"), "w") as f:
        json.dump(tasks, f, indent=2)


if __name__ == "__main__":
    from sys import argv
    read_dot(argv[1], argv[2])
