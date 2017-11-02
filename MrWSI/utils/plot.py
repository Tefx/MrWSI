import os.path
import pygraphviz as pgv
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from MrWSI.core.problem import COMM_INPUT
import numpy as np


def plot_cmp_results(log, field, typ="box", base=0):
    fig, ax = plt.subplots(figsize=(6, 4))
    # for alg, res in log.items():
    # ax.plot(res[field], label=alg)
    labels = log.keys()
    data = []
    for alg in labels:
        data.append(log[alg][field])
    if typ == "hist":
        if base == 0:
            ax.hist(data, bins=1000, label=labels,
                    histtype="step", normed=False, cumulative=True)
            ax.set_xlim(0, 1)
        else:
            ax.hist(data, bins=1000, label=labels,
                    histtype="step", normed=True, cumulative=True)
        ax.legend(loc="lower right")
    elif typ == "box":
        ax.axhline(y=base)
        ax.boxplot(data, labels=labels, showfliers=False, whis=[10, 90])
        # for i, y in enumerate(data):
            # x = np.random.normal(i + 1, 0.01 * len(data), len(y))
            # ax.plot(x, y, "k.", alpha=0.2)
    fig.tight_layout()
    plt.savefig(os.path.join(".", "{}.png".format(field)), dpi=200)


def plot_usage(platform, dim, name):
    fig, axes = plt.subplots(
        len(platform.machines),
        sharex=True,
        figsize=(20, 3 * len(platform.machines)))
    if len(platform.machines) == 1:
        axes = [axes]

    for ax, machine in zip(axes, platform.machines):
        usages = machine.usages(dim)
        capacity = machine.vm_type.capacities[dim]
        sts = []
        points = []
        for st, usage in usages:
            points.extend([usage, usage])
            sts.extend([st, st])
        sts = sts[1:-1]
        points = points[:-2]
        ax.grid(True)
        # print("A")
        # print(len(sts), sts)
        # print(len(points), points)
        ax.fill_between(sts, points, 0, facecolor="green")
        ax.set_ylim(0, capacity / 1000.0)
        ax.set_ylabel(dim)

    axes[-1].set_xlabel("Times (s)")
    fig.tight_layout()
    plt.savefig(os.path.join(".", "{}.{}.png".format(name, dim)))


def draw_dag(problem, path):
    graph = pgv.AGraph(directed=True)
    for task in problem.tasks:
        runtime = task.mean_runtime()
        graph.add_node(task, label="{} <{}>".format(task, runtime))
        for comm in task.communications(COMM_INPUT):
            graph.add_edge(
                comm.from_task,
                comm.to_task,
                label="<{}>".format(comm.mean_runtime()))
    graph.draw(path, prog="dot")
