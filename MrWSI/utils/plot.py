import os.path
import pygraphviz as pgv
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from MrWSI.core.problem import COMM_INPUT
import numpy as np
from matplotlib.ticker import PercentFormatter


def plot_cmp_results(log, field, typ="box", base=0):
    fig, ax = plt.subplots(figsize=(6, 4))
    # for alg, res in log.items():
    # ax.plot(res[field], label=alg)
    labels = log.keys()
    data = []
    for alg in labels:
        data.append([0] + log[alg][field])
    if typ == "hist":
        if base == 0:
            ax.hist(data, bins=1000, label=labels,
                    histtype="step", normed=False, cumulative=True)
            ax.set_xlim(0, 1)
        else:
            ax.axvline(x=1, lw=0.5)
            ax.hist(data, bins=1000, label=labels,
                    histtype="step", density=True, cumulative=-1, lw=0.5)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.minorticks_on()
            ax.set_xlim(xmin=1)
            # ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            # ax.set_xscale("log")
            plt.grid(b=True, which="major", linestyle="-", linewidth="0.5")
            plt.grid(b=True, which="minor", linestyle="--", linewidth="0.3")
        ax.legend(loc="upper right")
        # ax.legend(loc="upper left")
    elif typ == "box":
        ax.axhline(y=base)
        ax.boxplot(data, labels=labels, showmeans=True, showfliers=False, whis=[10, 90])
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
        graph.add_node(task, label="{} <{}>".format(task, int(runtime/100)))
        for comm in task.communications(COMM_INPUT):
            graph.add_edge(
                comm.from_task,
                comm.to_task,
                label="<{}>".format(int(comm.mean_runtime()/100)))
    graph.draw(path, prog="dot")
