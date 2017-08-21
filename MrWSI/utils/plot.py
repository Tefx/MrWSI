import os.path


def plot_usage(platform, dim, name):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(platform.machines), sharex=True, figsize=(20, 3 * len(platform.machines)))
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
