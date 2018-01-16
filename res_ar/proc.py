import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ccrs = [
    0.125, 0.25, 0.5, 0.75, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 10
    # 0.125, 0.25, 0.5, 1, 2, 4, 8
    # 0.25, 0.5, 1, 2
]
# ccrs = range(21)
res = {}

for ccr in ccrs:
    with open("res_ar/{}.res".format(ccr)) as f:
        for line in f:
            alg, value, _ = line.split()
            if alg not in res:
                res[alg] = []
            res[alg].append(float(value))

base = "EFT[fcfs]"
# base = "CAN8.6.t0(MCPU)"
# for alg, values in res.items():
    # if alg == base:
        # continue
    # else:
        # res[alg] = [x / y for x, y in zip(values, res[base])]

ignore_list = [
    # "EFT[fcfs]",
    # "CAEFT(PU)",
    # "CA2Fit5(PU)",
    "CAN3(PU)",
    "CAN6.2(PU)",
    "CAN6.2.2(N2PU)",
]
for alg in ignore_list:
    if alg in res:
        del res[alg]

fig, ax = plt.subplots(figsize=(4.8, 3.6))
for alg, values in res.items():
    ax.plot(ccrs, values, lw=0.5, label=alg)
ax.axhline(y=1, lw=0.5, ls='--')
# ax.set_xscale("log", basex=2)
ax.legend()
fig.tight_layout()
plt.savefig("res.png", dpi=300)
