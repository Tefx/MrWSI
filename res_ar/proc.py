import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ccrs = [
    0.125, 0.25, 0.5, 0.75, 1, 2,
    3, 4, 5, 6, 7, 8, 9, 10
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

for alg, values in res.items():
    if alg == "EFT[fcfs]":
        continue
    else:
        res[alg] = [x / y for x, y in zip(values, res["EFT[fcfs]"])]
del res["EFT[fcfs]"]
# del res["EFT[fair]"]
# del res["CAN2(PU)"]
# del res["CAN3.2(PU)"]
# del res["CAN3.2.1(PU)"]
del res["CAN6(PU)"]
# del res["CA2Fit5(PU)"]
# del res["CA3.3(PU)"]

fig, ax = plt.subplots(figsize=(6, 4))
for alg, values in res.items():
    ax.plot(ccrs, values, lw=0.5, label=alg)
ax.axhline(y=1, lw=0.5, ls='--')
# ax.set_xscale("log", basex=2)
ax.legend()
fig.tight_layout()
plt.savefig("res.png", dpi=300)
