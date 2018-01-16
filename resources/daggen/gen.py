#!/usr/bin/env python3

import sh
from random import choice
import os.path
from multiprocessing import Pool

path = "resources/daggen"

daggen = sh.Command(os.path.join(path, "daggen"))
# dot = sh.dot

num_gen = 1000
# n_set = [25, 50, 75, 100]
n_set = [75]
jump_set = [1, 3, 5]
fat_set = [0.2, 0.4, 0.6, 0.8]
# fat_set = [0.5]
ccr_set = [30]
regularity_set = [0.2, 0.4, 0.6, 0.8]
density_set = [0.2, 0.4, 0.6, 0.8]
# density_set = [0.8]


def gen_dag(i):
    n = choice(n_set)
    jump = choice(jump_set)
    fat = choice(fat_set)
    regularity = choice(regularity_set)
    density = choice(density_set)

    name = "gen/{}.dot".format(
        "_".join(map(str, [i, n, jump, fat, regularity, density])))
    name = os.path.join(path, name)
    print("Generating {}...".format(name))
    with open(name, "w") as f:
        source = str(
            daggen(
                dot=True,
                n=n,
                jump=jump,
                fat=fat,
                regularity=regularity,
                density=density,
                ccr=1
            ))
        f.write(source)
    # dot(name, "-Tpng", "-O")


if __name__ == "__main__":
    Pool().map(gen_dag, range(num_gen))
