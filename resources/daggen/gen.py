#!/usr/bin/env python3

import sh
from random import choice
import os.path

path = "resources/daggen"

daggen = sh.Command(os.path.join(path, "daggen"))
dot = sh.dot

# n_set = [25, 50, 75, 100]
n_set = [20]
jump_set = [1, 3, 5]
fat_set = [0.2, 0.4, 0.6, 0.8]
# fat_set = [0.8]
regularity_set = [0.2, 0.4, 0.6, 0.8]
density_set = [0.2, 0.4, 0.6, 0.8]
# density_set = [0.8]

for i in range(100):
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
                density=density))
        f.write(source)
    dot(name, "-Tpng", "-O")
