#!/usr/bin/env bash

declare -a ccrs=(0.125 0.25 0.5 0.75 1 2 3 4 5 6 7 8 9 10)
#declare -a ccrs=(0.125 0.25 0.5 1 2 4 8)

rm -r resources/workflows/random*

MrWSI/utils/randag.py gen_dot
for ccr in ${ccrs[@]}
do
    MrWSI/utils/randag.py dot2dax $ccr
done

for ccr in ${ccrs[@]}
do
    #python setup.py build_ext --inplace
    if [ ! "$(ls -A ./results)" ]; then
        rm results/*
    fi
    echo "solve wrks with ccr=$ccr"
    python test.py resources/workflows/random_$ccr > res_ar/$ccr.res
    #python test_mrkt.py resources/workflows/random_$ccr > res_ar/$ccr.res
done
