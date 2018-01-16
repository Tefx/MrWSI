#!/usr/bin/env bash

declare -a ccrs=(0.125 0.25 0.5 0.75 1 2 3 4 5 6 7 8 9 10)
#declare -a ccrs=(0.25 0.5 1 2 )

rm -r resources/workflows/random*

MrWSI/utils/randag.py gen_dot
for ccr in ${ccrs[@]}
do
    MrWSI/utils/randag.py dot2dax $ccr
done

if [ ! "$(ls -A ./results)" ]; then
    rm results/*
fi

#python setup.py build_ext --inplace
python test_mrkt_batch.py ${ccrs[@]}
