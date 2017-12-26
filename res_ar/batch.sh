#!/usr/bin/env bash

for ccr in 0.125 0.25 0.5 0.75 1 2 3 4 5 6 7 8 9 10
#for ccr in 0.125 0.25 0.5 0.75 1 2
do
    MrWSI/utils/randag.py $ccr
    python setup.py build_ext --inplace
    rm ./results/*
    #python test.py > res_ar/$ccr.res
    python test_mrkt.py > res_ar/$ccr.res
done
