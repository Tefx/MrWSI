export LD_PRELOAD=/usr/lib/libjemalloc.so.2
export CC=clang
export LD_LIBRARY_PATH=$(pwd)/MrWSI/core:$LD_LIBRARY_PATH

workon MRSched
