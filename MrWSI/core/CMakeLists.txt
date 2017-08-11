project(MrWSI)

cmake_minimum_required(VERSION 3.7)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer -Ofast -march=native -ffast-math -funroll-loops")
#set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -ffast-math -funroll-loops")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${MrWSI_SOURCE_DIR})

subdirs(src)
