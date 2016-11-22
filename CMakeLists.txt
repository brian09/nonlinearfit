cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)



# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 --cudart static --use_fast_math -std=c++11 --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_37,code=compute_37 -gencode arch=compute_50,code=compute_50 -gencode arch=compute_52,code=compute_52 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52
    )











# Specify include directories
#include_directories(
#    kernels
#    utility
#    )

# Specify library paths
#link_directories(
#    /opt/foobar/lib
#    /opt/joestuff/lib
#    )

# For compilation ...
# Specify target & source files to compile it from

set(
    CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_HOME_DIRECTORY}/lib
    )

cuda_add_library(
	nonlinearfit
	nonlinearfit.cu
	STATIC
    )

