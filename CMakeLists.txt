cmake_minimum_required (VERSION 3.18)

set(CMAKE_CUDA_HOST_COMPILER "NVHPC")

project (HelloCuda LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")

add_compile_options(-cuda)
add_link_options(-cuda)
add_executable(hello_cuda hello_cuda.cpp)

