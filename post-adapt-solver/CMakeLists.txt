cmake_minimum_required(VERSION 3.10)
project(post_adapt_solver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_executable(solver solver.cpp)
add_executable(test_correctness test_correctness.cpp)
add_executable(search_ratio search_ratio.cpp)
add_executable(search_ratio_graph search_ratio_graph.cpp)
