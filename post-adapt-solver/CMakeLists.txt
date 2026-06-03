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
add_executable(search_ratio search_ratio.cpp)
add_executable(search_ratio_graph search_ratio_graph.cpp)
add_executable(search_ratio_exhaustive search_ratio_exhaustive.cpp)
add_executable(search_symmetric_gap search_symmetric_gap.cpp)
add_executable(optimize_ratio_sa optimize_ratio_sa.cpp)
add_executable(optimize_ratio_int optimize_ratio_int.cpp)
add_executable(topology_search topology_search.cpp)
