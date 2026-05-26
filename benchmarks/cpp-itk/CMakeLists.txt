cmake_minimum_required(VERSION 3.20)
project(StackProcessingItkBenchmarks LANGUAGES CXX)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(benchmark_itk benchmark_itk.cxx)
target_compile_features(benchmark_itk PRIVATE cxx_std_17)
target_link_libraries(benchmark_itk PRIVATE ${ITK_LIBRARIES})
