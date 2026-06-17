cmake_minimum_required(VERSION 3.20)
project(StackProcessingItkBenchmarks LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(
  ITK
  REQUIRED
  COMPONENTS
    ITKCommon
    ITKIOTIFF
    ITKThresholding
    ITKBinaryMathematicalMorphology
    ITKConnectedComponents
    ITKConvolution
    ITKFFT
    ITKSmoothing
)
include(${ITK_USE_FILE})

get_property(_itk_benchmark_includes DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
if(_itk_benchmark_includes)
  list(FILTER _itk_benchmark_includes EXCLUDE REGEX "/Library/Developer/CommandLineTools/SDKs/.*/usr/include$")
  set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES ${_itk_benchmark_includes})
endif()

add_executable(benchmark_itk benchmark_itk.cxx)
target_compile_features(benchmark_itk PRIVATE cxx_std_17)
target_link_libraries(benchmark_itk PRIVATE ${ITK_LIBRARIES})
