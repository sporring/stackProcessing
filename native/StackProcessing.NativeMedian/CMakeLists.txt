cmake_minimum_required(VERSION 3.16)

project(StackProcessingNativeMedian LANGUAGES CXX)

if(NOT DEFINED STACKPROCESSING_NATIVE_OUTPUT_DIR)
    set(STACKPROCESSING_NATIVE_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../lib")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)

add_library(spnth SHARED MedianNthElement.cpp)
target_compile_features(spnth PRIVATE cxx_std_17)
target_link_libraries(spnth PRIVATE PkgConfig::FFTW3F)

set_target_properties(
    spnth
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${STACKPROCESSING_NATIVE_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${STACKPROCESSING_NATIVE_OUTPUT_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${STACKPROCESSING_NATIVE_OUTPUT_DIR}"
)

if(MSVC)
    target_compile_options(spnth PRIVATE /O2)
else()
    target_compile_options(spnth PRIVATE -O3)
endif()
