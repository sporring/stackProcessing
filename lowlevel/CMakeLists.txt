cmake_minimum_required(VERSION 3.16)

project(StackProcessingLowLevel LANGUAGES CXX)

if(NOT DEFINED STACKPROCESSING_NATIVE_OUTPUT_DIR)
    set(STACKPROCESSING_NATIVE_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/../lib")
endif()

add_library(lowlevel SHARED lowlevel.cpp)
target_compile_features(lowlevel PRIVATE cxx_std_17)

if(WIN32)
    find_package(FFTW3 CONFIG QUIET)

    if(TARGET FFTW3::fftw3f)
        target_link_libraries(lowlevel PRIVATE FFTW3::fftw3f)
    elseif(TARGET unofficial::fftw3::fftw3f)
        target_link_libraries(lowlevel PRIVATE unofficial::fftw3::fftw3f)
    else()
        find_path(FFTW3F_INCLUDE_DIR fftw3.h)
        find_library(FFTW3F_LIBRARY NAMES fftw3f libfftw3f-3)

        if(NOT FFTW3F_INCLUDE_DIR OR NOT FFTW3F_LIBRARY)
            message(FATAL_ERROR "Could not find single-precision FFTW on Windows. Install fftw3 with vcpkg and pass -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake, or set FFTW3F_INCLUDE_DIR and FFTW3F_LIBRARY.")
        endif()

        target_include_directories(lowlevel PRIVATE "${FFTW3F_INCLUDE_DIR}")
        target_link_libraries(lowlevel PRIVATE "${FFTW3F_LIBRARY}")
    endif()
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
    target_link_libraries(lowlevel PRIVATE PkgConfig::FFTW3F)
endif()

set_target_properties(
    lowlevel
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${STACKPROCESSING_NATIVE_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${STACKPROCESSING_NATIVE_OUTPUT_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${STACKPROCESSING_NATIVE_OUTPUT_DIR}"
)

if(MSVC)
    target_compile_options(lowlevel PRIVATE /O2)
else()
    target_compile_options(lowlevel PRIVATE -O3)
endif()
