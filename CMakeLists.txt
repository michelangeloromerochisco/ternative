cmake_minimum_required(VERSION 3.18)
project(ternative LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(TERNATIVE_CUDA "Enable CUDA backend" OFF)
option(TERNATIVE_BUILD_TESTS "Build tests" ON)

# OpenBLAS detection
if(WIN32)
    find_library(OPENBLAS_LIB
        NAMES openblas libopenblas
        PATHS
            "C:/Program Files/OpenBLAS/lib"
            "C:/Program Files (x86)/OpenBLAS/lib"
            "C:/vcpkg/installed/x64-windows/lib"
            "C:/vcpkg/installed/x64-windows-static/lib"
            "C:/tools/openblas/lib"
            "C:/openblas/lib"
            "C:/Users/$ENV{USERNAME}/OpenBLAS/lib"
    )
    find_path(OPENBLAS_INCLUDE
        NAMES cblas.h
        PATHS
            "C:/Program Files/OpenBLAS/include"
            "C:/Program Files (x86)/OpenBLAS/include"
            "C:/vcpkg/installed/x64-windows/include"
            "C:/vcpkg/installed/x64-windows-static/include"
            "C:/tools/openblas/include"
            "C:/openblas/include"
            "C:/Users/$ENV{USERNAME}/OpenBLAS/include"
    )
else()
    find_library(OPENBLAS_LIB
        NAMES openblas libopenblas
        PATHS /usr/lib /usr/local/lib /opt/OpenBLAS/lib
    )
    find_path(OPENBLAS_INCLUDE
        NAMES cblas.h
        PATHS /usr/include /usr/local/include /opt/OpenBLAS/include
    )
endif()

if(OPENBLAS_LIB AND OPENBLAS_INCLUDE)
    message(STATUS "OpenBLAS found: ${OPENBLAS_LIB}")
    message(STATUS "OpenBLAS include: ${OPENBLAS_INCLUDE}")
    set(TERNATIVE_OPENBLAS_FOUND TRUE)
else()
    message(STATUS "OpenBLAS NOT found — using fallback matmul")
    set(TERNATIVE_OPENBLAS_FOUND FALSE)
endif()

# Fetch dependencies
include(FetchContent)

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(json)

# Sources
set(SOURCES
    src/gguf.cpp
    src/gguf_write.cpp
    src/tensor.cpp
    src/model.cpp
    src/ops_cpu.cpp
    src/sampler.cpp
    src/tokenizer.cpp
    src/server.cpp
    src/cache.cpp
    src/stream.cpp
    src/request.cpp
)

set(HEADERS
    include/ternative/gguf.h
    include/ternative/tensor.h
    include/ternative/model.h
    include/ternative/ops.h
    include/ternative/sampler.h
    include/ternative/tokenizer.h
    include/ternative/server.h
    include/ternative/cache.h
    include/ternative/stream.h
    include/ternative/request.h
)

# Main executable
add_executable(ternative src/main.cpp ${SOURCES} ${HEADERS})
target_include_directories(ternative PRIVATE include)
target_link_libraries(ternative PRIVATE nlohmann_json::nlohmann_json)

# Platform
if(TERNATIVE_OPENBLAS_FOUND)
    target_include_directories(ternative PRIVATE ${OPENBLAS_INCLUDE})
    target_link_libraries(ternative PRIVATE ${OPENBLAS_LIB})
    target_compile_definitions(ternative PRIVATE TERNATIVE_USE_OPENBLAS)
endif()

if(WIN32)
    target_compile_definitions(ternative PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    target_link_libraries(ternative PRIVATE ws2_32)
    # MSVC-specific flags — CXX only, never passed to nvcc
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        target_compile_options(ternative PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()
    # /openmp: parallelize loops; /fp:fast: FMA fusion; /O2: full opt
    target_compile_options(ternative PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp /fp:fast /O2>)
endif()

# CUDA
if(TERNATIVE_CUDA)
    # Must be set before enable_language(CUDA) — CMake reads it during compiler detection
    # RTX 3050 = Ampere SM86; SM75 = Turing fallback; SM80 = A100 portability
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    target_sources(ternative PRIVATE cuda/ops_cuda.cu)
    target_compile_definitions(ternative PRIVATE TERNATIVE_USE_CUDA)
    set_target_properties(ternative PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "75;80;86"
    )
    # Link CUDA runtime
    target_link_libraries(ternative PRIVATE CUDA::cudart)
    find_package(CUDAToolkit REQUIRED)
    # CUDA compile flags: fast math, max registers for GEMV kernel
    target_compile_options(ternative PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            --maxrregcount=64
            -O3
        >
    )
endif()

# Tests
if(TERNATIVE_BUILD_TESTS)
    add_executable(ternative_test tests/test_basic.cpp ${SOURCES} ${HEADERS})
    target_include_directories(ternative_test PRIVATE include)
    target_link_libraries(ternative_test PRIVATE nlohmann_json::nlohmann_json)
    if(TERNATIVE_OPENBLAS_FOUND)
        target_include_directories(ternative_test PRIVATE ${OPENBLAS_INCLUDE})
        target_link_libraries(ternative_test PRIVATE ${OPENBLAS_LIB})
        target_compile_definitions(ternative_test PRIVATE TERNATIVE_USE_OPENBLAS)
    endif()
    if(WIN32)
        target_compile_definitions(ternative_test PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
        target_link_libraries(ternative_test PRIVATE ws2_32)
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            target_compile_options(ternative_test PRIVATE /arch:AVX2)
        endif()
        target_compile_options(ternative_test PRIVATE /openmp)
    endif()
endif()
