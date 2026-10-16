cmake_minimum_required(VERSION 3.14)
project(OpenGNC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- FetchContent for dependencies ---
include(FetchContent)

# 1. Fetch Eigen 3.4.0 (Lightweight header-only)
message(STATUS "Fetching Eigen 3.4.0...")
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)

FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
  target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})
endif()

if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

# 2. Fetch Asio (Header-only networking)
message(STATUS "Fetching Asio 1.30.2...")
FetchContent_Declare(
    asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG        asio-1-30-2
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(asio)
if(NOT asio_POPULATED)
    FetchContent_Populate(asio)
endif()

# 3. Fetch pybind11 for Python integration (optional)
option(BUILD_PYTHON_BINDINGS "Build Python extension bindings via pybind11" OFF)
if(BUILD_PYTHON_BINDINGS)
    message(STATUS "Fetching pybind11...")
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# --- Header-only Library ---
add_library(opengnc INTERFACE)
target_include_directories(opengnc INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
    $<BUILD_INTERFACE:${asio_SOURCE_DIR}/asio/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(opengnc INTERFACE Eigen3::Eigen)
target_compile_definitions(opengnc INTERFACE ASIO_STANDALONE)

if(WIN32)
    target_compile_definitions(opengnc INTERFACE _WIN32_WINNT=0x0601)
    target_link_libraries(opengnc INTERFACE ws2_32 mswsock)
endif()

# --- Tests ---
option(BUILD_TESTING "Build C++ unit tests" ON)
if(BUILD_TESTING)
    enable_testing()
    
    # Simplified test runner for now:
    add_executable(test_runner cpp/tests/test_runner.cpp)
    target_link_libraries(test_runner PRIVATE opengnc)
    
    if(WIN32)
        # Inherited from opengnc interface
    endif()
    
    add_test(NAME AllTests COMMAND test_runner)
endif()

# --- Python Extension Module ---
if(BUILD_PYTHON_BINDINGS AND pybind11_FOUND)
    pybind11_add_module(opengnc_py cpp/src/bindings.cpp)
    target_link_libraries(opengnc_py PRIVATE opengnc)
endif()

# --- Benchmarks ---
# add_executable(benchmarks cpp/benchmarks/benchmark_filters.cpp)
# target_link_libraries(benchmarks PRIVATE opengnc)
