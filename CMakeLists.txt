cmake_minimum_required(VERSION 3.20)
project(vectorforge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Deliberately NOT -march=native or -mavx2 globally. The SIMD kernels carry
# per-function target attributes and are selected at runtime via cpuid, so one
# binary runs correctly on any x86-64 host and still uses AVX2 where present.
# A global -mavx2 would let the compiler hoist VEX instructions into the
# dispatcher itself and crash before the feature check ever runs.
add_compile_options(-O3 -fno-math-errno -Wall -Wextra -Wno-unused-parameter)

# Link the runtime statically on MinGW. Without this the binaries resolve
# libstdc++-6.dll / libgcc_s_seh-1.dll / libwinpthread-1.dll off PATH, and any
# machine with an older 32-bit MinGW installed (C:\MinGW\bin is a common one)
# hands back a 32-bit DLL to a 64-bit process -> 0xC000007B at launch. Static
# linking makes the benchmark binaries runnable by anyone who downloads them,
# which matters more here than the few hundred KB it costs.
if(MINGW)
  add_link_options(-static-libgcc -static-libstdc++ -static)
endif()

add_library(vfcore STATIC
  core/src/distance.cpp
  core/src/hnsw.cpp
)
target_include_directories(vfcore PUBLIC core/include)

find_package(Threads REQUIRED)
target_link_libraries(vfcore PUBLIC Threads::Threads)

# --- tests -------------------------------------------------------------------
enable_testing()

add_executable(test_distance core/tests/test_distance.cpp)
target_link_libraries(test_distance PRIVATE vfcore)
add_test(NAME distance COMMAND test_distance)

add_executable(test_hnsw core/tests/test_hnsw.cpp)
target_link_libraries(test_hnsw PRIVATE vfcore)
add_test(NAME hnsw COMMAND test_hnsw)

# --- benchmarks --------------------------------------------------------------
add_executable(bench_recall core/bench/bench_recall.cpp)
target_link_libraries(bench_recall PRIVATE vfcore)
