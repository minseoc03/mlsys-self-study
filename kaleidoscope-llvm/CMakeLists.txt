cmake_minimum_required(VERSION 3.15)
project(Kaleidoscope LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set path to LLVM
set(LLVM_DIR "/opt/homebrew/opt/llvm/lib/cmake/llvm")

# Find LLVM
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Get required include and library paths
include_directories(${LLVM_INCLUDE_DIRS})

# Source and header paths
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Create executable
add_executable(kaleidoscope ${SOURCES})

# Add our own include directory
target_include_directories(kaleidoscope PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

# Add required LLVM components
llvm_map_components_to_libnames(llvm_libs
  core
  executionengine
  mcjit
  native
  orcjit
  passes
  support
  target
)

# Link LLVM to the executable
target_link_libraries(kaleidoscope PRIVATE ${llvm_libs})

# Ensure compatibility with C++17
target_compile_features(kaleidoscope PRIVATE cxx_std_17)
