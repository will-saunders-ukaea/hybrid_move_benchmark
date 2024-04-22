cmake_minimum_required(VERSION 3.24)

project(
  NESO-Workspace-hybrid-move
  VERSION 0.0.1
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

# find MPI
find_package(MPI REQUIRED)
find_package(NESO-Particles REQUIRED)

# put all targets in bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
  message(STATUS "Set CMAKE_BUILD_TYPE=RelWithDebInfo")
endif()

# add_compile_options(-O3) set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -DNDEBUG
# -Ofast") set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -DNDEBUG -O3
# -DNESO_PARTICLES_PROFILING_REGION")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -DNDEBUG -O3")
message(STATUS ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})

set(EXECUTABLE hybrid_move_bench)
set(EXECUTABLE_SOURCE main.cpp)

add_executable(${EXECUTABLE} ${EXECUTABLE_SOURCE})

target_include_directories(${EXECUTABLE} PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(
  ${EXECUTABLE}
  PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
target_link_libraries(${EXECUTABLE} PUBLIC NESO-Particles::NESO-Particles)

# define the test executable as a sycl target
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${EXECUTABLE_SOURCE})
