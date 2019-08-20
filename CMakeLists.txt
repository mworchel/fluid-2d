cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(fluid-2d LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(SFML REQUIRED network audio graphics window system)

set(APP_FILES "src/app.cpp"
              "src/app.hpp"
              "src/density_grid_renderer.cu"
              "src/density_grid_renderer.cuh"
              "src/fluid_solver.hpp"
              "src/fluid_solver_cpu.cpp"
              "src/fluid_solver_cpu.hpp"
              "src/fluid_solver_gpu.cu"
              "src/fluid_solver_gpu.cuh"
              "src/gpu_buffer.hpp"
              "src/grid.hpp"
              "src/grid_renderer.cpp"
              "src/grid_renderer.hpp"
              "src/kernel_launcher.hpp"
              "src/linear_buffer.hpp"
              "src/main.cpp"
              "src/pitched_buffer.hpp"
              "src/simulation.cpp"
              "src/simulation.hpp"
              "src/utilities.hpp"
              "src/velocity_grid_renderer.cu"
              "src/velocity_grid_renderer.cuh"
              ) 

add_executable(fluid-2d-app ${APP_FILES})

if(SFML_FOUND)
  include_directories(${SFML_INCLUDE_DIR})
  target_link_libraries(fluid-2d-app ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})
endif()