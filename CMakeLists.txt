cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(fluid-2d LANGUAGES CXX CUDA)

add_executable(fluid-2d-app app.hpp density_grid_renderer.cuh fluid_solver.hpp fluid_solver_cpu.hpp fluid_solver_gpu.cuh grid.hpp grid_renderer.hpp kernel_launcher.hpp pitched_buffer.hpp simulation.hpp velocity_grid_renderer.cuh app.cpp density_grid_renderer.cu fluid_solver_cpu.cpp fluid_solver_gpu.cu grid_renderer.cpp main.cpp simulation.cpp velocity_grid_renderer.cu)

target_compile_features(fluid-2d-app PUBLIC cxx_std_11)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

find_package(SFML REQUIRED network audio graphics window system)
if(SFML_FOUND)
  include_directories(${SFML_INCLUDE_DIR})
  target_link_libraries(fluid-2d-app ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})
endif()