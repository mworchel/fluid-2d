cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(fluid-2d LANGUAGES CXX CUDA)

file(GLOB APP_SOURCES "./src/*.hpp" "./src/*.cpp" "./src/*.cuh" "./src/*.cu") 

add_executable(fluid-2d-app ${APP_SOURCES})

target_compile_features(fluid-2d-app PUBLIC cxx_std_11)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

find_package(SFML REQUIRED network audio graphics window system)
if(SFML_FOUND)
  include_directories(${SFML_INCLUDE_DIR})
  target_link_libraries(fluid-2d-app ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})
endif()