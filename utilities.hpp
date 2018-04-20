#pragma once

#include "grid.hpp"
#include "pitched_buffer.hpp"
#include "linear_buffer.hpp"

#include <cuda_runtime.h>

template<typename T>
inline void copy(pitched_buffer<T>& destionation_buffer, grid<T> const& source_grid)
{
  cudaMemcpy2D(destionation_buffer.data(), destionation_buffer.pitch(),
               source_grid.data(), sizeof(T) * source_grid.cols(),
               destionation_buffer.byte_width(), destionation_buffer.height(), 
               cudaMemcpyHostToDevice);
}

template<typename T>
inline void copy(grid<T>& grid, pitched_buffer<T> const& buffer)
{
  cudaMemcpy2D(grid.data(), sizeof(T) * grid.cols(),
               buffer.data(), buffer.pitch(),
               sizeof(T) * grid.cols(), grid.rows(), 
               cudaMemcpyDeviceToHost);
}

template<typename T>
inline void copy(pitched_buffer<T>& destionation, pitched_buffer<T> const& source)
{
  cudaMemcpy2D(destionation.data(), destionation.pitch(),
               source.data(), source.pitch(),
               destionation.byte_width(), destionation.height(), 
               cudaMemcpyDeviceToDevice);
}

template<typename T>
inline void copy(linear_buffer<T>& destionation_buffer, grid<T> const& source_grid)
{
  cudaMemcpy(destionation_buffer.data(), source_grid.data(), destionation_buffer.byte_size(), cudaMemcpyHostToDevice);
}

template<typename T>
inline void copy(grid<T>& grid, linear_buffer<T> const& buffer)
{
  cudaMemcpy(grid.data(), buffer.data(), sizeof(T) * grid.cols() * grid.rows(), cudaMemcpyDeviceToHost);
}

template<typename T>
inline void copy(linear_buffer<T>& destionation, linear_buffer<T> const& source)
{
  cudaMemcpy(destionation.data(), source.data(), destionation.byte_size(), cudaMemcpyDeviceToDevice);
}