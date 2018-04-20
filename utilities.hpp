#pragma once

#include "grid.hpp"
#include "pitched_buffer.hpp"

#include <cuda_runtime.h>

template<typename T>
inline void copy(pitched_buffer<T>& destionation_buffer, grid<T> const& source_grid)
{
  cudaMemcpy2D(destionation_buffer.buffer(), destionation_buffer.pitch(),
               source_grid.data(), sizeof(T) * source_grid.cols(),
               destionation_buffer.byte_width(), destionation_buffer.height(), 
               cudaMemcpyHostToDevice);
}

template<typename T>
inline void copy(grid<T>& grid, pitched_buffer<T> const& buffer)
{
  cudaMemcpy2D(grid.data(), sizeof(T) * grid.cols(),
               buffer.buffer(), buffer.pitch(),
               sizeof(T) * grid.cols(), grid.rows(), 
               cudaMemcpyDeviceToHost);
}

template<typename T>
inline void copy(pitched_buffer<T>& destionation, pitched_buffer<T> const& source)
{
  cudaMemcpy2D(destionation.buffer(), destionation.pitch(),
               source.buffer(), source.pitch(),
               destionation.byte_width(), destionation.height(), 
               cudaMemcpyDeviceToDevice);
}