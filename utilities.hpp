#pragma once

#include "grid.hpp"
#include "pitched_buffer.hpp"
#include "linear_buffer.hpp"

#include <cuda_runtime.h>

/**
 * Copy data from a grid to a pitched buffer
 */
template<typename T>
inline cudaError_t copy(pitched_buffer<T>& destination_buffer, grid<T> const& source_grid)
{
  return cudaMemcpy2D(destination_buffer.data(), destination_buffer.pitch(),
                      source_grid.data(), sizeof(T) * source_grid.cols(),
                      destination_buffer.byte_width(), destination_buffer.height(),
                      cudaMemcpyHostToDevice);
}

/**
* Copy data from a pitched buffer to a grid
*/
template<typename T>
inline cudaError_t copy(grid<T>& destination_grid, pitched_buffer<T> const& source_buffer)
{
  return cudaMemcpy2D(destination_grid.data(), sizeof(T) * destination_grid.cols(),
                      source_buffer.data(), source_buffer.pitch(),
                      sizeof(T) * destination_grid.cols(), destination_grid.rows(),
                      cudaMemcpyDeviceToHost);
}

/**
* Copy data from a pitched buffer to another pitched buffer
*/
template<typename T>
inline cudaError_t copy(pitched_buffer<T>& destination_buffer, pitched_buffer<T> const& source_buffer)
{
  return cudaMemcpy2D(destination_buffer.data(), destination_buffer.pitch(),
                      source_buffer.data(), source_buffer.pitch(),
                      destination_buffer.byte_width(), destination_buffer.height(),
                      cudaMemcpyDeviceToDevice);
}

/**
* Copy data from a pitched buffer to linear memory without pitch
*/
template<typename T>
inline cudaError_t copy(T* destination, size_t width, size_t height, pitched_buffer<T> const& source_buffer, cudaMemcpyKind kind)
{
  return cudaMemcpy2D(destination, sizeof(T) * width,
                      source_buffer.data(), source_buffer.pitch(),
                      sizeof(T) * width, height,
                      kind);
}

/**
* Copy data from a grid to a linear buffer
*/
template<typename T>
inline cudaError_t copy(linear_buffer<T>& destination_buffer, grid<T> const& source_grid)
{
  return cudaMemcpy(destination_buffer.data(), source_grid.data(), destination_buffer.byte_size(), cudaMemcpyHostToDevice);
}

/**
 * Copy data from a linear buffer to a grid
 */
template<typename T>
inline cudaError_t copy(grid<T>& destination_grid, linear_buffer<T> const& source_buffer)
{
  return cudaMemcpy(destination_grid.data(), source_buffer.data(), sizeof(T) * destination_grid.cols() * destination_grid.rows(), cudaMemcpyDeviceToHost);
}

/**
* Copy data from a linear buffer to  another linear buffer
*/
template<typename T>
inline cudaError_t copy(linear_buffer<T>& destination_buffer, linear_buffer<T> const& source_buffer)
{
  return cudaMemcpy(destination_buffer.data(), source_buffer.data(), destination_buffer.byte_size(), cudaMemcpyDeviceToDevice);
}

/**
* Copy data from a linear buffer to linear memory without pitch
*/
template<typename T>
inline cudaError_t copy(T* destination, size_t width, size_t height, linear_buffer<T> const& source_buffer, cudaMemcpyKind kind)
{
  return cudaMemcpy(destination, source_buffer.data(), sizeof(T) * width * height, kind);
}
