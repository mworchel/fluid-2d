#pragma once

#include <cuda_runtime.h>

template<typename T>
class element_accessor
{
public:
	element_accessor(T* buffer, size_t const pitch)
		: m_buffer(buffer)
		, m_pitch(pitch)
	{
	}

	inline __device__ T& at(size_t x, size_t y)
	{
		return *(reinterpret_cast<T*>(reinterpret_cast<char*>(m_buffer) + y * m_pitch) + x);
	}

	inline __device__ T const& at(size_t x, size_t y) const
	{
		return *(reinterpret_cast<T*>(reinterpret_cast<char*>(m_buffer) + y * m_pitch) + x);
	}

private:
	T* m_buffer;
	size_t m_pitch;
};

template<typename T>
class pitched_buffer
{
public:
	pitched_buffer(size_t width, size_t height)
		: m_width(width)
		, m_height(height)
	{
		cudaMallocPitch(&m_buffer, &m_pitch, byte_width(), m_height);
	}

	~pitched_buffer()
	{
		cudaFree(m_buffer);
	}

	void clear(T value)
	{
	cudaMemset2D(m_buffer, m_pitch, value, byte_width(), m_height);
	}

	T* buffer()
	{
		return m_buffer;
	}

	T const* buffer() const
	{
		return m_buffer;
	}

	size_t pitch() const
	{
		return m_pitch;
	}
	
	element_accessor<T> accessor()
	{
		return element_accessor<T>{ m_buffer, m_pitch };
	}

	element_accessor<T> const accessor() const
	{
		return element_accessor<T>{ m_buffer, m_pitch };
	}

	size_t width()
	{
	return m_width;
	}

	size_t height()
	{
		return m_height;
	}

	size_t byte_width()
	{
		return sizeof(T) * m_width;
	}

private:
	size_t m_width;
	size_t m_height;
	T*     m_buffer;
	size_t m_pitch;
};
