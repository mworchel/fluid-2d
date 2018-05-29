#pragma once

#include <vector>

template<typename T>
class grid {
public:
    grid(size_t rows, size_t cols, T initial_value)
        : m_rows(rows)
        , m_cols(cols)
        , m_data(rows * cols, initial_value)
        , m_initial_value(initial_value) {}

    size_t rows() const {
        return m_rows;
    }

    size_t cols() const {
        return m_rows;
    }

    const T* data() const {
        return m_data.data();
    }

    T* data() {
        return m_data.data();
    }

    T operator() (const size_t y, const size_t x) const {
        return m_data[y * m_cols + x];
    }

    T& operator() (const size_t y, const size_t x) {
        return m_data[y * m_cols + x];
    }

    auto begin() {
        return std::begin(m_data);
    }

    auto end() {
        return std::end(m_data);
    }

    auto cbegin() const {
        return std::cbegin(m_data);
    }

    auto cend() const {
        return std::cend(m_data);
    }

private:
    T              m_initial_value;
    size_t         m_rows;
    size_t         m_cols;
    std::vector<T> m_data;
};
