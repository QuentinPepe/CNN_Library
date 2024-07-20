#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <immintrin.h>
#include <cmath>

namespace nnm {

    class Vector {
    private:
        alignas(32) std::vector<float> data;

    public:
        explicit Vector(size_t size) : data(size) {}

        Vector(size_t size, float initial_value) : data(size, initial_value) {}

        Vector(std::initializer_list<float> init) : data(init) {}

        float &operator[](size_t index) {
            return data[index];
        }

        const float &operator[](size_t index) const {
            return data[index];
        }

        [[nodiscard]] size_t size() const {
            return data.size();
        }

        Vector operator+(const Vector &other) const {
            if (size() != other.size()) {
                throw std::invalid_argument("Vector sizes do not match for addition");
            }

            Vector result(size());
            size_t i = 0;

            // Use AVX2 operations for blocks of 8 elements
            for (; i + 8 <= size(); i += 8) {
                __m256 a = _mm256_loadu_ps(&data[i]);
                __m256 b = _mm256_loadu_ps(&other.data[i]);
                __m256 sum = _mm256_add_ps(a, b);
                _mm256_storeu_ps(&result.data[i], sum);
            }

            // Handle the remaining elements
            for (; i < size(); ++i) {
                result[i] = data[i] + other[i];
            }

            return result;
        }

        [[nodiscard]] float dot(const Vector &other) const {
            if (size() != other.size()) {
                throw std::invalid_argument("Vector sizes do not match for dot product");
            }

            __m256 sum = _mm256_setzero_ps();
            size_t i = 0;

            // Use AVX2 operations for blocks of 8 elements
            for (; i + 8 <= size(); i += 8) {
                __m256 a = _mm256_loadu_ps(&data[i]);
                __m256 b = _mm256_loadu_ps(&other.data[i]);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
            }

            // Sum the results from AVX2
            float partial_sum[8];
            _mm256_storeu_ps(partial_sum, sum);
            float dot_product = partial_sum[0] + partial_sum[1] + partial_sum[2] + partial_sum[3] +
                                partial_sum[4] + partial_sum[5] + partial_sum[6] + partial_sum[7];

            // Handle the remaining elements
            for (; i < size(); ++i) {
                dot_product += data[i] * other[i];
            }

            return dot_product;
        }

        [[nodiscard]] float norm() const {
            return std::sqrt(this->dot(*this));
        }

        Vector operator*(float scalar) const {
            Vector result(size());
            size_t i = 0;

            // Use AVX2 operations for blocks of 8 elements
            __m256 s = _mm256_set1_ps(scalar);
            for (; i + 8 <= size(); i += 8) {
                __m256 a = _mm256_loadu_ps(&data[i]);
                __m256 product = _mm256_mul_ps(a, s);
                _mm256_storeu_ps(&result.data[i], product);
            }

            // Handle the remaining elements
            for (; i < size(); ++i) {
                result[i] = data[i] * scalar;
            }

            return result;
        }

        void fill(float value) {
            size_t i = 0;

            // Use AVX2 operations for blocks of 8 elements
            __m256 v = _mm256_set1_ps(value);
            for (; i + 8 <= size(); i += 8) {
                _mm256_storeu_ps(&data[i], v);
            }

            // Handle the remaining elements
            for (; i < size(); ++i) {
                data[i] = value;
            }
        }
    };

} // namespace nnm
