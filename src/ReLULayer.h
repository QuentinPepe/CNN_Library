#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include <algorithm>
#include <immintrin.h>

namespace nnm {

    class ReLULayer : public Layer<Tensor4D, Tensor4D> {
    public:
        ReLULayer() = default;

        Tensor4D forward(const Tensor4D &x) override {
            Tensor4D relu_output(x.getBatchSize(), x.getChannels(), x.getHeight(), x.getWidth());

            const float *input_data = x.getData().data();
            float *output_data = relu_output.getData().data();
            size_t total_elements = x.getBatchSize() * x.getChannels() * x.getHeight() * x.getWidth();

            // SIMD optimization for AVX2
            size_t i = 0;
            for (; i + 7 < total_elements; i += 8) {
                __m256 input_vec = _mm256_loadu_ps(input_data + i);
                __m256 zero_vec = _mm256_setzero_ps();
                __m256 result_vec = _mm256_max_ps(input_vec, zero_vec);
                _mm256_storeu_ps(output_data + i, result_vec);
            }

            // Handle remaining elements
            for (; i < total_elements; ++i) {
                output_data[i] = std::max(0.0f, input_data[i]);
            }

            return relu_output;
        }

        std::string get_name() const override {
            return "ReLULayer";
        }

        size_t get_input_size() const override {
            return 0;  // ReLU doesn't change the size
        }

        size_t get_output_size() const override {
            return 0;  // ReLU doesn't change the size
        }

    };

} // namespace nnm