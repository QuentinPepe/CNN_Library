#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include <random>
#include <cmath>

namespace nnm {

    class LinearLayer : public Layer<Tensor4D, Tensor4D> {
    private:
        Tensor4D weights;
        Tensor4D bias;
        size_t in_features;
        size_t out_features;

    public:
        LinearLayer(size_t in_features, size_t out_features)
                : in_features(in_features), out_features(out_features),
                  weights(1, out_features, in_features, 1),
                  bias(1, out_features, 1, 1) {

            // Xavier/Glorot initialization
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0 / std::sqrt(in_features), 1.0 / std::sqrt(in_features));

            for (size_t i = 0; i < out_features; ++i) {
                for (size_t j = 0; j < in_features; ++j) {
                    weights(0, i, j, 0) = dis(gen);
                }
                bias(0, i, 0, 0) = 0.0f;
            }
        }

        Tensor4D forward(const Tensor4D &input) {
            size_t batch_size = input.getBatchSize();
            Tensor4D output(batch_size, out_features, 1, 1);

            for (size_t n = 0; n < batch_size; ++n) {
                for (size_t j = 0; j < out_features; ++j) {
                    __m256 sum_vec = _mm256_setzero_ps();
                    size_t i;

                    for (i = 0; i + 7 < in_features; i += 8) {
                        __m256 input_vec = _mm256_loadu_ps(&input(n, i, 0, 0));
                        __m256 weight_vec = _mm256_loadu_ps(&weights(0, i, j, 0));
                        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(input_vec, weight_vec));
                    }

                    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                    __m128 sum = _mm_add_ps(sum_high, sum_low);
                    sum = _mm_hadd_ps(sum, sum);
                    sum = _mm_hadd_ps(sum, sum);
                    float result = _mm_cvtss_f32(sum);

                    for (; i < in_features; ++i) {
                        result += input(n, i, 0, 0) * weights(0, i, j, 0);
                    }

                    output(n, j, 0, 0) = result + bias(0, j, 0, 0);
                }
            }

            return output;
        }

        std::string get_name() const override {
            return "LinearLayer";
        }

        size_t get_input_size() const override {
            return in_features;
        }

        size_t get_output_size() const override {
            return out_features;
        }

        std::unique_ptr<Layer<Tensor4D, Tensor4D>> clone() const override {
            return std::make_unique<LinearLayer>(*this);
        }

        const Tensor4D &get_weights() const {
            return weights;
        }

        const Tensor4D &get_bias() const {
            return bias;
        }

        void set_weights(const Tensor4D &new_weights) {
            if (new_weights.getChannels() != in_features || new_weights.getHeight() != out_features) {
                throw std::invalid_argument("New weights dimensions do not match layer dimensions");
            }
            weights = new_weights;
        }

        void set_bias(const Tensor4D &new_bias) {
            if (new_bias.getChannels() != out_features || new_bias.getHeight() != 1 || new_bias.getWidth() != 1) {
                throw std::invalid_argument("New bias dimensions do not match layer dimensions");
            }
            bias = new_bias;
        }
    };

} // namespace nnm
