#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include "Vector.h"
#include <cmath>
#include <memory>

namespace nnm {

    class BatchNorm2d : public Layer<Tensor4D, Tensor4D> {
    private:
        size_t num_features;
        float eps;

        Vector weight;
        Vector bias;
        Vector running_mean;
        Vector running_var;

    public:
        BatchNorm2d(size_t num_features, float eps = 1e-5)
                : num_features(num_features), eps(eps),
                  weight(num_features), bias(num_features),
                  running_mean(num_features), running_var(num_features) {
        }

        void set_parameters(const Vector &weight, const Vector &bias,
                            const Vector &running_mean, const Vector &running_var) {
            if (weight.size() != num_features || bias.size() != num_features ||
                running_mean.size() != num_features || running_var.size() != num_features) {
                throw std::invalid_argument("Parameter sizes do not match num_features");
            }
            this->weight = weight;
            this->bias = bias;
            this->running_mean = running_mean;
            this->running_var = running_var;
        }

        Tensor4D forward(const Tensor4D &input) override {
            if (input.getChannels() != num_features) {
                throw std::invalid_argument("Input channel dimension doesn't match num_features");
            }

            Tensor4D output(input.getBatchSize(), num_features, input.getHeight(), input.getWidth());

            for (size_t n = 0; n < input.getBatchSize(); ++n) {
                for (size_t c = 0; c < num_features; ++c) {
                    float inv_std = 1.0f / std::sqrt(running_var[c] + eps);
                    for (size_t h = 0; h < input.getHeight(); ++h) {
                        for (size_t w = 0; w < input.getWidth(); ++w) {
                            float normalized = (input(n, c, h, w) - running_mean[c]) * inv_std;
                            output(n, c, h, w) = normalized * weight[c] + bias[c];
                        }
                    }
                }
            }

            return output;
        }

        std::string get_name() const override {
            return "BatchNorm2d";
        }

        size_t get_input_size() const override {
            return num_features;
        }

        size_t get_output_size() const override {
            return num_features;
        }

        std::unique_ptr<Layer<Tensor4D, Tensor4D>> clone() const override {
            return std::make_unique<BatchNorm2d>(*this);
        }
    };

} // namespace nnm