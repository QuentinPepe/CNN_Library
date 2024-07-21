#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include <cmath>
#include <memory>

namespace nnm {

    class BatchNorm2d : public Layer<Tensor4D, Tensor4D> {
    private:
        size_t num_features;
        float eps;

        Tensor4D weight;
        Tensor4D bias;
        Tensor4D running_mean;
        Tensor4D running_var;

    public:
        BatchNorm2d(size_t num_features, float eps = 1e-5)
                : num_features(num_features), eps(eps),
                  weight(1, num_features, 1, 1),
                  bias(1, num_features, 1, 1),
                  running_mean(1, num_features, 1, 1),
                  running_var(1, num_features, 1, 1) {
            // Initialize weights to 1 and biases to 0
            weight.fill(1.0f);
            bias.fill(0.0f);
            running_mean.fill(0.0f);
            running_var.fill(1.0f);
        }

        void set_parameters(const Tensor4D &weight, const Tensor4D &bias,
                            const Tensor4D &running_mean, const Tensor4D &running_var) {
            if (weight.getChannels() != num_features || bias.getChannels() != num_features ||
                running_mean.getChannels() != num_features || running_var.getChannels() != num_features) {
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
                    float inv_std = 1.0f / std::sqrt(running_var(0, c, 0, 0) + eps);
                    for (size_t h = 0; h < input.getHeight(); ++h) {
                        for (size_t w = 0; w < input.getWidth(); ++w) {
                            float normalized = (input(n, c, h, w) - running_mean(0, c, 0, 0)) * inv_std;
                            output(n, c, h, w) = normalized * weight(0, c, 0, 0) + bias(0, c, 0, 0);
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