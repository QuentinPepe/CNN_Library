#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>

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

        Tensor4D forward(const Tensor4D &input) override {
            size_t batch_size = input.getBatchSize();
            size_t input_size = input.getChannels() * input.getHeight() * input.getWidth();

            if (input_size != in_features) {
                throw std::invalid_argument("Input size does not match layer's in_features");
            }

            std::cout << "Input dimensions: " << input.getBatchSize() << "x" << input.getChannels()
                      << "x" << input.getHeight() << "x" << input.getWidth() << std::endl;
            std::cout << "Weight dimensions: " << weights.getBatchSize() << "x" << weights.getChannels()
                      << "x" << weights.getHeight() << "x" << weights.getWidth() << std::endl;

            Tensor4D output(batch_size, out_features, 1, 1);

            for (size_t n = 0; n < batch_size; ++n) {
                for (size_t j = 0; j < out_features; ++j) {
                    float result = 0.0f;
                    for (size_t i = 0; i < in_features; ++i) {
                        size_t c = i / (input.getHeight() * input.getWidth());
                        size_t h = (i % (input.getHeight() * input.getWidth())) / input.getWidth();
                        size_t w = (i % (input.getHeight() * input.getWidth())) % input.getWidth();
                        result += input(n, c, h, w) * weights(0, j, i, 0);
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

        const Tensor4D &get_weights() const {
            return weights;
        }

        const Tensor4D &get_bias() const {
            return bias;
        }

        void set_weights(const Tensor4D &new_weights) {
            if (new_weights.getBatchSize() != 1 ||
                new_weights.getChannels() != out_features ||
                new_weights.getHeight() != in_features ||
                new_weights.getWidth() != 1) {
                throw std::invalid_argument("New weights dimensions do not match layer dimensions");
            }
            weights = new_weights;
        }

        void set_bias(const Tensor4D &new_bias) {
            if (new_bias.getBatchSize() != 1 ||
                new_bias.getChannels() != out_features ||
                new_bias.getHeight() != 1 ||
                new_bias.getWidth() != 1) {
                throw std::invalid_argument("New bias dimensions do not match layer dimensions");
            }
            bias = new_bias;
        }
    };

} // namespace nnm