#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include "Matrix.h"
#include "Vector.h"
#include <cmath>

namespace nnm {

    class FullyConnectedLayer : public Layer<Tensor4D, Matrix> {
    private:
        Matrix weights;
        Vector biases;
        size_t input_size;
        size_t output_size;

    public:
        FullyConnectedLayer(size_t input_size, size_t output_size)
                : input_size(input_size), output_size(output_size),
                  weights(input_size, output_size), biases(output_size) {
            // Initialize weights and biases (you might want to use a better initialization)
            for (size_t i = 0; i < input_size; ++i) {
                for (size_t j = 0; j < output_size; ++j) {
                    weights(i, j) = static_cast<float>(rand()) / RAND_MAX - 0.5f;
                }
            }
            for (size_t i = 0; i < output_size; ++i) {
                biases[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
            }
        }

        Matrix forward(const Tensor4D &x) override {
            size_t N = x.getBatchSize();
            size_t D = x.getChannels() * x.getHeight() * x.getWidth();

            if (D != input_size) {
                throw std::invalid_argument("Input size doesn't match the layer's input size");
            }

            Matrix x_reshaped(N, D);
            for (size_t n = 0; n < N; ++n) {
                size_t index = 0;
                for (size_t c = 0; c < x.getChannels(); ++c) {
                    for (size_t h = 0; h < x.getHeight(); ++h) {
                        for (size_t w = 0; w < x.getWidth(); ++w) {
                            x_reshaped(n, index++) = x(n, c, h, w);
                        }
                    }
                }
            }

            Matrix fc_output = x_reshaped * weights;
            for (size_t n = 0; n < N; ++n) {
                for (size_t m = 0; m < output_size; ++m) {
                    fc_output(n, m) += biases[m];
                }
            }

            return fc_output;
        }

        std::string get_name() const override {
            return "FullyConnectedLayer";
        }

        size_t get_input_size() const override {
            return input_size;
        }

        size_t get_output_size() const override {
            return output_size;
        }

        std::unique_ptr<Layer<Tensor4D, Matrix>> clone() const override {
            return std::make_unique<FullyConnectedLayer>(*this);
        }

        void set_weights(const Matrix &new_weights) {
            if (new_weights.getRows() != input_size || new_weights.getCols() != output_size) {
                throw std::invalid_argument("New weights dimensions don't match the layer's dimensions");
            }
            weights = new_weights;
        }

        void set_biases(const Vector &new_biases) {
            if (new_biases.size() != output_size) {
                throw std::invalid_argument("New biases size doesn't match the layer's output size");
            }
            biases = new_biases;
        }

        const Matrix &get_weights() const { return weights; }

        const Vector &get_biases() const { return biases; }
    };

} // namespace nnm