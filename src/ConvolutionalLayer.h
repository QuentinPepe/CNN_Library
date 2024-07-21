#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include "Matrix.h"
#include <random>

namespace nnm {

    class ConvolutionalLayer : public Layer<Tensor4D, Tensor4D> {
    private:
        size_t in_channels, out_channels, kernel_size, stride, padding;
        Tensor4D weights;
        Tensor4D bias;
        Tensor4D weight_gradients;
        Tensor4D bias_gradients;

    public:
        ConvolutionalLayer(size_t in_channels, size_t out_channels, size_t kernel_size,
                           size_t stride = 1, size_t padding = 0)
                : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
                  stride(stride), padding(padding),
                  weights(out_channels, in_channels, kernel_size, kernel_size),
                  bias(1, out_channels, 1, 1),
                  weight_gradients(out_channels, in_channels, kernel_size, kernel_size),
                  bias_gradients(1, out_channels, 1, 1) {

            // Xavier/Glorot initialization for weights
            std::random_device rd;
            std::mt19937 gen(rd());
            float limit = std::sqrt(
                    6.0f / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size));
            std::uniform_real_distribution<> dis(-limit, limit);

            for (size_t o = 0; o < out_channels; ++o) {
                for (size_t i = 0; i < in_channels; ++i) {
                    for (size_t h = 0; h < kernel_size; ++h) {
                        for (size_t w = 0; w < kernel_size; ++w) {
                            weights(o, i, h, w) = static_cast<float>(dis(gen));
                        }
                    }
                }
            }

            // Initialize biases to zero
            bias.fill(0.0f);

            // Initialize gradients to zero
            weight_gradients.fill(0.0f);
            bias_gradients.fill(0.0f);
        }

        Tensor4D forward(const Tensor4D &input) override {
            size_t N = input.getBatchSize();
            size_t H = input.getHeight();
            size_t W = input.getWidth();

            size_t H_out = 1 + (H + 2 * padding - kernel_size) / stride;
            size_t W_out = 1 + (W + 2 * padding - kernel_size) / stride;

            Tensor4D output(N, out_channels, H_out, W_out);

            Tensor4D padded_input = input.pad({{0,       0},
                                               {0,       0},
                                               {padding, padding},
                                               {padding, padding}});

            for (size_t n = 0; n < N; ++n) {
                for (size_t f = 0; f < out_channels; ++f) {
                    int height_index = 0;
                    for (size_t i = 0; i < H; i += stride) {
                        int width_index = 0;
                        for (size_t j = 0; j < W; j += stride) {
                            Tensor4D x_slice = padded_input.subTensor(n, 0, i, j,
                                                                      1, in_channels, kernel_size, kernel_size);
                            Tensor4D w_slice = weights.subTensor(f, 0, 0, 0,
                                                                 1, in_channels, kernel_size, kernel_size);
                            float sum = x_slice.elementWiseMul(w_slice).sum() + bias(0, f, 0, 0);
                            output(n, f, height_index, width_index) = sum;
                            width_index++;
                        }
                        height_index++;
                    }
                }
            }

            return output;
        }

        [[nodiscard]] std::string get_name() const override {
            return "ConvolutionalLayer";
        }

        [[nodiscard]] size_t get_input_size() const override {
            return in_channels;
        }

        [[nodiscard]] size_t get_output_size() const override {
            return out_channels;
        }

        void set_weights(const Tensor4D &new_weights) {
            weights = new_weights;
        }

        void set_bias(const Tensor4D &new_bias) {
            bias = new_bias;
        }

        [[nodiscard]] const Tensor4D &get_weights() const { return weights; }

        [[nodiscard]] const Tensor4D &get_bias() const { return bias; }

        [[nodiscard]] size_t get_padding() const { return padding; }

        [[nodiscard]] size_t get_kernel_size() const { return kernel_size; }

        [[nodiscard]] size_t get_stride() const { return stride; }

        Tensor4D get_weight_gradients() {
            return weight_gradients;
        }

        Tensor4D get_bias_gradients() {
            return bias_gradients;
        }
    };

} // namespace nnm