#include "ConvolutionalLayer.h"
#include <cmath>
#include <stdexcept>

namespace nnm {

    ConvolutionalLayer::ConvolutionalLayer(size_t in_channels, size_t out_channels, size_t kernel_size,
                                           size_t stride, size_t padding)
            : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
              stride(stride), padding(padding),
              weights(out_channels, in_channels, kernel_size, kernel_size),
              bias(out_channels),
              weight_gradients(out_channels, in_channels, kernel_size, kernel_size),
              bias_gradients(out_channels) {

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
        for (size_t i = 0; i < out_channels; ++i) {
            bias[i] = 0.0f;
        }

        // Initialize gradients to zero
        weight_gradients.fill(0.0f);
        bias_gradients.fill(0.0f);
    }


    Tensor4D ConvolutionalLayer::forward(const Tensor4D &input) {
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
                        float sum = x_slice.elementWiseMul(w_slice).sum() + bias[f];
                        output(n, f, height_index, width_index) = sum;
                        width_index++;
                    }
                    height_index++;
                }

            }
        }

        return output;
    }

    std::string ConvolutionalLayer::get_name() const {
        return "ConvolutionalLayer";
    }

    size_t ConvolutionalLayer::get_input_size() const {
        return in_channels;
    }

    size_t ConvolutionalLayer::get_output_size() const {
        return out_channels;
    }

    void ConvolutionalLayer::set_weights(const Tensor4D &new_weights) {
        weights = new_weights;
    }

    void ConvolutionalLayer::set_bias(const Vector &new_bias) {
        bias = new_bias;
    }

    std::unique_ptr<Layer<Tensor4D, Tensor4D>> ConvolutionalLayer::clone() const {
        return std::make_unique<ConvolutionalLayer>(*this);
    }

    Tensor4D ConvolutionalLayer::get_weight_gradients() {
        return weight_gradients;
    }

    Vector ConvolutionalLayer::get_bias_gradients() {
        return bias_gradients;
    }


} // namespace nnm
