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
                for (size_t i = 0; i < H_out; ++i) {
                    for (size_t j = 0; j < W_out; ++j) {
                        Tensor4D x_slice = padded_input.subTensor(n, 0, i * stride, j * stride,
                                                                  1, in_channels, kernel_size, kernel_size);
                        Tensor4D w_slice = weights.subTensor(f, 0, 0, 0,
                                                             1, in_channels, kernel_size, kernel_size);

                        float sum = x_slice.elementWiseMul(w_slice).sum() + bias[f];
                        output(n, f, i, j) = sum;
                    }
                }
            }
        }

        return output;
    }


    Tensor4D ConvolutionalLayer::backward(const Tensor4D &input, const Tensor4D &output_gradient) {
        throw std::runtime_error("Not implemented");
    }

    void ConvolutionalLayer::update_parameters(float learning_rate) {
        // Implement update_parameters functionality
    }

    void ConvolutionalLayer::save(std::ostream &os) const {
        // Implement save functionality
    }

    void ConvolutionalLayer::load(std::istream &is) {
        // Implement load functionality
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
        // Implement set_weights functionality
    }

    void ConvolutionalLayer::set_bias(const Vector &new_bias) {
        // Implement set_bias functionality
    }

    std::unique_ptr<Layer<Tensor4D>> ConvolutionalLayer::clone() const {
        return std::make_unique<ConvolutionalLayer>(*this);
    }


} // namespace nnm
