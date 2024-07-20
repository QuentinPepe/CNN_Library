#include "ConvolutionalLayer.h"
#include <cmath>
#include <stdexcept>

namespace nnm {

    ConvolutionalLayer::ConvolutionalLayer(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride,
                                           size_t padding)
            : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
              stride(stride), padding(padding),
              weights(out_channels, in_channels * kernel_size * kernel_size),
              bias(out_channels),
              weight_gradients(out_channels, in_channels * kernel_size * kernel_size),
              bias_gradients(out_channels) {

        // Initialization of weights
        double step = 2.0 / (out_channels * in_channels * kernel_size * kernel_size - 1);
        double current = -1.0;
        for (size_t i = 0; i < weights.getRows(); ++i) {
            for (size_t j = 0; j < weights.getCols(); ++j) {
                weights(i, j) = static_cast<float>(current);
                current += step;
            }
        }

        // Initialization of biases
        step = 2.0 / (out_channels - 1);
        current = -1.0;
        for (size_t i = 0; i < out_channels; ++i) {
            bias[i] = static_cast<float>(current);
            current += step;
        }
    }

    Matrix ConvolutionalLayer::add_padding(const Matrix &input) const {
        size_t H = static_cast<size_t>(std::sqrt(input.getCols() / in_channels));
        size_t W = H;
        size_t padded_height = H + 2 * padding;
        size_t padded_width = W + 2 * padding;

        Matrix padded_input(padded_height, padded_width * in_channels, 0.0f);

        for (size_t c = 0; c < in_channels; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    padded_input(h + padding, w + padding + c * padded_width) =
                            input(0, w + h * W + c * H * W);
                }
            }
        }

        return padded_input;
    }

    Matrix ConvolutionalLayer::forward(const Matrix &input) {
        size_t N = input.getRows();
        size_t H = static_cast<size_t>(std::sqrt(input.getCols() / in_channels));
        size_t W = H;

        size_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
        size_t W_out = (W + 2 * padding - kernel_size) / stride + 1;

        Matrix output(N, out_channels * H_out * W_out);

        Matrix x_padded = add_padding(input);

        for (size_t n = 0; n < N; ++n) {
            for (size_t f = 0; f < out_channels; ++f) {
                for (size_t i = 0; i < H_out; ++i) {
                    for (size_t j = 0; j < W_out; ++j) {
                        double sum = 0.0;
                        for (size_t c = 0; c < in_channels; ++c) {
                            for (size_t p = 0; p < kernel_size; ++p) {
                                for (size_t q = 0; q < kernel_size; ++q) {
                                    size_t h_index = i * stride + p;
                                    size_t w_index = j * stride + q;
                                    sum += static_cast<double>(x_padded(n, (c * (H + 2 * padding) + h_index) *
                                                                           (W + 2 * padding) + w_index)) *
                                           weights(f, (c * kernel_size + p) * kernel_size + q);
                                }
                            }
                        }
                        output(n, (f * H_out + i) * W_out + j) = static_cast<float>(sum + bias[f]);
                    }
                }
            }
        }

        return output;
    }

    Matrix ConvolutionalLayer::backward(const Matrix &input, const Matrix &output_gradient) {
        // Implement backward pass
        return Matrix(input.getRows(), input.getCols());
    }

    void ConvolutionalLayer::update_parameters(float learning_rate) {
        for (size_t i = 0; i < weights.getRows(); ++i) {
            for (size_t j = 0; j < weights.getCols(); ++j) {
                weights(i, j) -= learning_rate * weight_gradients(i, j);
            }
        }
        for (size_t i = 0; i < bias.size(); ++i) {
            bias[i] -= learning_rate * bias_gradients[i];
        }
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

    std::unique_ptr<Layer> ConvolutionalLayer::clone() const {
        return std::make_unique<ConvolutionalLayer>(*this);
    }

    void ConvolutionalLayer::set_weights(const Matrix &new_weights) {
        if (new_weights.getRows() != weights.getRows() || new_weights.getCols() != weights.getCols()) {
            throw std::invalid_argument("New weights dimensions do not match");
        }
        weights = new_weights;
    }

    void ConvolutionalLayer::set_bias(const Vector &new_bias) {
        if (new_bias.size() != bias.size()) {
            throw std::invalid_argument("New bias size does not match");
        }
        bias = new_bias;
    }

} // namespace nnm
