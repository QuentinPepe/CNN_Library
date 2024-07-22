#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

namespace nnm {

    class SoftMaxLayer : public Layer<Tensor4D, Tensor4D> {
    private:
        int dimension;

    public:
        SoftMaxLayer(int dimension = 1) : dimension(dimension) {}

        Tensor4D forward(const Tensor4D &input) override {
            size_t batch_size = input.getBatchSize();
            size_t channels = input.getChannels();
            size_t height = input.getHeight();
            size_t width = input.getWidth();

            Tensor4D output(batch_size, channels, height, width);

            for (size_t n = 0; n < batch_size; ++n) {
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t h = 0; h < height; ++h) {
                        for (size_t w = 0; w < width; ++w) {
                            if (dimension == 0) {
                                // Apply softmax along the batch dimension
                                float max_val = -std::numeric_limits<float>::infinity();
                                for (size_t b = 0; b < batch_size; ++b) {
                                    max_val = std::max(max_val, input(b, c, h, w));
                                }
                                float sum_exp = 0.0f;
                                for (size_t b = 0; b < batch_size; ++b) {
                                    sum_exp += std::exp(input(b, c, h, w) - max_val);
                                }
                                for (size_t b = 0; b < batch_size; ++b) {
                                    output(b, c, h, w) = std::exp(input(b, c, h, w) - max_val) / sum_exp;
                                }
                            } else if (dimension == 1) {
                                // Apply softmax along the channels dimension
                                float max_val = -std::numeric_limits<float>::infinity();
                                for (size_t c = 0; c < channels; ++c) {
                                    max_val = std::max(max_val, input(n, c, h, w));
                                }
                                float sum_exp = 0.0f;
                                for (size_t c = 0; c < channels; ++c) {
                                    sum_exp += std::exp(input(n, c, h, w) - max_val);
                                }
                                for (size_t c = 0; c < channels; ++c) {
                                    output(n, c, h, w) = std::exp(input(n, c, h, w) - max_val) / sum_exp;
                                }
                            } else if (dimension == 2) {
                                // Apply softmax along the height dimension
                                float max_val = -std::numeric_limits<float>::infinity();
                                for (size_t h = 0; h < height; ++h) {
                                    max_val = std::max(max_val, input(n, c, h, w));
                                }
                                float sum_exp = 0.0f;
                                for (size_t h = 0; h < height; ++h) {
                                    sum_exp += std::exp(input(n, c, h, w) - max_val);
                                }
                                for (size_t h = 0; h < height; ++h) {
                                    output(n, c, h, w) = std::exp(input(n, c, h, w) - max_val) / sum_exp;
                                }
                            } else if (dimension == 3) {
                                // Apply softmax along the width dimension
                                float max_val = -std::numeric_limits<float>::infinity();
                                for (size_t w = 0; w < width; ++w) {
                                    max_val = std::max(max_val, input(n, c, h, w));
                                }
                                float sum_exp = 0.0f;
                                for (size_t w = 0; w < width; ++w) {
                                    sum_exp += std::exp(input(n, c, h, w) - max_val);
                                }
                                for (size_t w = 0; w < width; ++w) {
                                    output(n, c, h, w) = std::exp(input(n, c, h, w) - max_val) / sum_exp;
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }

        std::string get_name() const override {
            return "SoftMaxLayer";
        }

        size_t get_input_size() const override {
            return 0;  // Not applicable as it's determined at runtime
        }

        size_t get_output_size() const override {
            return 0;  // Not applicable as it's determined at runtime
        }

    };

} // namespace nnm