#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include <cmath>
#include <algorithm>

namespace nnm {

    class MaxPoolingLayer : public Layer<Tensor4D, Tensor4D> {
    private:
        size_t pooling_height;
        size_t pooling_width;
        size_t stride;

    public:
        MaxPoolingLayer(size_t pooling_height, size_t pooling_width, size_t stride)
                : pooling_height(pooling_height), pooling_width(pooling_width), stride(stride) {}

        Tensor4D forward(const Tensor4D &x) override {
            size_t N = x.getBatchSize();
            size_t F = x.getChannels();
            size_t H = x.getHeight();
            size_t W = x.getWidth();

            size_t height_pooled_out = 1 + (H - pooling_height) / stride;
            size_t width_pooled_out = 1 + (W - pooling_width) / stride;

            Tensor4D pooled_output(N, F, height_pooled_out, width_pooled_out);

            for (size_t n = 0; n < N; ++n) {
                for (size_t f = 0; f < F; ++f) {
                    for (size_t i = 0; i < height_pooled_out; ++i) {
                        for (size_t j = 0; j < width_pooled_out; ++j) {
                            size_t ii = i * stride;
                            size_t jj = j * stride;

                            float max_val = std::numeric_limits<float>::lowest();
                            for (size_t ph = 0; ph < pooling_height; ++ph) {
                                for (size_t pw = 0; pw < pooling_width; ++pw) {
                                    max_val = std::max(max_val, x(n, f, ii + ph, jj + pw));
                                }
                            }
                            pooled_output(n, f, i, j) = max_val;
                        }
                    }
                }
            }

            return pooled_output;
        }

        std::string get_name() const override {
            return "MaxPoolingLayer";
        }

        size_t get_input_size() const override {
            // This should return the input volume size, but it's not applicable for pooling layers
            return 0;
        }

        size_t get_output_size() const override {
            // This should return the output volume size, but it's not applicable for pooling layers
            return 0;
        }

    };

} // namespace nnm