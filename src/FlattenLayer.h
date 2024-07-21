#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include <stdexcept>

namespace nnm {

    class Flatten : public Layer<Tensor4D, Tensor4D> {
    private:
        int start_dim;
        int end_dim;

    public:
        Flatten(int start_dim = 1, int end_dim = -1) : start_dim(start_dim), end_dim(end_dim) {}

        Tensor4D forward(const Tensor4D &input) override {
            size_t batch_size = input.getBatchSize();
            size_t channels = input.getChannels();
            size_t height = input.getHeight();
            size_t width = input.getWidth();

            int real_end_dim = (end_dim == -1) ? 3 : end_dim;

            if (start_dim < 0 || start_dim > 3 || real_end_dim < 0 || real_end_dim > 3 || start_dim > real_end_dim) {
                throw std::invalid_argument("Invalid start_dim or end_dim");
            }

            size_t rows = batch_size;
            size_t cols = channels * height * width;

            Tensor4D output(1, rows, cols, 1);

            for (size_t n = 0; n < batch_size; ++n) {
                size_t index = 0;
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t h = 0; h < height; ++h) {
                        for (size_t w = 0; w < width; ++w) {
                            output(0, n, index++, 0) = input(n, c, h, w);
                        }
                    }
                }
            }

            return output;
        }

        std::string get_name() const override {
            return "Flatten";
        }

        size_t get_input_size() const override {
            return 0;  // Not applicable for Flatten
        }

        size_t get_output_size() const override {
            return 0;  // Not applicable for Flatten
        }

    };

} // namespace nnm