#pragma once

#include "Layer.h"
#include "Matrix.h"
#include <algorithm>

namespace nnm {

    class ReLULayer : public Layer<Matrix, Matrix> {
    public:
        ReLULayer() = default;

        Matrix forward(const Matrix &x) override {
            Matrix relu_output(x.getRows(), x.getCols());
            for (size_t i = 0; i < x.getRows(); ++i) {
                for (size_t j = 0; j < x.getCols(); ++j) {
                    relu_output(i, j) = std::max(0.0f, x(i, j));
                }
            }
            return relu_output;
        }

        std::string get_name() const override {
            return "ReLULayer";
        }

        size_t get_input_size() const override {
            return 0;  // ReLU doesn't change the size
        }

        size_t get_output_size() const override {
            return 0;  // ReLU doesn't change the size
        }

        std::unique_ptr<Layer<Matrix, Matrix>> clone() const override {
            return std::make_unique<ReLULayer>(*this);
        }
    };

} // namespace nnm