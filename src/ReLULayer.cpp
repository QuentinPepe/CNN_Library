#include "ReLULayer.h"
#include <algorithm>
#include <cmath>

namespace nnm {

    ReLULayer::ReLULayer() : last_input(nullptr) {}

    ReLULayer::~ReLULayer() {
        delete last_input;
    }

    Matrix ReLULayer::forward(const Matrix &input) {
        delete last_input;
        last_input = new Matrix(input);
        Matrix output(input.getRows(), input.getCols());

#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < input.getRows(); ++i) {
            for (size_t j = 0; j < input.getCols(); ++j) {
                output(i, j) = std::max(0.0f, input(i, j));
            }
        }

        return output;
    }

    Matrix ReLULayer::backward(const Matrix &input, const Matrix &output_gradient) {
        if (!last_input) {
            throw std::runtime_error("Backward called before forward");
        }

        Matrix input_gradient(input.getRows(), input.getCols());

#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < input.getRows(); ++i) {
            for (size_t j = 0; j < input.getCols(); ++j) {
                input_gradient(i, j) = (*last_input)(i, j) > 0 ? output_gradient(i, j) : 0;
            }
        }

        return input_gradient;
    }

    void ReLULayer::update_parameters(float learning_rate) {
        // ReLU has no parameters to update
    }

    void ReLULayer::save(std::ostream &os) const {
        // ReLU has no parameters to save
    }

    void ReLULayer::load(std::istream &is) {
        // ReLU has no parameters to load
    }

    std::string ReLULayer::get_name() const {
        return "ReLULayer";
    }

    size_t ReLULayer::get_input_size() const {
        return last_input ? last_input->getCols() : 0;
    }

    size_t ReLULayer::get_output_size() const {
        return last_input ? last_input->getCols() : 0;
    }

    std::unique_ptr<Layer> ReLULayer::clone() const {
        return std::make_unique<ReLULayer>(*this);
    }

} // namespace nnm