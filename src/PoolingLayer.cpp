#include "PoolingLayer.h"
#include <algorithm>
#include <stdexcept>
#include <limits>

namespace nnm {

    PoolingLayer::PoolingLayer(size_t kernel_size, size_t stride)
            : kernel_size(kernel_size), stride(stride), last_input(nullptr) {}

    PoolingLayer::~PoolingLayer() {
        delete last_input;
    }

    Matrix PoolingLayer::forward(const Matrix &input) {
        size_t input_height = input.getRows();
        size_t input_width = input.getCols();
        size_t output_height = (input_height - kernel_size) / stride + 1;
        size_t output_width = (input_width - kernel_size) / stride + 1;

        Matrix output(output_height, output_width);
        max_indices.clear();
        max_indices.reserve(output_height * output_width);

        for (size_t i = 0; i < output_height; ++i) {
            for (size_t j = 0; j < output_width; ++j) {
                float max_val = std::numeric_limits<float>::lowest();
                size_t max_i = 0, max_j = 0;

                for (size_t ki = 0; ki < kernel_size; ++ki) {
                    for (size_t kj = 0; kj < kernel_size; ++kj) {
                        size_t input_i = i * stride + ki;
                        size_t input_j = j * stride + kj;
                        if (input(input_i, input_j) > max_val) {
                            max_val = input(input_i, input_j);
                            max_i = input_i;
                            max_j = input_j;
                        }
                    }
                }

                output(i, j) = max_val;
                max_indices.emplace_back(max_i, max_j);
            }
        }

        delete last_input;
        last_input = new Matrix(input);
        return output;
    }

    Matrix PoolingLayer::backward(const Matrix &input, const Matrix &output_gradient) {
        size_t input_height = last_input->getRows();
        size_t input_width = last_input->getCols();

        Matrix input_gradient(input_height, input_width);

        size_t idx = 0;
        for (size_t i = 0; i < output_gradient.getRows(); ++i) {
            for (size_t j = 0; j < output_gradient.getCols(); ++j) {
                const auto &[max_i, max_j] = max_indices[idx++];
                input_gradient(max_i, max_j) += output_gradient(i, j);
            }
        }

        return input_gradient;
    }

    void PoolingLayer::update_parameters(float learning_rate) {
        // Pooling layer has no parameters to update
    }

    void PoolingLayer::save(std::ostream &os) const {
        os << kernel_size << " " << stride << "\n";
    }

    void PoolingLayer::load(std::istream &is) {
        is >> kernel_size >> stride;
    }

    std::string PoolingLayer::get_name() const {
        return "PoolingLayer";
    }

    size_t PoolingLayer::get_input_size() const {
        return last_input ? last_input->getCols() : 0;
    }

    size_t PoolingLayer::get_output_size() const {
        return last_input ? (last_input->getCols() - kernel_size) / stride + 1 : 0;
    }

    std::unique_ptr<Layer> PoolingLayer::clone() const {
        auto cloned = std::make_unique<PoolingLayer>(kernel_size, stride);
        if (last_input) {
            cloned->last_input = new Matrix(*last_input);
        }
        cloned->max_indices = max_indices;
        return cloned;
    }

} // namespace nnm