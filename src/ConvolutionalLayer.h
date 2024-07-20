#pragma once

#include "Layer.h"
#include "Matrix.h"
#include "Vector.h"
#include <random>

namespace nnm {

    class ConvolutionalLayer : public Layer {
    private:
        size_t in_channels, out_channels, kernel_size, stride, padding;
        Matrix weights;
        Vector bias;
        Matrix weight_gradients;
        Vector bias_gradients;

    public:
        ConvolutionalLayer(size_t in_channels, size_t out_channels, size_t kernel_size,
                           size_t stride = 1, size_t padding = 0);

        Matrix forward(const Matrix &input) override;

        Matrix backward(const Matrix &input, const Matrix &output_gradient) override;

        void update_parameters(float learning_rate) override;

        void save(std::ostream &os) const override;

        void load(std::istream &is) override;

        [[nodiscard]] std::string get_name() const override;

        [[nodiscard]] size_t get_input_size() const override;

        [[nodiscard]] size_t get_output_size() const override;

        [[nodiscard]] std::unique_ptr<Layer> clone() const override;

        void set_weights(const Matrix &new_weights);

        void set_bias(const Vector &new_bias);

        [[nodiscard]] const Matrix &get_weights() const { return weights; }

        [[nodiscard]] const Vector &get_bias() const { return bias; }

        [[nodiscard]] size_t get_padding() const { return padding; }

        [[nodiscard]] size_t get_kernel_size() const { return kernel_size; }

        [[nodiscard]] size_t get_stride() const { return stride; }

        Matrix add_padding(const Matrix &input) const;
    };

} // namespace nnm