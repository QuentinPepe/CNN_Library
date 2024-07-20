#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include "Matrix.h"
#include "Vector.h"
#include <random>

namespace nnm {

    class ConvolutionalLayer : public Layer<Tensor4D> {
    private:
        size_t in_channels, out_channels, kernel_size, stride, padding;
        Tensor4D weights;
        Vector bias;
        Tensor4D weight_gradients;
        Vector bias_gradients;

    public:
        ConvolutionalLayer(size_t in_channels, size_t out_channels, size_t kernel_size,
                           size_t stride = 1, size_t padding = 0);

        Tensor4D forward(const Tensor4D &input) override;

        Tensor4D backward(const Tensor4D &input, const Tensor4D &output_gradient) override;

        void update_parameters(float learning_rate) override;

        void save(std::ostream &os) const override;

        void load(std::istream &is) override;

        [[nodiscard]] std::string get_name() const override;

        [[nodiscard]] size_t get_input_size() const override;

        [[nodiscard]] size_t get_output_size() const override;

        [[nodiscard]] std::unique_ptr<Layer<Tensor4D>> clone() const override;

        void set_weights(const Tensor4D &new_weights);

        void set_bias(const Vector &new_bias);

        [[nodiscard]] const Tensor4D &get_weights() const { return weights; }

        [[nodiscard]] const Vector &get_bias() const { return bias; }

        [[nodiscard]] size_t get_padding() const { return padding; }

        [[nodiscard]] size_t get_kernel_size() const { return kernel_size; }

        [[nodiscard]] size_t get_stride() const { return stride; }
    };

} // namespace nnm