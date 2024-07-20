#pragma once

#include "Layer.h"

namespace nnm {

    class FullyConnectedLayer : public Layer {
    private:
        Matrix weights;
        Vector bias;

    public:
        FullyConnectedLayer(size_t input_size, size_t output_size);

        Matrix forward(const Matrix &input) override;

        Matrix backward(const Matrix &input, const Matrix &output_gradient) override;

        void update_parameters(float learning_rate) override;

        void save(std::ostream &os) const override;

        void load(std::istream &is) override;
    };

} // namespace nnm