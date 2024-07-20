#pragma once

#include "Layer.h"
#include <vector>

namespace nnm {

    class PoolingLayer : public Layer {
    public:
        PoolingLayer(size_t kernel_size, size_t stride);

        ~PoolingLayer() override;

        Matrix forward(const Matrix &input) override;

        Matrix backward(const Matrix &input, const Matrix &output_gradient) override;

        void update_parameters(float learning_rate) override;

        void save(std::ostream &os) const override;

        void load(std::istream &is) override;

        std::string get_name() const override;

        size_t get_input_size() const override;

        size_t get_output_size() const override;

        std::unique_ptr<Layer> clone() const override;

    private:
        size_t kernel_size;
        size_t stride;
        Matrix *last_input;
        std::vector<std::pair<size_t, size_t>> max_indices;
    };

} // namespace nnm