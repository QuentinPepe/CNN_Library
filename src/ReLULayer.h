#pragma once

#include "Layer.h"

namespace nnm {

    class ReLULayer : public Layer {
    public:
        ReLULayer();

        ~ReLULayer() override;

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
        Matrix *last_input;
    };

} // namespace nnm