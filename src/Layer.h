#pragma once

#include "Matrix.h"
#include "Vector.h"
#include <memory>
#include <string>

namespace nnm {

    class Layer {
    public:
        virtual ~Layer() = default;

        virtual Matrix forward(const Matrix &input) = 0;

        virtual Matrix backward(const Matrix &input, const Matrix &output_gradient) = 0;

        virtual void update_parameters(float learning_rate) = 0;

        virtual void save(std::ostream &os) const = 0;

        virtual void load(std::istream &is) = 0;

        virtual std::string get_name() const = 0;

        virtual size_t get_input_size() const = 0;

        virtual size_t get_output_size() const = 0;

        virtual std::unique_ptr<Layer> clone() const = 0;
    };

} // namespace nnm