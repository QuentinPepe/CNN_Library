#pragma once

#include <memory>
#include <string>
#include <iostream>

namespace nnm {

    template<typename T>
    class Layer {
    public:
        virtual ~Layer() = default;

        virtual T forward(const T &input) = 0;

        virtual T backward(const T &input, const T &output_gradient) = 0;

        virtual void update_parameters(float learning_rate) = 0;

        virtual void save(std::ostream &os) const = 0;

        virtual void load(std::istream &is) = 0;

        virtual std::string get_name() const = 0;

        virtual size_t get_input_size() const = 0;

        virtual size_t get_output_size() const = 0;

        virtual std::unique_ptr<Layer<T>> clone() const = 0;
    };

} // namespace nnm