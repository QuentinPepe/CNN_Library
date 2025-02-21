#pragma GCC push_options
#pragma GCC target("avx2")
#pragma GCC optimize("O3")


#pragma once

#include <memory>
#include <string>
#include <iostream>

namespace nnm {

    template<typename InputType, typename OutputType>
    class Layer {
    public:
        virtual ~Layer() = default;

        virtual OutputType forward(const InputType &input) = 0;

        virtual std::string get_name() const = 0;

        virtual size_t get_input_size() const = 0;

        virtual size_t get_output_size() const = 0;
    };

} // namespace nnm