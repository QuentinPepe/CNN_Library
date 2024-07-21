#pragma once

#include "Layer.h"
#include "Tensor4D.h"
#include <cmath>
#include <limits>

namespace nnm {

    class Tanh : public Layer<Tensor4D, Tensor4D> {
    private:
        static constexpr double tiny = 1.0e-300;
        static constexpr double huge = 1.0e300;

        static double tanh_impl(double x) {
            double t, z;
            int32_t jx, ix;

            union {
                double f;
                uint64_t i;
            } u = {x};
            jx = u.i >> 32;
            ix = jx & 0x7fffffff;

            if (ix >= 0x7ff00000) {
                if (jx >= 0)
                    return 1.0 / x + 1.0;  // tanh(+-inf)=+-1
                else
                    return 1.0 / x - 1.0;  // tanh(NaN) = NaN
            }

            if (ix < 0x40360000) {  // |x| < 22
                if (ix < 0x3e300000) {  // |x| < 2**-28
                    if (huge + x > 1.0)
                        return x;
                }
                if (ix >= 0x3ff00000) {  // |x| >= 1
                    t = std::expm1(2.0 * std::fabs(x));
                    z = 1.0 - 2.0 / (t + 2.0);
                } else {
                    t = std::expm1(-2.0 * std::fabs(x));
                    z = -t / (t + 2.0);
                }
            } else {  // |x| >= 22, return +-1
                z = 1.0 - tiny;
            }
            return jx >= 0 ? z : -z;
        }

    public:
        Tanh() = default;

        Tensor4D forward(const Tensor4D &input) override {
            Tensor4D output(input.getBatchSize(), input.getChannels(), input.getHeight(), input.getWidth());

            for (size_t n = 0; n < input.getBatchSize(); ++n) {
                for (size_t c = 0; c < input.getChannels(); ++c) {
                    for (size_t h = 0; h < input.getHeight(); ++h) {
                        for (size_t w = 0; w < input.getWidth(); ++w) {
                            output(n, c, h, w) = static_cast<float>(tanh_impl(input(n, c, h, w)));
                        }
                    }
                }
            }

            return output;
        }

        std::string get_name() const override {
            return "Tanh";
        }

        size_t get_input_size() const override {
            return 0;  // Not applicable for Tanh
        }

        size_t get_output_size() const override {
            return 0;  // Not applicable for Tanh
        }

    };

} // namespace nnm