#include <gtest/gtest.h>
#include "Layer.h"

namespace nnm {

    class TestLayer : public Layer {
    public:
        Matrix forward(const Matrix &input) override { return input; }

        Matrix backward(const Matrix &input, const Matrix &output_gradient) override { return output_gradient; }

        void update_parameters(float learning_rate) override {}

        void save(std::ostream &os) const override {}

        void load(std::istream &is) override {}

        std::string get_name() const override { return "TestLayer"; }

        size_t get_input_size() const override { return 10; }

        size_t get_output_size() const override { return 10; }

        std::unique_ptr<Layer> clone() const override { return std::make_unique<TestLayer>(*this); }
    };

    TEST(LayerTest, InterfaceTest) {
        TestLayer layer;

        EXPECT_EQ(layer.get_name(), "TestLayer");
        EXPECT_EQ(layer.get_input_size(), 10);
        EXPECT_EQ(layer.get_output_size(), 10);

        Matrix input(10, 1);
        Matrix output = layer.forward(input);
        EXPECT_EQ(output.getRows(), 10);
        EXPECT_EQ(output.getCols(), 1);

        Matrix gradient(10, 1);
        Matrix input_gradient = layer.backward(input, gradient);
        EXPECT_EQ(input_gradient.getRows(), 10);
        EXPECT_EQ(input_gradient.getCols(), 1);

        layer.update_parameters(0.01f);

        std::unique_ptr<Layer> clone = layer.clone();
        EXPECT_EQ(clone->get_name(), "TestLayer");
    }

} // namespace nnm