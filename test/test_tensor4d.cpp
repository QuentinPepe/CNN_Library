#include <gtest/gtest.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include "Tensor4D.h"

using json = nlohmann::json;

class Tensor4DTest : public ::testing::Test {
protected:
    json test_cases;

    void SetUp() override {
        std::ifstream f("tensor4d_test_cases.json");
        f >> test_cases;
    }

    nnm::Tensor4D create_tensor_from_json(const json &j) {
        if (!j.is_array()) {
            throw std::runtime_error("JSON input is not an array");
        }

        size_t batch_size = j.size();
        if (batch_size == 0) {
            throw std::runtime_error("Empty JSON array");
        }

        size_t channels = j[0].size();
        if (channels == 0) {
            throw std::runtime_error("Empty channel in JSON array");
        }

        size_t height = j[0][0].size();
        if (height == 0) {
            throw std::runtime_error("Empty height in JSON array");
        }

        size_t width = j[0][0][0].size();

        nnm::Tensor4D tensor(batch_size, channels, height, width);

        for (size_t n = 0; n < batch_size; ++n) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        tensor(n, c, h, w) = j[n][c][h][w].get<float>();
                    }
                }
            }
        }

        return tensor;
    }

    json tensor_to_json(const nnm::Tensor4D &tensor) {
        json j = json::array();
        for (size_t n = 0; n < tensor.getBatchSize(); ++n) {
            json batch = json::array();
            for (size_t c = 0; c < tensor.getChannels(); ++c) {
                json channel = json::array();
                for (size_t h = 0; h < tensor.getHeight(); ++h) {
                    json row = json::array();
                    for (size_t w = 0; w < tensor.getWidth(); ++w) {
                        row.push_back(tensor(n, c, h, w));
                    }
                    channel.push_back(row);
                }
                batch.push_back(channel);
            }
            j.push_back(batch);
        }
        return j;
    }

    bool tensor_almost_equal(const nnm::Tensor4D &a, const nnm::Tensor4D &b, float tolerance = 1e-5f) {
        if (a.getBatchSize() != b.getBatchSize() || a.getChannels() != b.getChannels() ||
            a.getHeight() != b.getHeight() || a.getWidth() != b.getWidth()) {
            return false;
        }
        for (size_t n = 0; n < a.getBatchSize(); ++n) {
            for (size_t c = 0; c < a.getChannels(); ++c) {
                for (size_t h = 0; h < a.getHeight(); ++h) {
                    for (size_t w = 0; w < a.getWidth(); ++w) {
                        if (std::abs(a(n, c, h, w) - b(n, c, h, w)) > tolerance) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

};

TEST_F(Tensor4DTest, Constructor) {

}

TEST_F(Tensor4DTest, Addition) {
    for (const auto &test: test_cases) {
        if (test["test_name"] == "addition") {
            auto t1 = create_tensor_from_json(test["input"]["t1"]);
            auto t2 = create_tensor_from_json(test["input"]["t2"]);

            auto result = t1 + t2;

            auto expected = create_tensor_from_json(test["output"]["result"]);
            EXPECT_TRUE(tensor_almost_equal(result, expected));
            EXPECT_NEAR(result.sum(), test["output"]["sum"].get<float>(), 1e-5f);
        }
    }
}

TEST_F(Tensor4DTest, Subtraction) {
    for (const auto &test: test_cases) {
        if (test["test_name"] == "subtraction") {
            auto t1 = create_tensor_from_json(test["input"]["t1"]);
            auto t2 = create_tensor_from_json(test["input"]["t2"]);

            auto result = t1 - t2;

            auto expected = create_tensor_from_json(test["output"]["result"]);
            EXPECT_TRUE(tensor_almost_equal(result, expected));
            EXPECT_NEAR(result.sum(), test["output"]["sum"].get<float>(), 1e-5f);
        }
    }
}

TEST_F(Tensor4DTest, ElementWiseMultiplication) {
    for (const auto &test: test_cases) {
        if (test["test_name"] == "element_wise_multiplication") {
            auto t1 = create_tensor_from_json(test["input"]["t1"]);
            auto t2 = create_tensor_from_json(test["input"]["t2"]);

            auto result = t1.elementWiseMul(t2);

            auto expected = create_tensor_from_json(test["output"]["result"]);
            EXPECT_TRUE(tensor_almost_equal(result, expected));
            EXPECT_NEAR(result.sum(), test["output"]["sum"].get<float>(), 1e-5f);
        }
    }
}

TEST_F(Tensor4DTest, Padding) {
    for (const auto &test: test_cases) {
        if (test["test_name"] == "padding") {
            auto tensor = create_tensor_from_json(test["input"]["tensor"]);
            int pad_h = test["input"]["pad_h"];
            int pad_w = test["input"]["pad_w"];

            auto result = tensor.pad(pad_h, pad_w);

            auto expected = create_tensor_from_json(test["output"]["result"]);
            EXPECT_TRUE(tensor_almost_equal(result, expected));
            EXPECT_EQ(result.getBatchSize(), expected.getBatchSize());
            EXPECT_EQ(result.getChannels(), expected.getChannels());
            EXPECT_EQ(result.getHeight(), expected.getHeight());
            EXPECT_EQ(result.getWidth(), expected.getWidth());
        }
    }
}

TEST_F(Tensor4DTest, SubTensor) {
    for (const auto &test: test_cases) {
        if (test["test_name"] == "sub_tensor") {
            auto tensor = create_tensor_from_json(test["input"]["tensor"]);
            auto input = test["input"];

            auto result = tensor.subTensor(
                    input["start_n"], input["start_c"], input["start_h"], input["start_w"],
                    input["sub_batch"], input["sub_channels"], input["sub_height"], input["sub_width"]
            );

            auto expected = create_tensor_from_json(test["output"]["result"]);
            EXPECT_TRUE(tensor_almost_equal(result, expected));
            EXPECT_EQ(result.getBatchSize(), expected.getBatchSize());
            EXPECT_EQ(result.getChannels(), expected.getChannels());
            EXPECT_EQ(result.getHeight(), expected.getHeight());
            EXPECT_EQ(result.getWidth(), expected.getWidth());
        }
    }
}

// Test constructor with default value
TEST_F(Tensor4DTest, ConstructorWithDefaultValue) {
    nnm::Tensor4D tensor(2, 3, 4, 5);
    for (size_t n = 0; n < tensor.getBatchSize(); ++n) {
        for (size_t c = 0; c < tensor.getChannels(); ++c) {
            for (size_t h = 0; h < tensor.getHeight(); ++h) {
                for (size_t w = 0; w < tensor.getWidth(); ++w) {
                    EXPECT_FLOAT_EQ(tensor(n, c, h, w), 0.0f);
                }
            }
        }
    }
}

// Test constructor with specific value
TEST_F(Tensor4DTest, ConstructorWithSpecificValue) {
    nnm::Tensor4D tensor(2, 3, 4, 5, 1.5f);
    for (size_t n = 0; n < tensor.getBatchSize(); ++n) {
        for (size_t c = 0; c < tensor.getChannels(); ++c) {
            for (size_t h = 0; h < tensor.getHeight(); ++h) {
                for (size_t w = 0; w < tensor.getWidth(); ++w) {
                    EXPECT_FLOAT_EQ(tensor(n, c, h, w), 1.5f);
                }
            }
        }
    }
}

// Test element-wise addition
TEST_F(Tensor4DTest, Addition2) {
    nnm::Tensor4D tensor1(2, 2, 2, 2, 1.0f);
    nnm::Tensor4D tensor2(2, 2, 2, 2, 2.0f);
    nnm::Tensor4D result = tensor1 + tensor2;

    for (size_t n = 0; n < tensor1.getBatchSize(); ++n) {
        for (size_t c = 0; c < tensor1.getChannels(); ++c) {
            for (size_t h = 0; h < tensor1.getHeight(); ++h) {
                for (size_t w = 0; w < tensor1.getWidth(); ++w) {
                    EXPECT_FLOAT_EQ(result(n, c, h, w), 3.0f);
                }
            }
        }
    }
}

// Test element-wise subtraction
TEST_F(Tensor4DTest, Subtraction2) {
    nnm::Tensor4D tensor1(2, 2, 2, 2, 3.0f);
    nnm::Tensor4D tensor2(2, 2, 2, 2, 1.0f);
    nnm::Tensor4D result = tensor1 - tensor2;

    for (size_t n = 0; n < tensor1.getBatchSize(); ++n) {
        for (size_t c = 0; c < tensor1.getChannels(); ++c) {
            for (size_t h = 0; h < tensor1.getHeight(); ++h) {
                for (size_t w = 0; w < tensor1.getWidth(); ++w) {
                    EXPECT_FLOAT_EQ(result(n, c, h, w), 2.0f);
                }
            }
        }
    }
}

// Test element-wise multiplication
TEST_F(Tensor4DTest, ElementWiseMultiplication2) {
    nnm::Tensor4D tensor1(2, 2, 2, 2, 2.0f);
    nnm::Tensor4D tensor2(2, 2, 2, 2, 3.0f);
    nnm::Tensor4D result = tensor1.elementWiseMul(tensor2);

    for (size_t n = 0; n < tensor1.getBatchSize(); ++n) {
        for (size_t c = 0; c < tensor1.getChannels(); ++c) {
            for (size_t h = 0; h < tensor1.getHeight(); ++h) {
                for (size_t w = 0; w < tensor1.getWidth(); ++w) {
                    EXPECT_FLOAT_EQ(result(n, c, h, w), 6.0f);
                }
            }
        }
    }
}

// Test sum of elements
TEST_F(Tensor4DTest, Sum) {
    nnm::Tensor4D tensor(2, 2, 2, 2, 1.0f);
    EXPECT_FLOAT_EQ(tensor.sum(), 16.0f);
}

// Test fill method
TEST_F(Tensor4DTest, Fill) {
    nnm::Tensor4D tensor(2, 2, 2, 2);
    tensor.fill(3.0f);
    for (size_t n = 0; n < tensor.getBatchSize(); ++n) {
        for (size_t c = 0; c < tensor.getChannels(); ++c) {
            for (size_t h = 0; h < tensor.getHeight(); ++h) {
                for (size_t w = 0; w < tensor.getWidth(); ++w) {
                    EXPECT_FLOAT_EQ(tensor(n, c, h, w), 3.0f);
                }
            }
        }
    }
}

// Test padding
TEST_F(Tensor4DTest, Padding2) {
    nnm::Tensor4D tensor(1, 1, 2, 2, 1.0f);
    nnm::Tensor4D padded = tensor.pad(1, 1);

    EXPECT_EQ(padded.getBatchSize(), 1);
    EXPECT_EQ(padded.getChannels(), 1);
    EXPECT_EQ(padded.getHeight(), 4);
    EXPECT_EQ(padded.getWidth(), 4);

    for (size_t h = 1; h < 3; ++h) {
        for (size_t w = 1; w < 3; ++w) {
            EXPECT_FLOAT_EQ(padded(0, 0, h, w), 1.0f);
        }
    }
    for (size_t h = 0; h < 4; ++h) {
        EXPECT_FLOAT_EQ(padded(0, 0, h, 0), 0.0f);
        EXPECT_FLOAT_EQ(padded(0, 0, h, 3), 0.0f);
    }
    for (size_t w = 0; w < 4; ++w) {
        EXPECT_FLOAT_EQ(padded(0, 0, 0, w), 0.0f);
        EXPECT_FLOAT_EQ(padded(0, 0, 3, w), 0.0f);
    }
}

// Test subTensor
TEST_F(Tensor4DTest, SubTensor2) {
    nnm::Tensor4D tensor(2, 2, 4, 4, 1.0f);
    nnm::Tensor4D sub = tensor.subTensor(0, 0, 1, 1, 1, 1, 2, 2);

    EXPECT_EQ(sub.getBatchSize(), 1);
    EXPECT_EQ(sub.getChannels(), 1);
    EXPECT_EQ(sub.getHeight(), 2);
    EXPECT_EQ(sub.getWidth(), 2);

    for (size_t n = 0; n < sub.getBatchSize(); ++n) {
        for (size_t c = 0; c < sub.getChannels(); ++c) {
            for (size_t h = 0; h < sub.getHeight(); ++h) {
                for (size_t w = 0; w < sub.getWidth(); ++w) {
                    EXPECT_FLOAT_EQ(sub(n, c, h, w), 1.0f);
                }
            }
        }
    }
}


TEST_F(Tensor4DTest, ChannelToMatrix) {
    nnm::Tensor4D tensor(2, 3, 4, 4);
    for (size_t n = 0; n < 2; ++n) {
        for (size_t c = 0; c < 3; ++c) {
            for (size_t h = 0; h < 4; ++h) {
                for (size_t w = 0; w < 4; ++w) {
                    tensor(n, c, h, w) = static_cast<float>(n * 100 + c * 10 + h * 4 + w);
                }
            }
        }
    }

    size_t batch_index = 0;
    size_t channel_index = 1;

    nnm::Matrix result = tensor.channelToMatrix(tensor, batch_index, channel_index);

    EXPECT_EQ(result.getRows(), 4);
    EXPECT_EQ(result.getCols(), 4);

    for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 4; ++w) {
            EXPECT_FLOAT_EQ(result(h, w), tensor(batch_index, channel_index, h, w));
            EXPECT_FLOAT_EQ(result(h, w), static_cast<float>(10 + h * 4 + w));
        }
    }
}

TEST_F(Tensor4DTest, Padding3) {
    nnm::Tensor4D tensor(2, 2, 3, 3, 1.0f);

    std::vector<std::pair<size_t, size_t>> padding = {
            {1, 1},  // Batch dimension: pad 1 before and 1 after
            {0, 2},  // Channel dimension: pad 0 before and 2 after
            {2, 1},  // Height dimension: pad 2 before and 1 after
            {1, 2}   // Width dimension: pad 1 before and 2 after
    };

    nnm::Tensor4D padded = tensor.pad(padding);

    EXPECT_EQ(padded.getBatchSize(), 4);   // 2 + 1 + 1
    EXPECT_EQ(padded.getChannels(), 4);    // 2 + 0 + 2
    EXPECT_EQ(padded.getHeight(), 6);      // 3 + 2 + 1
    EXPECT_EQ(padded.getWidth(), 6);       // 3 + 1 + 2

    for (size_t n = 0; n < padded.getBatchSize(); ++n) {
        for (size_t c = 0; c < padded.getChannels(); ++c) {
            for (size_t h = 0; h < padded.getHeight(); ++h) {
                for (size_t w = 0; w < padded.getWidth(); ++w) {
                    if (n > 0 && n < 3 && c < 2 && h > 1 && h < 5 && w > 0 && w < 4) {
                        EXPECT_FLOAT_EQ(padded(n, c, h, w), 1.0f);
                    } else {
                        // Padded areas
                        EXPECT_FLOAT_EQ(padded(n, c, h, w), 0.0f);
                    }
                }
            }
        }
    }
}


TEST_F(Tensor4DTest, ComprehensivePadding) {
    nnm::Tensor4D tensor(2, 2, 3, 3, 1.0f);

    std::vector<std::pair<size_t, size_t>> padding = {
            {1, 1},  // Batch dimension: pad 1 before and 1 after
            {0, 2},  // Channel dimension: pad 0 before and 2 after
            {2, 1},  // Height dimension: pad 2 before and 1 after
            {1, 2}   // Width dimension: pad 1 before and 2 after
    };

    nnm::Tensor4D padded = tensor.pad(padding);

    EXPECT_EQ(padded.getBatchSize(), 4);
    EXPECT_EQ(padded.getChannels(), 4);
    EXPECT_EQ(padded.getHeight(), 6);
    EXPECT_EQ(padded.getWidth(), 6);

    auto isOriginalValue = [](size_t n, size_t c, size_t h, size_t w) {
        return (n == 1 || n == 2) && c < 2 && h >= 2 && h < 5 && w >= 1 && w < 4;
    };

    for (size_t n = 0; n < padded.getBatchSize(); ++n) {
        for (size_t c = 0; c < padded.getChannels(); ++c) {
            for (size_t h = 0; h < padded.getHeight(); ++h) {
                for (size_t w = 0; w < padded.getWidth(); ++w) {
                    if (isOriginalValue(n, c, h, w)) {
                        EXPECT_FLOAT_EQ(padded(n, c, h, w), 1.0f)
                                            << "Mismatch at position (" << n << ", " << c << ", " << h << ", " << w
                                            << ")";
                    } else {
                        EXPECT_FLOAT_EQ(padded(n, c, h, w), 0.0f)
                                            << "Mismatch at position (" << n << ", " << c << ", " << h << ", " << w
                                            << ")";
                    }
                }
            }
        }
    }

    // Additional specific checks based on the NumPy output
    EXPECT_FLOAT_EQ(padded(1, 0, 2, 1), 1.0f);
    EXPECT_FLOAT_EQ(padded(1, 1, 4, 3), 1.0f);
    EXPECT_FLOAT_EQ(padded(2, 1, 3, 2), 1.0f);
    EXPECT_FLOAT_EQ(padded(0, 0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(padded(3, 3, 5, 5), 0.0f);
}