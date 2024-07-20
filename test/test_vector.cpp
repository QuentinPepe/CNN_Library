#include <gtest/gtest.h>
#include "Vector.h"
#include <cmath>
#include <chrono>

namespace {

    class VectorTest : public ::testing::Test {
    protected:
        void SetUp() override {}
    };

    TEST_F(VectorTest, Constructor) {
        nnm::Vector v(5);
        EXPECT_EQ(v.size(), 5);
    }

    TEST_F(VectorTest, InitializerListConstructor) {
        nnm::Vector v{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        EXPECT_EQ(v.size(), 5);
        EXPECT_FLOAT_EQ(v[0], 1.0f);
        EXPECT_FLOAT_EQ(v[4], 5.0f);
    }

    TEST_F(VectorTest, Addition) {
        nnm::Vector v1{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        nnm::Vector v2{5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        nnm::Vector result = v1 + v2;

        EXPECT_EQ(result.size(), 5);
        for (int i = 0; i < 5; ++i) {
            EXPECT_FLOAT_EQ(result[i], 6.0f);
        }
    }

    TEST_F(VectorTest, DotProduct) {
        nnm::Vector v1{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        nnm::Vector v2{5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        float dot = v1.dot(v2);

        EXPECT_FLOAT_EQ(dot, 35.0f);
    }

    TEST_F(VectorTest, Norm) {
        nnm::Vector v{3.0f, 4.0f};
        float norm = v.norm();

        EXPECT_FLOAT_EQ(norm, 5.0f);
    }

    TEST_F(VectorTest, ScalarMultiplication) {
        nnm::Vector v{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        nnm::Vector result = v * 2.0f;

        EXPECT_EQ(result.size(), 5);
        for (int i = 0; i < 5; ++i) {
            EXPECT_FLOAT_EQ(result[i], v[i] * 2.0f);
        }
    }

    TEST_F(VectorTest, LargeVectorOperations) {
        const size_t size = 1000000;
        nnm::Vector v1(size);
        nnm::Vector v2(size);

        for (size_t i = 0; i < size; ++i) {
            v1[i] = static_cast<float>(i);
            v2[i] = static_cast<float>(size - i);
        }

        auto start = std::chrono::high_resolution_clock::now();
        float dot = v1.dot(v2);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        std::cout << "Time to compute dot product of two vectors of size " << size << ": " << diff.count() << " s"
                  << std::endl;

        // Le résultat exact dépendra de la taille, mais nous pouvons vérifier qu'il est non nul
        EXPECT_NE(dot, 0.0f);
    }

}  // namespace

