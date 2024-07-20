#include <gtest/gtest.h>
#include "Matrix.h"
#include "Vector.h"
#include <cmath>
#include <chrono>

namespace {

    class MatrixConvolutionTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // Code commun de configuration pour chaque test
        }

        void ExpectMatricesEqual(const nnm::Matrix &a, const nnm::Matrix &b, float epsilon = 1e-5f) {
            ASSERT_EQ(a.getRows(), b.getRows());
            ASSERT_EQ(a.getCols(), b.getCols());
            for (size_t i = 0; i < a.getRows(); ++i) {
                for (size_t j = 0; j < a.getCols(); ++j) {
                    EXPECT_NEAR(a(i, j), b(i, j), epsilon);
                }
            }
        }
    };

    TEST_F(MatrixConvolutionTest, Padding) {
        nnm::Matrix m(2, 2);
        m(0, 0) = 1.0f;
        m(0, 1) = 2.0f;
        m(1, 0) = 3.0f;
        m(1, 1) = 4.0f;

        nnm::Matrix padded = m.pad(1, 1);

        EXPECT_EQ(padded.getRows(), 4);
        EXPECT_EQ(padded.getCols(), 4);

        nnm::Matrix expected(4, 4);
        expected(0, 0) = 0.0f;
        expected(0, 1) = 0.0f;
        expected(0, 2) = 0.0f;
        expected(0, 3) = 0.0f;
        expected(1, 0) = 0.0f;
        expected(1, 1) = 1.0f;
        expected(1, 2) = 2.0f;
        expected(1, 3) = 0.0f;
        expected(2, 0) = 0.0f;
        expected(2, 1) = 3.0f;
        expected(2, 2) = 4.0f;
        expected(2, 3) = 0.0f;
        expected(3, 0) = 0.0f;
        expected(3, 1) = 0.0f;
        expected(3, 2) = 0.0f;
        expected(3, 3) = 0.0f;

        ExpectMatricesEqual(padded, expected);
    }

    TEST_F(MatrixConvolutionTest, SubMatrix) {
        nnm::Matrix m(3, 3);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                m(i, j) = static_cast<float>(i * 3 + j + 1);
            }
        }

        nnm::Matrix sub = m.subMatrix(0, 0, 2, 2);

        EXPECT_EQ(sub.getRows(), 2);
        EXPECT_EQ(sub.getCols(), 2);

        nnm::Matrix expected(2, 2);
        expected(0, 0) = 1.0f;
        expected(0, 1) = 2.0f;
        expected(1, 0) = 4.0f;
        expected(1, 1) = 5.0f;

        ExpectMatricesEqual(sub, expected);
    }

    TEST_F(MatrixConvolutionTest, ElementWiseMultiplication) {
        nnm::Matrix a(2, 2);
        a(0, 0) = 1.0f;
        a(0, 1) = 2.0f;
        a(1, 0) = 3.0f;
        a(1, 1) = 4.0f;

        nnm::Matrix b(2, 2);
        b(0, 0) = 2.0f;
        b(0, 1) = 3.0f;
        b(1, 0) = 4.0f;
        b(1, 1) = 5.0f;

        nnm::Matrix c = a.elementWiseMul(b);

        nnm::Matrix expected(2, 2);
        expected(0, 0) = 2.0f;
        expected(0, 1) = 6.0f;
        expected(1, 0) = 12.0f;
        expected(1, 1) = 20.0f;

        ExpectMatricesEqual(c, expected);
    }

    TEST_F(MatrixConvolutionTest, Sum) {
        nnm::Matrix m(2, 2);
        m(0, 0) = 1.0f;
        m(0, 1) = 2.0f;
        m(1, 0) = 3.0f;
        m(1, 1) = 4.0f;

        float sum = m.sum();

        EXPECT_FLOAT_EQ(sum, 10.0f);
    }

    TEST_F(MatrixConvolutionTest, Reshape) {
        nnm::Matrix m(2, 3);
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                m(i, j) = static_cast<float>(i * 3 + j + 1);
            }
        }

        m.reshape(3, 2);

        EXPECT_EQ(m.getRows(), 3);
        EXPECT_EQ(m.getCols(), 2);

        nnm::Matrix expected(3, 2);
        expected(0, 0) = 1.0f;
        expected(0, 1) = 2.0f;
        expected(1, 0) = 3.0f;
        expected(1, 1) = 4.0f;
        expected(2, 0) = 5.0f;
        expected(2, 1) = 6.0f;

        ExpectMatricesEqual(m, expected);
    }


}  // namespace
