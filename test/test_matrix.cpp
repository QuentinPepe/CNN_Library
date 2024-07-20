#include <gtest/gtest.h>
#include "Matrix.h"
#include <cmath>
#include <chrono>

namespace {

    class MatrixTest : public ::testing::Test {
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

    TEST_F(MatrixTest, Constructor) {
        nnm::Matrix m(3, 4);
        EXPECT_EQ(m.getRows(), 3);
        EXPECT_EQ(m.getCols(), 4);
    }

    TEST_F(MatrixTest, AccessOperator) {
        nnm::Matrix m(2, 2);
        m(0, 0) = 1.0f;
        m(0, 1) = 2.0f;
        m(1, 0) = 3.0f;
        m(1, 1) = 4.0f;

        EXPECT_FLOAT_EQ(m(0, 0), 1.0f);
        EXPECT_FLOAT_EQ(m(0, 1), 2.0f);
        EXPECT_FLOAT_EQ(m(1, 0), 3.0f);
        EXPECT_FLOAT_EQ(m(1, 1), 4.0f);
    }

    TEST_F(MatrixTest, MultiplicationSmall) {
        nnm::Matrix a(2, 3);
        nnm::Matrix b(3, 2);

        a(0, 0) = 1.0f;
        a(0, 1) = 2.0f;
        a(0, 2) = 3.0f;
        a(1, 0) = 4.0f;
        a(1, 1) = 5.0f;
        a(1, 2) = 6.0f;

        b(0, 0) = 7.0f;
        b(0, 1) = 8.0f;
        b(1, 0) = 9.0f;
        b(1, 1) = 10.0f;
        b(2, 0) = 11.0f;
        b(2, 1) = 12.0f;

        nnm::Matrix c = a * b;

        EXPECT_EQ(c.getRows(), 2);
        EXPECT_EQ(c.getCols(), 2);
        EXPECT_FLOAT_EQ(c(0, 0), 58.0f);
        EXPECT_FLOAT_EQ(c(0, 1), 64.0f);
        EXPECT_FLOAT_EQ(c(1, 0), 139.0f);
        EXPECT_FLOAT_EQ(c(1, 1), 154.0f);
    }

    TEST_F(MatrixTest, MultiplicationInvalidDimensions) {
        nnm::Matrix a(2, 3);
        nnm::Matrix b(2, 2);

        EXPECT_THROW(a * b, std::invalid_argument);
    }


    TEST_F(MatrixTest, MultiplicationPerformance) {
        const size_t size = 1000;
        nnm::Matrix a(size, size);
        nnm::Matrix b(size, size);

        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                a(i, j) = static_cast<float>(i + j);
                b(i, j) = static_cast<float>(i - j);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        nnm::Matrix c = a * b;
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        std::cout << "Time to multiply two " << size << "x" << size << " matrices: " << diff.count() << " s"
                  << std::endl;

        EXPECT_LT(diff.count(), 5.0);
    }

    TEST_F(MatrixTest, MultiplicationSquareMatrices) {
        nnm::Matrix a(3, 3);
        nnm::Matrix b(3, 3);

        // Initialize matrices
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                a(i, j) = static_cast<float>(i * 3 + j + 1);
                b(i, j) = static_cast<float>(j * 3 + i + 1);
            }
        }

        nnm::Matrix c = a * b;

        // Expected result
        nnm::Matrix expected(3, 3);
        expected(0, 0) = 14.0f;
        expected(0, 1) = 32.0f;
        expected(0, 2) = 50.0f;
        expected(1, 0) = 32.0f;
        expected(1, 1) = 77.0f;
        expected(1, 2) = 122.0f;
        expected(2, 0) = 50.0f;
        expected(2, 1) = 122.0f;
        expected(2, 2) = 194.0f;

        ExpectMatricesEqual(c, expected);
    }

    TEST_F(MatrixTest, MultiplicationNonSquareMatrices) {
        nnm::Matrix a(2, 4);
        nnm::Matrix b(4, 3);

        // Initialize matrices
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                a(i, j) = static_cast<float>(i * 4 + j + 1);
            }
        }
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                b(i, j) = static_cast<float>(i * 3 + j + 1);
            }
        }

        nnm::Matrix c = a * b;

        // Expected result
        nnm::Matrix expected(2, 3);
        expected(0, 0) = 70.0f;
        expected(0, 1) = 80.0f;
        expected(0, 2) = 90.0f;
        expected(1, 0) = 158.0f;
        expected(1, 1) = 184.0f;
        expected(1, 2) = 210.0f;

        ExpectMatricesEqual(c, expected);
    }


    TEST_F(MatrixTest, MultiplicationPerformanceSquare) {
        const size_t size = 1000;
        nnm::Matrix a(size, size);
        nnm::Matrix b(size, size);

        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                a(i, j) = static_cast<float>(i + j);
                b(i, j) = static_cast<float>(i - j);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        nnm::Matrix c = a * b;
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        std::cout << "Time to multiply two " << size << "x" << size << " matrices: " << diff.count() << " s"
                  << std::endl;

        EXPECT_LT(diff.count(), 5.0);
    }

    TEST_F(MatrixTest, MultiplicationPerformanceNonSquare) {
        const size_t m = 1000, n = 800, p = 1200;
        nnm::Matrix a(m, n);
        nnm::Matrix b(n, p);

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                a(i, j) = static_cast<float>(i + j);
            }
        }
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < p; ++j) {
                b(i, j) = static_cast<float>(i - j);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        nnm::Matrix c = a * b;
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        std::cout << "Time to multiply " << m << "x" << n << " and " << n << "x" << p << " matrices: " << diff.count()
                  << " s" << std::endl;

        EXPECT_LT(diff.count(), 5.0);
    }


}  // namespace
