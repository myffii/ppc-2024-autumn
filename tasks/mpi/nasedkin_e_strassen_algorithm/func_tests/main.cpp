#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <memory>
#include <gtest/gtest.h>
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

std::vector<double> generate_random_matrix(size_t size) {
    std::vector<double> matrix(size * size);
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < size * size; ++i) {
        matrix[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }
    return matrix;
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_2x2) {
    auto taskData = std::make_shared<ppc::core::TaskData>();

    size_t size = 2;
    std::vector<double> matrixA = generate_random_matrix(size);
    std::vector<double> matrixB = generate_random_matrix(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskData->inputs_count.emplace_back(size * size);
    taskData->inputs_count.emplace_back(size * size);

    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_4x4) {
    auto taskData = std::make_shared<ppc::core::TaskData>();

    size_t size = 4;
    std::vector<double> matrixA = generate_random_matrix(size);
    std::vector<double> matrixB = generate_random_matrix(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskData->inputs_count.emplace_back(size * size);
    taskData->inputs_count.emplace_back(size * size);

    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_8x8) {
    auto taskData = std::make_shared<ppc::core::TaskData>();

    size_t size = 8;
    std::vector<double> matrixA = generate_random_matrix(size);
    std::vector<double> matrixB = generate_random_matrix(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskData->inputs_count.emplace_back(size * size);
    taskData->inputs_count.emplace_back(size * size);

    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_16x16) {
    auto taskData = std::make_shared<ppc::core::TaskData>();

    size_t size = 16;
    std::vector<double> matrixA = generate_random_matrix(size);
    std::vector<double> matrixB = generate_random_matrix(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskData->inputs_count.emplace_back(size * size);
    taskData->inputs_count.emplace_back(size * size);

    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}