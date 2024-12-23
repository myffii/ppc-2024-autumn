#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <memory>
#include <gtest/gtest.h>
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

double* generate_random_matrix(size_t size) {
    auto* matrix = new double[size * size];
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < size * size; ++i) {
        matrix[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }
    return matrix;
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_2x2) {
auto taskData = std::make_shared<ppc::core::TaskData>();

size_t size = 2;
double* matrixA = generate_random_matrix(size);
double* matrixB = generate_random_matrix(size);

taskData->inputs.push_back(matrixA);
taskData->inputs.push_back(matrixB);
taskData->inputs_count.push_back(size * size);
taskData->inputs_count.push_back(size * size);

nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";

delete[] matrixA;
delete[] matrixB;
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_4x4) {
auto taskData = std::make_shared<ppc::core::TaskData>();

size_t size = 4;
double* matrixA = generate_random_matrix(size);
double* matrixB = generate_random_matrix(size);

taskData->inputs.push_back(matrixA);
taskData->inputs.push_back(matrixB);
taskData->inputs_count.push_back(size * size);
taskData->inputs_count.push_back(size * size);

nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";

delete[] matrixA;
delete[] matrixB;
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_8x8) {
auto taskData = std::make_shared<ppc::core::TaskData>();

size_t size = 8;
double* matrixA = generate_random_matrix(size);
double* matrixB = generate_random_matrix(size);

taskData->inputs.push_back(matrixA);
taskData->inputs.push_back(matrixB);
taskData->inputs_count.push_back(size * size);
taskData->inputs_count.push_back(size * size);

nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";

delete[] matrixA;
delete[] matrixB;
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_16x16) {
auto taskData = std::make_shared<ppc::core::TaskData>();

size_t size = 16;
double* matrixA = generate_random_matrix(size);
double* matrixB = generate_random_matrix(size);

taskData->inputs.push_back(matrixA);
taskData->inputs.push_back(matrixB);
taskData->inputs_count.push_back(size * size);
taskData->inputs_count.push_back(size * size);

nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";

delete[] matrixA;
delete[] matrixB;
}