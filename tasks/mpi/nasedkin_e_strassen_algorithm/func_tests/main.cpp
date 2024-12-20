#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

namespace nasedkin_e_strassen_algorithm_mpi {

// Function to generate random matrices of size n x n
    std::pair<std::vector<double>, std::vector<double>> generate_random_matrices(int n, double min_val = -10.0, double max_val = 10.0) {
        std::vector<double> A(n * n);
        std::vector<double> B(n * n);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(min_val, max_val);

        for (int i = 0; i < n * n; ++i) {
            A[i] = dist(gen);
            B[i] = dist(gen);
        }

        return {A, B};
    }

// Function to calculate the residual for matrix multiplication
    double calculate_residual(const std::vector<double>& A, const std::vector<double>& B, const std::vector<double>& C, size_t n) {
        std::vector<double> AB(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t k = 0; k < n; ++k) {
                    AB[i * n + j] += A[i * n + k] * B[k * n + j];
                }
            }
        }

        double residual = 0.0;
        for (size_t i = 0; i < n * n; ++i) {
            residual += (AB[i] - C[i]) * (AB[i] - C[i]);
        }

        return std::sqrt(residual);
    }

}  // namespace nasedkin_e_strassen_algorithm_mpi

// Function to run tests for a given matrix size
void run_strassen_test_for_matrix_size(size_t matrix_size) {
    boost::mpi::communicator world;

    auto [A_flat, B_flat] = nasedkin_e_strassen_algorithm_mpi::generate_random_matrices(matrix_size);

    std::vector<double> C_parallel(matrix_size * matrix_size, 0.0);
    std::vector<double> C_sequential(matrix_size * matrix_size, 0.0);

    size_t matrix_size_copy = matrix_size;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
        taskDataPar->inputs_count.emplace_back(1);
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
        taskDataPar->inputs_count.emplace_back(A_flat.size());
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
        taskDataPar->inputs_count.emplace_back(B_flat.size());
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_parallel.data()));
        taskDataPar->outputs_count.emplace_back(C_parallel.size());
    }

    nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmParallel strassen_parallel(taskDataPar);
    ASSERT_TRUE(strassen_parallel.validation());
    strassen_parallel.pre_processing();
    strassen_parallel.run();
    strassen_parallel.post_processing();

    if (world.rank() == 0) {
        std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
        taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
        taskDataSeq->inputs_count.emplace_back(1);
        taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
        taskDataSeq->inputs_count.emplace_back(A_flat.size());
        taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
        taskDataSeq->inputs_count.emplace_back(B_flat.size());
        taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_sequential.data()));
        taskDataSeq->outputs_count.emplace_back(C_sequential.size());

        nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmSequential strassen_sequential(taskDataSeq);
        ASSERT_TRUE(strassen_sequential.validation());
        strassen_sequential.pre_processing();
        strassen_sequential.run();
        strassen_sequential.post_processing();
    }

    if (world.rank() == 0) {
        double residual_parallel = nasedkin_e_strassen_algorithm_mpi::calculate_residual(A_flat, B_flat, C_parallel, matrix_size);
        double residual_sequential = nasedkin_e_strassen_algorithm_mpi::calculate_residual(A_flat, B_flat, C_sequential, matrix_size);

        ASSERT_LT(residual_parallel, 1e-3) << "Parallel solution did not match the expected result.";
        ASSERT_LT(residual_sequential, 1e-3) << "Sequential solution did not match the expected result.";
    } else {
        ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
    }
}

// Func tests for matrix sizes 2x2, 4x4, 8x8, 16x16, 32x32
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_2x2) { run_strassen_test_for_matrix_size(2); }
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_4x4) { run_strassen_test_for_matrix_size(4); }
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_8x8) { run_strassen_test_for_matrix_size(8); }
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_16x16) { run_strassen_test_for_matrix_size(16); }
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_32x32) { run_strassen_test_for_matrix_size(32); }