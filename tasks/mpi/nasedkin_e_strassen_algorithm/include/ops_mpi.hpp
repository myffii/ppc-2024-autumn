#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm {

    class StrassenAlgorithmMPI : public ppc::core::Task {
    public:
        explicit StrassenAlgorithmMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;

    private:
        std::vector<double> matrix_add(const std::vector<double>& matrixA, const std::vector<double>& matrixB, size_t size);
        std::vector<double> matrix_subtract(const std::vector<double>& matrixA, const std::vector<double>& matrixB, size_t size);
        bool power_of_two(size_t number);
        bool matrix_is_square(size_t matrixSize);
        std::vector<double> strassen_multiply(const std::vector<double>& matrixA, const std::vector<double>& matrixB, size_t size);
        std::vector<double> pad_matrix(const std::vector<double>& matrix, size_t original_size, size_t new_size);

        std::vector<double> inputMatrixA;
        std::vector<double> inputMatrixB;
        std::vector<double> outputMatrix;
        size_t matrixSize;
    };

}  // namespace nasedkin_e_strassen_algorithm