#pragma once

#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm {

class StrassenAlgorithmMPI : public ppc::core::Task {
 public:
  explicit StrassenAlgorithmMPI(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static void generate_random_matrix(int size, std::vector<double>& matrix);
  void set_matrices(const std::vector<double>& A, const std::vector<double>& B);

 private:
  std::vector<double> matrixA, matrixB, resultMatrix;
  boost::mpi::communicator world;

  std::vector<double> strassen_multiply(const std::vector<double>& A, const std::vector<double>& B, int size);
  static std::vector<double> add(const std::vector<double>& A, const std::vector<double>& B, int size);
  static std::vector<double> subtract(const std::vector<double>& A, const std::vector<double>& B, int size);
  static void split_matrix(const std::vector<double>& matrix, int size,
                           std::vector<double>& A11, std::vector<double>& A12,
                           std::vector<double>& A21, std::vector<double>& A22);
  static std::vector<double> merge_matrices(const std::vector<double>& C11, const std::vector<double>& C12,
                                            const std::vector<double>& C21, const std::vector<double>& C22, int size);
};

}  // namespace nasedkin_e_strassen_algorithm