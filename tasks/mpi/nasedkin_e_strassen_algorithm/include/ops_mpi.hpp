#pragma once

#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm_mpi {

class StrassenSequential : public ppc::core::Task {
 public:
  explicit StrassenSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> C_;
  size_t n;

  void strassen_multiply(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n);
};

class StrassenParallel : public ppc::core::Task {
 public:
  explicit StrassenParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> C_;
  size_t n;

  std::vector<double> local_A;
  std::vector<double> local_B;
  std::vector<double> local_C;

  std::vector<int> sizes_a;
  std::vector<int> displs_a;
  std::vector<int> sizes_b;
  std::vector<int> displs_b;

  boost::mpi::communicator world;

  void strassen_multiply_parallel(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n);
  static void calculate_distribution_a(int rows, int num_proc, std::vector<int>& sizes, std::vector<int>& displs);
  static void calculate_distribution_b(int rows, int num_proc, std::vector<int>& sizes, std::vector<int>& displs);
};

}  // namespace nasedkin_e_strassen_algorithm_mpi