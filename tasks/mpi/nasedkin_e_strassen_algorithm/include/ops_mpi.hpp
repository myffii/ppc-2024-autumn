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

class StrassenAlgorithmMPISequential : public ppc::core::Task {
 public:
  explicit StrassenAlgorithmMPISequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> A_;
  std::vector<std::vector<double>> B_;
  std::vector<std::vector<double>> C_;
  size_t n;

  static std::vector<std::vector<double>> brute_force(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
  static void split(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& a, std::vector<std::vector<double>>& b, std::vector<std::vector<double>>& c, std::vector<std::vector<double>>& d);
  std::vector<std::vector<double>> strassen(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
  static std::vector<std::vector<double>> subtract(const std::vector<std::vector<double>>& A,
                                                                            const std::vector<std::vector<double>>& B);
  static std::vector<std::vector<double>> add(const std::vector<std::vector<double>>& A,
                                                                       const std::vector<std::vector<double>>& B);
};

class StrassenAlgorithmMPIParallel : public ppc::core::Task {
 public:
  explicit StrassenAlgorithmMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> A_;
  std::vector<std::vector<double>> B_;
  std::vector<std::vector<double>> C_;
  size_t n;

  std::vector<std::vector<double>> local_A;
  std::vector<std::vector<double>> local_B;
  std::vector<std::vector<double>> local_C;

  std::vector<int> sizes_a;
  std::vector<int> displs_a;
  std::vector<int> sizes_b;
  std::vector<int> displs_b;

  boost::mpi::communicator world;
  static void calculate_distribution(int rows, int num_proc, std::vector<int>& sizes, std::vector<int>& displs);
  static std::vector<std::vector<double>> brute_force(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
  static void split(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& a, std::vector<std::vector<double>>& b, std::vector<std::vector<double>>& c, std::vector<std::vector<double>>& d);
  std::vector<std::vector<double>> strassen(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
  static std::vector<std::vector<double>> subtract(const std::vector<std::vector<double>>& A,
                                                                            const std::vector<std::vector<double>>& B);
  static std::vector<std::vector<double>> add(const std::vector<std::vector<double>>& A,
                                                                       const std::vector<std::vector<double>>& B);
};

}  // namespace nasedkin_e_strassen_algorithm_mpi