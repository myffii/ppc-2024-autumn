#pragma once

#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm {

class StrassenMatrixMultiplicationSequential : public ppc::core::Task {
 public:
  explicit StrassenMatrixMultiplicationSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> C_;
  int n_;

  static std::vector<double> strassen(const std::vector<double>& A, const std::vector<double>& B, int n);
  static std::vector<double> add(const std::vector<double>& A, const std::vector<double>& B, int n);
  static std::vector<double> subtract(const std::vector<double>& A, const std::vector<double>& B, int n);
};

class StrassenMatrixMultiplicationParallel : public ppc::core::Task {
 public:
  explicit StrassenMatrixMultiplicationParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> C_;
  int n_;
  boost::mpi::communicator world;

  std::vector<double> parallel_strassen(const std::vector<double>& A, const std::vector<double>& B, int n);
  std::vector<double> add(const std::vector<double>& A, const std::vector<double>& B, int n);
  std::vector<double> subtract(const std::vector<double>& A, const std::vector<double>& B, int n);
};

}  // namespace nasedkin_e_strassen_algorithm