#pragma once

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <vector>
#include <memory>
#include <string>

namespace nasedkin_e_strassen_algorithm {

class StrassenMPITaskSequential : public ppc::core::Task {
 public:
  explicit StrassenMPITaskSequential(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B);
  std::vector<std::vector<double>> run();
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> A_, B_;
  std::vector<std::vector<double>> strassen(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
  std::vector<std::vector<double>> brute_force(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
  std::vector<std::vector<double>> add(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
  std::vector<std::vector<double>> subtract(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
  std::vector<std::vector<double>> combine(const std::vector<std::vector<double>>& C11, const std::vector<std::vector<double>>& C12, const std::vector<std::vector<double>>& C21, const std::vector<std::vector<double>>& C22);
  std::vector<std::vector<double>> split(const std::vector<std::vector<double>>& matrix, int row_start, int col_start, int size);
};

class StrassenMPITaskParallel : public ppc::core::Task {
 public:
  explicit StrassenMPITaskParallel(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B);
  std::vector<std::vector<double>> run();
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> A_, B_;
  boost::mpi::communicator world;
  std::vector<std::vector<double>> strassen_parallel(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
  std::vector<std::vector<double>> brute_force(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
  std::vector<std::vector<double>> add(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
  std::vector<std::vector<double>> subtract(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
  std::vector<std::vector<double>> combine(const std::vector<std::vector<double>>& C11, const std::vector<std::vector<double>>& C12, const std::vector<std::vector<double>>& C21, const std::vector<std::vector<double>>& C22);
  std::vector<std::vector<double>> split(const std::vector<std::vector<double>>& matrix, int row_start, int col_start, int size);
};

}  // namespace nasedkin_e_strassen_algorithm