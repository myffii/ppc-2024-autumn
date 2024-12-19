#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <vector>

#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

TEST(StrassenMatrixMultiplication, test_small_matrix) {
  boost::mpi::communicator world;

  int n = 4;
  std::vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> B = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<double> C(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
    taskDataPar->outputs_count.emplace_back(C.size());
  }

  nasedkin_e_strassen_algorithm::StrassenMatrixMultiplicationParallel parallelTask(taskDataPar);
  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result = {80, 70, 60, 50, 240, 214, 188, 162, 400, 358, 316, 274, 560, 502, 444, 386};
    for (int i = 0; i < n * n; ++i) {
      ASSERT_NEAR(C[i], reference_result[i], 1e-6);
    }
  }
}