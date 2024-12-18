#include <gtest/gtest.h>

#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_2x2) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(2);

  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

  std::vector<std::vector<double>> matrixA;
  std::vector<std::vector<double>> matrixB;
  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(2, matrixA);
  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(2, matrixB);
  strassen_task.set_matrices(matrixA, matrixB);

  if (world.rank() == 0) {
    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
  }

  ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
  ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
  ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}