#include <gtest/gtest.h>
#include <boost/mpi.hpp>
#include <vector>
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_mpi_4x4) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(4);

  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

  if (world.rank() == 0) {
    std::vector<std::vector<double>> matrixA;
    std::vector<std::vector<double>> matrixB;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(4, matrixA);
    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(4, matrixB);
    strassen_task.set_matrices(matrixA, matrixB);
  }

  ASSERT_TRUE(strassen_task.validation()) << "Validation failed";
  ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed";
  ASSERT_TRUE(strassen_task.run()) << "Run failed";
  ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed";
}
