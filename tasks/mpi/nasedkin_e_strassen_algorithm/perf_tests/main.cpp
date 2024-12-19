#include <gtest/gtest.h>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/timer.hpp>

#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"
#include "core/perf/include/perf.hpp"

TEST(nasedkin_e_strassen_algorithm_mpi, perf_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int matrix_size = 8;
  if (world.rank() == 0) {
    std::cout << "Matrix size: " << matrix_size << std::endl;
    std::cout << "Number of processes: " << world.size() << std::endl;
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(matrix_size);

  auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI>(taskData);

  std::vector<std::vector<double>> matrixA, matrixB;
  if (world.rank() == 0) {
    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(matrix_size, matrixA);
    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(matrix_size, matrixB);
  }

  boost::mpi::broadcast(world, matrixA, 0);
  boost::mpi::broadcast(world, matrixB, 0);

  strassenTask->set_matrices(matrixA, matrixB);

  ASSERT_TRUE(strassenTask->validation()) << "Validation failed for valid input";

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  boost::mpi::timer timer;
  perfAttr->current_timer = [&] { return timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(strassenTask);

  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(nasedkin_e_strassen_algorithm_mpi, perf_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int matrix_size = 8;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(matrix_size);

  auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI>(taskData);

  std::vector<std::vector<double>> matrixA, matrixB;
  if (world.rank() == 0) {
    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(matrix_size, matrixA);
    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(matrix_size, matrixB);
  }

  boost::mpi::broadcast(world, matrixA, 0);
  boost::mpi::broadcast(world, matrixB, 0);

  strassenTask->set_matrices(matrixA, matrixB);

  ASSERT_TRUE(strassenTask->validation()) << "Validation failed for valid input";

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  boost::mpi::timer timer;
  perfAttr->current_timer = [&] { return timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(strassenTask);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
