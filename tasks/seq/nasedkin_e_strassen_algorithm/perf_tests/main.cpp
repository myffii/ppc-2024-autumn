#include <gtest/gtest.h>
#include <boost/mpi/timer.hpp>
#include "core/perf/include/perf.hpp"
#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"
#include "seq/nasedkin_e_strassen_algorithm/src/ops_seq.cpp"

TEST(nasedkin_e_strassen_algorithm_seq, test_pipeline_run) {
  int matrixSize = 64;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(matrixSize);

  auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq>(taskData);

  std::vector<std::vector<double>> matrixA;
  std::vector<std::vector<double>> matrixB;
  nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(matrixSize, matrixA);
  nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(matrixSize, matrixB);
  strassenTask->set_matrices(matrixA, matrixB);

  ASSERT_TRUE(strassenTask->validation()) << "Validation failed for valid input";

  strassenTask->pre_processing();
  strassenTask->run();
  strassenTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(strassenTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(nasedkin_e_strassen_algorithm_seq, test_task_run) {
  int matrixSize = 64;
  
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(matrixSize);

  auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq>(taskData);

  std::vector<std::vector<double>> matrixA;
  std::vector<std::vector<double>> matrixB;
  nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(matrixSize, matrixA);
  nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(matrixSize, matrixB);
  strassenTask->set_matrices(matrixA, matrixB);

  ASSERT_TRUE(strassenTask->validation()) << "Validation failed for valid input";

  strassenTask->pre_processing();
  strassenTask->run();
  strassenTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(strassenTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
}