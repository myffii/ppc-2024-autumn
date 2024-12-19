#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

TEST(StrassenMatrixMultiplication, test_pipeline_run) {
  boost::mpi::communicator world;

  int n = 1024;
  std::vector<double> A(n * n, 1.0);
  std::vector<double> B(n * n, 1.0);
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

  auto parallelTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenMatrixMultiplicationParallel>(taskDataPar);
  ASSERT_TRUE(parallelTask->validation());
  parallelTask->pre_processing();
  parallelTask->run();
  parallelTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(StrassenMatrixMultiplication, test_task_run) {
  boost::mpi::communicator world;

  int n = 1024;
  std::vector<double> A(n * n, 1.0);
  std::vector<double> B(n * n, 1.0);
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

  auto parallelTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenMatrixMultiplicationParallel>(taskDataPar);
  ASSERT_TRUE(parallelTask->validation());
  parallelTask->pre_processing();
  parallelTask->run();
  parallelTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}