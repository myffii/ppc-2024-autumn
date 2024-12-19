#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

namespace nasedkin_e_strassen_algorithm_mpi {

// Генерация случайной матрицы размером n x n
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generate_random_matrix(int n, double min_val = -10.0, double max_val = 10.0) {
  std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
  std::vector<std::vector<double>> B(n, std::vector<double>(n, 0.0));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min_val, max_val);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i][j] = dist(gen);
      B[i][j] = dist(gen);
    }
  }

  return {A, B};
}

}  // namespace nasedkin_e_strassen_algorithm_mpi

// Тест для проверки выполнения алгоритма Штрассена через pipeline
TEST(nasedkin_e_strassen_algorithm_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const size_t matrix_size = 512;
  auto [A, B] = nasedkin_e_strassen_algorithm_mpi::generate_random_matrix(matrix_size);

  std::vector<double> A_flat(matrix_size * matrix_size);
  std::vector<double> B_flat(matrix_size * matrix_size);
  std::vector<double> C_flat(matrix_size * matrix_size, 0.0);

  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      A_flat[i * matrix_size + j] = A[i][j];
      B_flat[i * matrix_size + j] = B[i][j];
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataPar->inputs_count.emplace_back(A_flat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataPar->inputs_count.emplace_back(B_flat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_flat.data()));
    taskDataPar->outputs_count.emplace_back(C_flat.size());
  }

  auto strassenTaskParallel = std::make_shared<nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmMPIParallel>(taskDataPar);
  ASSERT_EQ(strassenTaskParallel->validation(), true);
  strassenTaskParallel->pre_processing();
  strassenTaskParallel->run();
  strassenTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(strassenTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(matrix_size * matrix_size, C_flat.size());
  }
}

// Тест для проверки выполнения алгоритма Штрассена через task_run
TEST(nasedkin_e_strassen_algorithm_mpi, test_task_run) {
  boost::mpi::communicator world;

  const size_t matrix_size = 512;
  auto [A, B] = nasedkin_e_strassen_algorithm_mpi::generate_random_matrix(matrix_size);

  std::vector<double> A_flat(matrix_size * matrix_size);
  std::vector<double> B_flat(matrix_size * matrix_size);
  std::vector<double> C_flat(matrix_size * matrix_size, 0.0);

  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      A_flat[i * matrix_size + j] = A[i][j];
      B_flat[i * matrix_size + j] = B[i][j];
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataPar->inputs_count.emplace_back(A_flat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataPar->inputs_count.emplace_back(B_flat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_flat.data()));
    taskDataPar->outputs_count.emplace_back(C_flat.size());
  }

  auto strassenTaskParallel = std::make_shared<nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmMPIParallel>(taskDataPar);
  ASSERT_EQ(strassenTaskParallel->validation(), true);
  strassenTaskParallel->pre_processing();
  strassenTaskParallel->run();
  strassenTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(strassenTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(matrix_size * matrix_size, C_flat.size());
  }
}