#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

namespace nasedkin_e_strassen_algorithm {

// Генерация случайной матрицы
std::vector<std::vector<double>> generate_random_matrix(size_t n, double min_val = -100.0, double max_val = 100.0) {
  std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min_val, max_val);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      matrix[i][j] = dist(gen);
    }
  }
  return matrix;
}

}  // namespace nasedkin_e_strassen_algorithm

TEST(nasedkin_e_strassen_algorithm_perf, test_pipeline_run) {
  boost::mpi::communicator world;

  const size_t matrix_size = 128;
  auto A = nasedkin_e_strassen_algorithm::generate_random_matrix(matrix_size);
  auto B = nasedkin_e_strassen_algorithm::generate_random_matrix(matrix_size);

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size() * A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size() * B.size());
  }

  // Создание параллельной задачи
  auto parallelTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenMPITaskParallel>(A, B);
  ASSERT_EQ(parallelTask->validation(), true);
  parallelTask->pre_processing();
  parallelTask->run();
  parallelTask->post_processing();

  // Создание Perf атрибутов
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;  // Количество запусков для усреднения
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Создание и инициализация результатов Perf
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание Perf анализатора
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(nasedkin_e_strassen_algorithm_perf, test_task_run) {
  boost::mpi::communicator world;

  const size_t matrix_size = 128;
  auto A = nasedkin_e_strassen_algorithm::generate_random_matrix(matrix_size);
  auto B = nasedkin_e_strassen_algorithm::generate_random_matrix(matrix_size);

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size() * A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size() * B.size());
  }

  // Создание параллельной задачи
  auto parallelTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenMPITaskParallel>(A, B);
  ASSERT_EQ(parallelTask->validation(), true);
  parallelTask->pre_processing();
  parallelTask->run();
  parallelTask->post_processing();

  // Создание Perf атрибутов
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;  // Количество запусков для усреднения
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Создание и инициализация результатов Perf
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание Perf анализатора
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}