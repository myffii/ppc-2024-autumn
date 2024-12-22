#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <memory>
#include <gtest/gtest.h>
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

TEST(nasedkin_e_strassen_algorithm_mpi, test_pipeline_run) {
auto taskData = std::make_shared<ppc::core::TaskData>();

size_t size = 8;
double* matrixA = generate_random_matrix(size);
double* matrixB = generate_random_matrix(size);

taskData->inputs.push_back(matrixA);
taskData->inputs.push_back(matrixB);
taskData->inputs_count.push_back(size * size);
taskData->inputs_count.push_back(size * size);

auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI>(taskData);

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

// Очистка памяти
delete[] matrixA;
delete[] matrixB;
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_task_run) {
auto taskData = std::make_shared<ppc::core::TaskData>();

size_t size = 8;
double* matrixA = generate_random_matrix(size);
double* matrixB = generate_random_matrix(size);

taskData->inputs.push_back(matrixA);
taskData->inputs.push_back(matrixB);
taskData->inputs_count.push_back(size * size);
taskData->inputs_count.push_back(size * size);

auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI>(taskData);

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

delete[] matrixA;
delete[] matrixB;
}