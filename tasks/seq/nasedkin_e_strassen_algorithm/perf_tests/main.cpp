#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"
#include "seq/nasedkin_e_strassen_algorithm/src/ops_seq.cpp"

TEST(nasedkin_e_strassen_algorithm_seq, test_pipeline_run) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(8);

  auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ>(taskData);

  ASSERT_TRUE(strassenTask->validation()) << "Validation failed for valid input";

  strassenTask->pre_processing();
  strassenTask->run();
  strassenTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  
  auto current_timer = []() {
    static auto start_time = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
  };

  perfAttr->current_timer = current_timer;

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(strassenTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(nasedkin_e_strassen_algorithm_seq, test_task_run) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(8);

  auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ>(taskData);

  ASSERT_TRUE(strassenTask->validation()) << "Validation failed for valid input";

  strassenTask->pre_processing();
  strassenTask->run();
  strassenTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  auto current_timer = []() {
    static auto start_time = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
  };

  perfAttr->current_timer = current_timer;

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(strassenTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
}
