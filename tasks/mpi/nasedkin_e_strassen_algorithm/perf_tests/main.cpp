#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <memory>
#include <gtest/gtest.h>
#include <boost/mpi/timer.hpp>
#include "core/task/include/task.hpp"
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

std::vector<double> generate_random_matrix(size_t size) {
    std::vector<double> matrix(size * size);
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < size * size; ++i) {
        matrix[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }
    return matrix;
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_pipeline_run) {
    auto taskData = std::make_shared<ppc::core::TaskData>();

    size_t size = 8;
    std::vector<double> matrixA = generate_random_matrix(size);
    std::vector<double> matrixB = generate_random_matrix(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskData->inputs_count.emplace_back(size * size);
    taskData->inputs_count.emplace_back(size * size);

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
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_task_run) {
    auto taskData = std::make_shared<ppc::core::TaskData>();

    size_t size = 16;
    std::vector<double> matrixA = generate_random_matrix(size);
    std::vector<double> matrixB = generate_random_matrix(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskData->inputs_count.emplace_back(size * size);
    taskData->inputs_count.emplace_back(size * size);

    auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI>(taskData);

    ASSERT_TRUE(strassenTask->validation()) << "Validation failed for valid input";

    const boost::mpi::timer timer;
    strassenTask->pre_processing();
    strassenTask->run();
    strassenTask->post_processing();
    double elapsed_time = timer.elapsed();

    ASSERT_TRUE(strassenTask->pre_processing()) << "Pre-processing failed";
    ASSERT_TRUE(strassenTask->run()) << "Run failed";
    ASSERT_TRUE(strassenTask->post_processing()) << "Post-processing failed";
}
