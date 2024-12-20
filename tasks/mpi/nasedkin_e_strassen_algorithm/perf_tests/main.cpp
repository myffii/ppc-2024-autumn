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

    std::pair<std::vector<double>, std::vector<double>> generate_random_matrix(int n, double min_val = -10.0, double max_val = 10.0) {
        std::vector<double> A(n * n);
        std::vector<double> B(n * n);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(min_val, max_val);

        for (int i = 0; i < n * n; ++i) {
            A[i] = dist(gen);
            B[i] = dist(gen);
        }

        return {A, B};
    }

    TEST(nasedkin_e_strassen_algorithm_mpi, test_pipeline_run) {
    boost::mpi::communicator world;

    const size_t matrix_size = 512;
    auto [A_flat, B_flat] = generate_random_matrix(matrix_size);

    std::vector<double> C_parallel(matrix_size * matrix_size, 0.0);

    size_t matrix_size_copy = matrix_size;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataPar->inputs_count.emplace_back(A_flat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataPar->inputs_count.emplace_back(B_flat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_parallel.data()));
    taskDataPar->outputs_count.emplace_back(C_parallel.size());
}

auto strassenTaskParallel = std::make_shared<StrassenAlgorithmParallel>(taskDataPar);
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
ASSERT_EQ(matrix_size * matrix_size, C_parallel.size());
}
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_task_run) {
boost::mpi::communicator world;

const size_t matrix_size = 512;
auto [A_flat, B_flat] = generate_random_matrix(matrix_size);

std::vector<double> C_parallel(matrix_size * matrix_size, 0.0);

size_t matrix_size_copy = matrix_size;

std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
if (world.rank() == 0) {
taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
taskDataPar->inputs_count.emplace_back(1);
taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
taskDataPar->inputs_count.emplace_back(A_flat.size());
taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
taskDataPar->inputs_count.emplace_back(B_flat.size());
taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_parallel.data()));
taskDataPar->outputs_count.emplace_back(C_parallel.size());
}

auto strassenTaskParallel = std::make_shared<StrassenAlgorithmParallel>(taskDataPar);
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
ASSERT_EQ(matrix_size * matrix_size, C_parallel.size());
}
}

}  // namespace nasedkin_e_strassen_algorithm_mpi