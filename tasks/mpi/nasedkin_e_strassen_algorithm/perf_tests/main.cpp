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

// Function to generate random matrices of size n x n
    std::pair<std::vector<double>, std::vector<double>> generate_random_matrices(int n, double min_val = -10.0, double max_val = 10.0) {
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

}  // namespace nasedkin_e_strassen_algorithm_mpi

// Perf tests
TEST(nasedkin_e_strassen_algorithm_mpi, test_pipeline_run) {
boost::mpi::communicator world;

const size_t matrix_size = 512;
auto [A_flat, B_flat] = nasedkin_e_strassen_algorithm_mpi::generate_random_matrices(matrix_size);

std::vector<size_t> in_size(1, matrix_size);
std::vector<double> out(matrix_size * matrix_size, 0.0);

std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
if (world.rank() == 0) {
taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
taskDataPar->inputs_count.emplace_back(in_size.size());
taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
taskDataPar->inputs_count.emplace_back(A_flat.size());
taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
taskDataPar->inputs_count.emplace_back(B_flat.size());
taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
taskDataPar->outputs_count.emplace_back(out.size());
}

auto strassenTaskParallel = std::make_shared<nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmParallel>(taskDataPar);
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
ASSERT_EQ(matrix_size * matrix_size, out.size());
}
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_task_run) {
boost::mpi::communicator world;

const size_t matrix_size = 512;
auto [A_flat, B_flat] = nasedkin_e_strassen_algorithm_mpi::generate_random_matrices(matrix_size);

std::vector<size_t> in_size(1, matrix_size);
std::vector<double> out(matrix_size * matrix_size, 0.0);

std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
if (world.rank() == 0) {
taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
taskDataPar->inputs_count.emplace_back(in_size.size());
taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
taskDataPar->inputs_count.emplace_back(A_flat.size());
taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
taskDataPar->inputs_count.emplace_back(B_flat.size());
taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
taskDataPar->outputs_count.emplace_back(out.size());
}

auto strassenTaskParallel = std::make_shared<nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmParallel>(taskDataPar);
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
ASSERT_EQ(matrix_size * matrix_size, out.size());
}
}