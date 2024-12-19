#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <random>
#include <vector>

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

// Функция для выполнения теста с заданным размером матрицы
void run_strassen_test_for_matrix_size(size_t matrix_size) {
  boost::mpi::communicator world;

  auto A = generate_random_matrix(matrix_size);
  auto B = generate_random_matrix(matrix_size);

  std::vector<std::vector<double>> parallel_result(matrix_size, std::vector<double>(matrix_size, 0.0));
  std::vector<std::vector<double>> sequential_result(matrix_size, std::vector<double>(matrix_size, 0.0));

  // Параллельная реализация
  StrassenMPITaskParallel parallel_task(A, B);
  auto parallel_C = parallel_task.run();

  // Последовательная реализация
  StrassenMPITaskSequential sequential_task(A, B);
  auto sequential_C = sequential_task.run();

  // Сравнение результатов
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      EXPECT_NEAR(parallel_C[i][j], sequential_C[i][j], 1e-6);
    }
  }
}

// Тесты для матриц разных размеров
TEST(nasedkin_e_strassen_algorithm, test_matrix_2) { run_strassen_test_for_matrix_size(2); }
TEST(nasedkin_e_strassen_algorithm, test_matrix_4) { run_strassen_test_for_matrix_size(4); }
TEST(nasedkin_e_strassen_algorithm, test_matrix_8) { run_strassen_test_for_matrix_size(8); }
TEST(nasedkin_e_strassen_algorithm, test_matrix_16) { run_strassen_test_for_matrix_size(16); }
TEST(nasedkin_e_strassen_algorithm, test_matrix_32) { run_strassen_test_for_matrix_size(32); }
TEST(nasedkin_e_strassen_algorithm, test_matrix_64) { run_strassen_test_for_matrix_size(64); }
TEST(nasedkin_e_strassen_algorithm, test_matrix_128) { run_strassen_test_for_matrix_size(128); }

// Тест для проверки валидации входных данных
TEST(nasedkin_e_strassen_algorithm, invalid_input_size) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    // Некорректный размер матрицы (не является степенью двойки)
    size_t invalid_size = 3;
    auto A = generate_random_matrix(invalid_size);
    auto B = generate_random_matrix(invalid_size);

    StrassenMPITaskParallel parallel_task(A, B);
    EXPECT_FALSE(parallel_task.validation());
  } else {
    EXPECT_TRUE(true);
  }
}

// Тест для проверки валидации входных данных (несоответствие размеров матриц)
TEST(nasedkin_e_strassen_algorithm, invalid_matrix_size_mismatch) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    // Матрицы разных размеров
    auto A = generate_random_matrix(4);
    auto B = generate_random_matrix(8);

    StrassenMPITaskParallel parallel_task(A, B);
    EXPECT_FALSE(parallel_task.validation());
  } else {
    EXPECT_TRUE(true);
  }
}

}  // namespace nasedkin_e_strassen_algorithm