#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

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

// Вычисление разности между двумя матрицами
std::vector<std::vector<double>> matrix_difference(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
  size_t n = A.size();
  std::vector<std::vector<double>> diff(n, std::vector<double>(n, 0.0));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      diff[i][j] = A[i][j] - B[i][j];
    }
  }

  return diff;
}

// Вычисление нормы разности матриц
double matrix_norm(const std::vector<std::vector<double>>& matrix) {
  double norm = 0.0;
  for (const auto& row : matrix) {
    for (double val : row) {
      norm += val * val;
    }
  }
  return std::sqrt(norm);
}

}  // namespace nasedkin_e_strassen_algorithm_mpi

// Тест для матрицы 2x2
/*TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_2x2) {
  boost::mpi::communicator world;

  auto [A, B] = nasedkin_e_strassen_algorithm_mpi::generate_random_matrix(2);

  std::vector<std::vector<double>> C_parallel(2, std::vector<double>(2, 0.0));
  std::vector<std::vector<double>> C_sequential(2, std::vector<double>(2, 0.0));

  size_t matrix_size_copy = 2;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataPar->inputs_count.emplace_back(1);

    std::vector<double> A_flat(4);
    std::vector<double> B_flat(4);

    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        A_flat[i * 2 + j] = A[i][j];
        B_flat[i * 2 + j] = B[i][j];
      }
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataPar->inputs_count.emplace_back(A_flat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataPar->inputs_count.emplace_back(B_flat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_parallel.data()));
    taskDataPar->outputs_count.emplace_back(C_parallel.size() * C_parallel.size());
  }

  nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmMPIParallel strassen_parallel(taskDataPar);
  ASSERT_TRUE(strassen_parallel.validation());
  strassen_parallel.pre_processing();
  strassen_parallel.run();
  strassen_parallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataSeq->inputs_count.emplace_back(1);

    std::vector<double> A_flat(4);
    std::vector<double> B_flat(4);

    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        A_flat[i * 2 + j] = A[i][j];
        B_flat[i * 2 + j] = B[i][j];
      }
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataSeq->inputs_count.emplace_back(A_flat.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataSeq->inputs_count.emplace_back(B_flat.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(C_sequential.size() * C_sequential.size());

    nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmMPISequential strassen_sequential(taskDataSeq);
    ASSERT_TRUE(strassen_sequential.validation());
    strassen_sequential.pre_processing();
    strassen_sequential.run();
    strassen_sequential.post_processing();
  }

  if (world.rank() == 0) {
    auto diff = nasedkin_e_strassen_algorithm_mpi::matrix_difference(C_parallel, C_sequential);
    double norm = nasedkin_e_strassen_algorithm_mpi::matrix_norm(diff);

    ASSERT_LT(norm, 1e-6) << "Parallel and sequential results differ by more than tolerance.";
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}*/

// Тест для матрицы 4x4
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_4x4) {
  boost::mpi::communicator world;

  auto [A, B] = nasedkin_e_strassen_algorithm_mpi::generate_random_matrix(4);

  std::vector<std::vector<double>> C_parallel(4, std::vector<double>(4, 0.0));
  std::vector<std::vector<double>> C_sequential(4, std::vector<double>(4, 0.0));

  size_t matrix_size_copy = 4;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataPar->inputs_count.emplace_back(1);

    std::vector<double> A_flat(16);
    std::vector<double> B_flat(16);

    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        A_flat[i * 4 + j] = A[i][j];
        B_flat[i * 4 + j] = B[i][j];
      }
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataPar->inputs_count.emplace_back(A_flat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataPar->inputs_count.emplace_back(B_flat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_parallel.data()));
    taskDataPar->outputs_count.emplace_back(C_parallel.size() * C_parallel.size());
  }

  nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmMPIParallel strassen_parallel(taskDataPar);
  ASSERT_TRUE(strassen_parallel.validation());
  strassen_parallel.pre_processing();
  strassen_parallel.run();
  strassen_parallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataSeq->inputs_count.emplace_back(1);

    std::vector<double> A_flat(16);
    std::vector<double> B_flat(16);

    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        A_flat[i * 4 + j] = A[i][j];
        B_flat[i * 4 + j] = B[i][j];
      }
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataSeq->inputs_count.emplace_back(A_flat.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataSeq->inputs_count.emplace_back(B_flat.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(C_sequential.size() * C_sequential.size());

    nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmMPISequential strassen_sequential(taskDataSeq);
    ASSERT_TRUE(strassen_sequential.validation());
    strassen_sequential.pre_processing();
    strassen_sequential.run();
    strassen_sequential.post_processing();
  }

  if (world.rank() == 0) {
    auto diff = nasedkin_e_strassen_algorithm_mpi::matrix_difference(C_parallel, C_sequential);
    double norm = nasedkin_e_strassen_algorithm_mpi::matrix_norm(diff);

    ASSERT_LT(norm, 1e-6) << "Parallel and sequential results differ by more than tolerance.";
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

// Тест для матрицы 8x8
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_8x8) {
  boost::mpi::communicator world;

  auto [A, B] = nasedkin_e_strassen_algorithm_mpi::generate_random_matrix(8);

  std::vector<std::vector<double>> C_parallel(8, std::vector<double>(8, 0.0));
  std::vector<std::vector<double>> C_sequential(8, std::vector<double>(8, 0.0));

  size_t matrix_size_copy = 8;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataPar->inputs_count.emplace_back(1);

    std::vector<double> A_flat(64);
    std::vector<double> B_flat(64);

    for (size_t i = 0; i < 8; ++i) {
      for (size_t j = 0; j < 8; ++j) {
        A_flat[i * 8 + j] = A[i][j];
        B_flat[i * 8 + j] = B[i][j];
      }
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataPar->inputs_count.emplace_back(A_flat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataPar->inputs_count.emplace_back(B_flat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_parallel.data()));
    taskDataPar->outputs_count.emplace_back(C_parallel.size() * C_parallel.size());
  }

  nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmMPIParallel strassen_parallel(taskDataPar);
  ASSERT_TRUE(strassen_parallel.validation());
  strassen_parallel.pre_processing();
  strassen_parallel.run();
  strassen_parallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataSeq->inputs_count.emplace_back(1);

    std::vector<double> A_flat(64);
    std::vector<double> B_flat(64);

    for (size_t i = 0; i < 8; ++i) {
      for (size_t j = 0; j < 8; ++j) {
        A_flat[i * 8 + j] = A[i][j];
        B_flat[i * 8 + j] = B[i][j];
      }
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataSeq->inputs_count.emplace_back(A_flat.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataSeq->inputs_count.emplace_back(B_flat.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(C_sequential.size() * C_sequential.size());

    nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmMPISequential strassen_sequential(taskDataSeq);
    ASSERT_TRUE(strassen_sequential.validation());
    strassen_sequential.pre_processing();
    strassen_sequential.run();
    strassen_sequential.post_processing();
  }

  if (world.rank() == 0) {
    auto diff = nasedkin_e_strassen_algorithm_mpi::matrix_difference(C_parallel, C_sequential);
    double norm = nasedkin_e_strassen_algorithm_mpi::matrix_norm(diff);

    ASSERT_LT(norm, 1e-6) << "Parallel and sequential results differ by more than tolerance.";
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

// Тест для матрицы 16x16
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_16x16) {
  boost::mpi::communicator world;

  auto [A, B] = nasedkin_e_strassen_algorithm_mpi::generate_random_matrix(16);

  std::vector<std::vector<double>> C_parallel(16, std::vector<double>(16, 0.0));
  std::vector<std::vector<double>> C_sequential(16, std::vector<double>(16, 0.0));

  size_t matrix_size_copy = 16;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataPar->inputs_count.emplace_back(1);

    std::vector<double> A_flat(256);
    std::vector<double> B_flat(256);

    for (size_t i = 0; i < 16; ++i) {
      for (size_t j = 0; j < 16; ++j) {
        A_flat[i * 16 + j] = A[i][j];
        B_flat[i * 16 + j] = B[i][j];
      }
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataPar->inputs_count.emplace_back(A_flat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataPar->inputs_count.emplace_back(B_flat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_parallel.data()));
    taskDataPar->outputs_count.emplace_back(C_parallel.size() * C_parallel.size());
  }

  nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmMPIParallel strassen_parallel(taskDataPar);
  ASSERT_TRUE(strassen_parallel.validation());
  strassen_parallel.pre_processing();
  strassen_parallel.run();
  strassen_parallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataSeq->inputs_count.emplace_back(1);

    std::vector<double> A_flat(256);
    std::vector<double> B_flat(256);

    for (size_t i = 0; i < 16; ++i) {
      for (size_t j = 0; j < 16; ++j) {
        A_flat[i * 16 + j] = A[i][j];
        B_flat[i * 16 + j] = B[i][j];
      }
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataSeq->inputs_count.emplace_back(A_flat.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataSeq->inputs_count.emplace_back(B_flat.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(C_sequential.size() * C_sequential.size());

    nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmMPISequential strassen_sequential(taskDataSeq);
    ASSERT_TRUE(strassen_sequential.validation());
    strassen_sequential.pre_processing();
    strassen_sequential.run();
    strassen_sequential.post_processing();
  }

  if (world.rank() == 0) {
    auto diff = nasedkin_e_strassen_algorithm_mpi::matrix_difference(C_parallel, C_sequential);
    double norm = nasedkin_e_strassen_algorithm_mpi::matrix_norm(diff);

    ASSERT_LT(norm, 1e-6) << "Parallel and sequential results differ by more than tolerance.";
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

// Тест для матрицы 32x32
TEST(nasedkin_e_strassen_algorithm_mpi, test_matrix_32x32) {
  boost::mpi::communicator world;

  auto [A, B] = nasedkin_e_strassen_algorithm_mpi::generate_random_matrix(32);

  std::vector<std::vector<double>> C_parallel(32, std::vector<double>(32, 0.0));
  std::vector<std::vector<double>> C_sequential(32, std::vector<double>(32, 0.0));

  size_t matrix_size_copy = 32;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataPar->inputs_count.emplace_back(1);

    std::vector<double> A_flat(1024);
    std::vector<double> B_flat(1024);

    for (size_t i = 0; i < 32; ++i) {
      for (size_t j = 0; j < 32; ++j) {
        A_flat[i * 32 + j] = A[i][j];
        B_flat[i * 32 + j] = B[i][j];
      }
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataPar->inputs_count.emplace_back(A_flat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataPar->inputs_count.emplace_back(B_flat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_parallel.data()));
    taskDataPar->outputs_count.emplace_back(C_parallel.size() * C_parallel.size());
  }

  nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmMPIParallel strassen_parallel(taskDataPar);
  ASSERT_TRUE(strassen_parallel.validation());
  strassen_parallel.pre_processing();
  strassen_parallel.run();
  strassen_parallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataSeq->inputs_count.emplace_back(1);

    std::vector<double> A_flat(1024);
    std::vector<double> B_flat(1024);

    for (size_t i = 0; i < 32; ++i) {
      for (size_t j = 0; j < 32; ++j) {
        A_flat[i * 32 + j] = A[i][j];
        B_flat[i * 32 + j] = B[i][j];
      }
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataSeq->inputs_count.emplace_back(A_flat.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_flat.data()));
    taskDataSeq->inputs_count.emplace_back(B_flat.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(C_sequential.size() * C_sequential.size());

    nasedkin_e_strassen_algorithm_mpi::StrassenAlgorithmMPISequential strassen_sequential(taskDataSeq);
    ASSERT_TRUE(strassen_sequential.validation());
    strassen_sequential.pre_processing();
    strassen_sequential.run();
    strassen_sequential.post_processing();
  }

  if (world.rank() == 0) {
    auto diff = nasedkin_e_strassen_algorithm_mpi::matrix_difference(C_parallel, C_sequential);
    double norm = nasedkin_e_strassen_algorithm_mpi::matrix_norm(diff);

    ASSERT_LT(norm, 1e-6) << "Parallel and sequential results differ by more than tolerance.";
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}