#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>

namespace nasedkin_e_strassen_algorithm {

void StrassenAlgorithmMPI::generate_random_matrix(int size, std::vector<std::vector<double>>& matrix) {
  matrix.resize(size, std::vector<double>(size));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 10);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix[i][j] = dis(gen);
    }
  }
}

void StrassenAlgorithmMPI::set_matrices(const std::vector<std::vector<double>>& A,
                                        const std::vector<std::vector<double>>& B) {
  matrixA = A;
  matrixB = B;
}

bool StrassenAlgorithmMPI::pre_processing() {
  int n = matrixA.size();
  int rows_per_process = n / world.size();
  int start_row = world.rank() * rows_per_process;
  int end_row = (world.rank() == world.size() - 1) ? n : start_row + rows_per_process;

  // Разделяем матрицы по строкам между процессами
  std::vector<int> sendcounts(world.size(), rows_per_process * n);
  std::vector<int> displs(world.size(), 0);
  for (int i = 0; i < world.size(); ++i) {
    displs[i] = i * rows_per_process * n;
  }
  sendcounts[world.size() - 1] = (n - (world.size() - 1) * rows_per_process) * n;

  std::vector<double> local_matrixA(sendcounts[world.rank()]);
  std::vector<double> local_matrixB(sendcounts[world.rank()]);

  // Используем scatterv с правильным количеством аргументов
  boost::mpi::scatterv(world, matrixA.data()->data(), sendcounts, displs, local_matrixA.data(), 0);
  boost::mpi::scatterv(world, matrixB.data()->data(), sendcounts, displs, local_matrixB.data(), 0);

  // Преобразуем локальные данные обратно в матрицы
  matrixA.clear();
  matrixB.clear();
  for (int i = 0; i < end_row - start_row; ++i) {
    matrixA.emplace_back(local_matrixA.begin() + i * n, local_matrixA.begin() + (i + 1) * n);
    matrixB.emplace_back(local_matrixB.begin() + i * n, local_matrixB.begin() + (i + 1) * n);
  }

  return true;
}

bool StrassenAlgorithmMPI::validation() {
  return !matrixA.empty() && matrixA.size() == matrixA[0].size();
}

bool StrassenAlgorithmMPI::run() {
  int n = matrixA.size();
  resultMatrix = strassen_multiply(matrixA, matrixB);

  // Собираем результаты от всех процессов
  std::vector<double> local_result(resultMatrix.size() * n);
  for (int i = 0; i < static_cast<int>(resultMatrix.size()); ++i) {  // Исправлено сравнение
    std::copy(resultMatrix[i].begin(), resultMatrix[i].end(), local_result.begin() + i * n);
  }

  std::vector<int> recvcounts(world.size(), resultMatrix.size() * n);
  std::vector<int> displs(world.size(), 0);
  for (int i = 0; i < world.size(); ++i) {
    displs[i] = i * resultMatrix.size() * n;
  }

  std::vector<double> global_result(n * n);
  boost::mpi::gatherv(world, local_result.data(), local_result.size(), global_result.data(), recvcounts, displs, 0);

  if (world.rank() == 0) {
    resultMatrix.clear();
    for (int i = 0; i < n; ++i) {
      resultMatrix.emplace_back(global_result.begin() + i * n, global_result.begin() + (i + 1) * n);
    }
  }

  return true;
}

bool StrassenAlgorithmMPI::post_processing() {
  if (world.rank() == 0) {
    std::cout << "Resulting Matrix:\n";
    for (const auto& row : resultMatrix) {
      for (auto elem : row) {
        std::cout << elem << " ";
      }
      std::cout << "\n";
    }
  }
  return true;
}

std::vector<std::vector<double>> StrassenAlgorithmMPI::add(const std::vector<std::vector<double>>& A,
                                                           const std::vector<std::vector<double>>& B) {
  int n = A.size();
  std::vector<std::vector<double>> C(n, std::vector<double>(n));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      C[i][j] = A[i][j] + B[i][j];
  return C;
}

std::vector<std::vector<double>> StrassenAlgorithmMPI::subtract(const std::vector<std::vector<double>>& A,
                                                                const std::vector<std::vector<double>>& B) {
  int n = A.size();
  std::vector<std::vector<double>> C(n, std::vector<double>(n));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      C[i][j] = A[i][j] - B[i][j];
  return C;
}

void StrassenAlgorithmMPI::split_matrix(const std::vector<std::vector<double>>& matrix,
                                        std::vector<std::vector<double>>& A11,
                                        std::vector<std::vector<double>>& A12,
                                        std::vector<std::vector<double>>& A21,
                                        std::vector<std::vector<double>>& A22) {
  int n = matrix.size() / 2;
  A11.resize(n, std::vector<double>(n));
  A12.resize(n, std::vector<double>(n));
  A21.resize(n, std::vector<double>(n));
  A22.resize(n, std::vector<double>(n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A11[i][j] = matrix[i][j];
      A12[i][j] = matrix[i][j + n];
      A21[i][j] = matrix[i + n][j];
      A22[i][j] = matrix[i + n][j + n];
    }
  }
}

std::vector<std::vector<double>> StrassenAlgorithmMPI::merge_matrices(const std::vector<std::vector<double>>& C11,
                                                                      const std::vector<std::vector<double>>& C12,
                                                                      const std::vector<std::vector<double>>& C21,
                                                                      const std::vector<std::vector<double>>& C22) {
  int n = C11.size();
  std::vector<std::vector<double>> C(2 * n, std::vector<double>(2 * n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i][j] = C11[i][j];
      C[i][j + n] = C12[i][j];
      C[i + n][j] = C21[i][j];
      C[i + n][j + n] = C22[i][j];
    }
  }
  return C;
}

std::vector<std::vector<double>> StrassenAlgorithmMPI::strassen_multiply(const std::vector<std::vector<double>>& A,
                                                                         const std::vector<std::vector<double>>& B) {
  int n = A.size();
  if (n <= 2) {
    return add(A, B);
  }

  std::vector<std::vector<double>> A11;
  std::vector<std::vector<double>> A12;
  std::vector<std::vector<double>> A21;
  std::vector<std::vector<double>> A22;
  std::vector<std::vector<double>> B11;
  std::vector<std::vector<double>> B12;
  std::vector<std::vector<double>> B21;
  std::vector<std::vector<double>> B22;
  split_matrix(A, A11, A12, A21, A22);
  split_matrix(B, B11, B12, B21, B22);

  auto P1 = strassen_multiply(add(A11, A22), add(B11, B22));
  auto P2 = strassen_multiply(add(A21, A22), B11);
  auto P3 = strassen_multiply(A11, subtract(B12, B22));
  auto P4 = strassen_multiply(A22, subtract(B21, B11));

  auto C11 = add(P1, P4);
  auto C12 = add(P3, P2);
  auto C21 = add(P2, P4);
  auto C22 = subtract(P1, P3);

  return merge_matrices(C11, C12, C21, C22);
}

}  // namespace nasedkin_e_strassen_algorithm