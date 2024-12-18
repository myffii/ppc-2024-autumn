#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <boost/mpi/collectives.hpp>

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

void StrassenAlgorithmMPI::distribute_matrix_rows(const std::vector<std::vector<double>>& matrix,
                                                  std::vector<double>& local_flat_matrix) {
  int rows = matrix.size();
  int cols = matrix[0].size();
  int local_rows = rows / world.size();

  std::vector<double> flat_matrix;
  for (const auto& row : matrix) {
    flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
  }

  if (world.rank() == 0) {
    boost::mpi::scatter(world, flat_matrix.data(), local_rows * cols, local_flat_matrix.data(), 0);
  } else {
    local_flat_matrix.resize(local_rows * cols);
    boost::mpi::scatter(world, flat_matrix.data(), local_rows * cols, local_flat_matrix.data(), 0);
  }
}

void StrassenAlgorithmMPI::gather_result_rows(const std::vector<double>& local_flat_result,
                                              std::vector<std::vector<double>>& global_result) {
  int rows = global_result.size();
  int cols = global_result[0].size();
  int local_rows = rows / world.size();

  std::vector<double> flat_result;
  if (world.rank() == 0) {
    flat_result.resize(rows * cols);
  }
  boost::mpi::gather(world, local_flat_result.data(), local_rows * cols, flat_result.data(), 0);

  if (world.rank() == 0) {
    global_result.clear();
    for (int i = 0; i < rows; ++i) {
      global_result.emplace_back(flat_result.begin() + i * cols, flat_result.begin() + (i + 1) * cols);
    }
  }
}

bool StrassenAlgorithmMPI::pre_processing() {
  if (world.rank() == 0) {
    if (matrixA.empty() || matrixB.empty() || matrixA.size() != matrixB.size()) return false;
  }

  std::vector<double> local_flat_matrixA;
  std::vector<double> local_flat_matrixB;
  distribute_matrix_rows(matrixA, local_flat_matrixA);
  distribute_matrix_rows(matrixB, local_flat_matrixB);

  return true;
}

bool StrassenAlgorithmMPI::validation() {
  return !matrixA.empty() && matrixA.size() == matrixA[0].size();
}

bool StrassenAlgorithmMPI::run() {

  std::vector<double> local_flat_matrixA;
  std::vector<double> local_flat_matrixB;
  std::vector<double> local_flat_result;

  distribute_matrix_rows(matrixA, local_flat_matrixA);
  distribute_matrix_rows(matrixB, local_flat_matrixB);

  local_flat_result = strassen_multiply(local_flat_matrixA, local_flat_matrixB);

  if (world.rank() == 0) {
    resultMatrix.resize(matrixA.size(), std::vector<double>(matrixA.size()));
  }
  gather_result_rows(local_flat_result, resultMatrix);

  return true;
}

bool StrassenAlgorithmMPI::post_processing() { return true; }

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
