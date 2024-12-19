#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <boost/mpi/collectives.hpp>

namespace nasedkin_e_strassen_algorithm {

void StrassenAlgorithmMPI::generate_random_matrix(int size, std::vector<double>& matrix) {
  matrix.resize(size * size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 10);
  for (int i = 0; i < size * size; ++i) {
    matrix[i] = dis(gen);
  }
}

void StrassenAlgorithmMPI::set_matrices(const std::vector<double>& A, const std::vector<double>& B) {
  matrixA = A;
  matrixB = B;
}

bool StrassenAlgorithmMPI::pre_processing() {
  if (world.rank() == 0) {
    boost::mpi::broadcast(world, matrixA, 0);
    boost::mpi::broadcast(world, matrixB, 0);
  } else {
    boost::mpi::broadcast(world, matrixA, 0);
    boost::mpi::broadcast(world, matrixB, 0);
  }
  return !matrixA.empty() && !matrixB.empty() && std::sqrt(matrixA.size()) == std::sqrt(matrixB.size());
}

bool StrassenAlgorithmMPI::validation() {
  int size = std::sqrt(matrixA.size());
  return size * size == matrixA.size();
}

bool StrassenAlgorithmMPI::run() {
  int rank = world.rank();
  int size = world.size();
  int n = std::sqrt(matrixA.size());

  int rows_per_proc = n / size;
  int start_row = rank * rows_per_proc;
  int end_row = (rank == size - 1) ? n : start_row + rows_per_proc;

  std::vector<double> local_A(rows_per_proc * n);
  std::vector<double> local_B(rows_per_proc * n);
  for (int i = 0; i < rows_per_proc; ++i) {
    for (int j = 0; j < n; ++j) {
      local_A[i * n + j] = matrixA[(start_row + i) * n + j];
      local_B[i * n + j] = matrixB[(start_row + i) * n + j];
    }
  }

  std::vector<double> local_result = strassen_multiply(local_A, local_B, n);

  boost::mpi::gather(world, local_result, resultMatrix, 0);
  return true;
}

bool StrassenAlgorithmMPI::post_processing() {
  if (world.rank() == 0) {
    int n = std::sqrt(resultMatrix.size());
    std::cout << "Resulting Matrix:\n";
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        std::cout << resultMatrix[i * n + j] << " ";
      }
      std::cout << "\n";
    }
  }
  return true;
}

std::vector<double> StrassenAlgorithmMPI::add(const std::vector<double>& A, const std::vector<double>& B, int size) {
  std::vector<double> C(size * size);
  for (int i = 0; i < size * size; ++i) {
    C[i] = A[i] + B[i];
  }
  return C;
}

std::vector<double> StrassenAlgorithmMPI::subtract(const std::vector<double>& A, const std::vector<double>& B, int size) {
  std::vector<double> C(size * size);
  for (int i = 0; i < size * size; ++i) {
    C[i] = A[i] - B[i];
  }
  return C;
}

void StrassenAlgorithmMPI::split_matrix(const std::vector<double>& matrix, int size,
                                        std::vector<double>& A11, std::vector<double>& A12,
                                        std::vector<double>& A21, std::vector<double>& A22) {
  int n = size / 2;
  A11.resize(n * n);
  A12.resize(n * n);
  A21.resize(n * n);
  A22.resize(n * n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A11[i * n + j] = matrix[i * size + j];
      A12[i * n + j] = matrix[i * size + j + n];
      A21[i * n + j] = matrix[(i + n) * size + j];
      A22[i * n + j] = matrix[(i + n) * size + j + n];
    }
  }
}

std::vector<double> StrassenAlgorithmMPI::merge_matrices(const std::vector<double>& C11, const std::vector<double>& C12,
                                                         const std::vector<double>& C21, const std::vector<double>& C22, int size) {
  int n = size / 2;
  std::vector<double> C(size * size);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i * size + j] = C11[i * n + j];
      C[i * size + j + n] = C12[i * n + j];
      C[(i + n) * size + j] = C21[i * n + j];
      C[(i + n) * size + j + n] = C22[i * n + j];
    }
  }
  return C;
}

std::vector<double> StrassenAlgorithmMPI::strassen_multiply(const std::vector<double>& A, const std::vector<double>& B, int size) {
  if (size == 1) {
    return {A[0] * B[0]};
  }

  std::vector<double> A11, A12, A21, A22;
  std::vector<double> B11, B12, B21, B22;
  split_matrix(A, size, A11, A12, A21, A22);
  split_matrix(B, size, B11, B12, B21, B22);

  std::vector<double> M1 = strassen_multiply(add(A11, A22, size / 2), add(B11, B22, size / 2), size / 2);
  std::vector<double> M2 = strassen_multiply(add(A21, A22, size / 2), B11, size / 2);
  std::vector<double> M3 = strassen_multiply(A11, subtract(B12, B22, size / 2), size / 2);
  std::vector<double> M4 = strassen_multiply(A22, subtract(B21, B11, size / 2), size / 2);
  std::vector<double> M5 = strassen_multiply(add(A11, A12, size / 2), B22, size / 2);
  std::vector<double> M6 = strassen_multiply(subtract(A21, A11, size / 2), add(B11, B12, size / 2), size / 2);
  std::vector<double> M7 = strassen_multiply(subtract(A12, A22, size / 2), add(B21, B22, size / 2), size / 2);

  std::vector<double> C11 = add(subtract(add(M1, M4, size / 2), M5, size / 2), M7, size / 2);
  std::vector<double> C12 = add(M3, M5, size / 2);
  std::vector<double> C21 = add(M2, M4, size / 2);
  std::vector<double> C22 = add(subtract(add(M1, M3, size / 2), M2, size / 2), M6, size / 2);

  return merge_matrices(C11, C12, C21, C22, size);
}

}  // namespace nasedkin_e_strassen_algorithm