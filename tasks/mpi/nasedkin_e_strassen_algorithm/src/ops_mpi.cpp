#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <cmath>
#include <iostream>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

namespace nasedkin_e_strassen_algorithm {

void generate_random_matrix(int size, std::vector<std::vector<double>>& matrix) {
  matrix.resize(size, std::vector<double>(size, 0.0));
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix[i][j] = static_cast<double>(std::rand() % 100) / 10.0;
    }
  }
}

void split_matrix(const std::vector<std::vector<double>>& matrix,
                  std::vector<std::vector<double>>& A11, std::vector<std::vector<double>>& A12,
                  std::vector<std::vector<double>>& A21, std::vector<std::vector<double>>& A22) {
  int n = matrix.size();
  int half = n / 2;
  A11.resize(half, std::vector<double>(half));
  A12.resize(half, std::vector<double>(half));
  A21.resize(half, std::vector<double>(half));
  A22.resize(half, std::vector<double>(half));

  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      A11[i][j] = matrix[i][j];
      A12[i][j] = matrix[i][j + half];
      A21[i][j] = matrix[i + half][j];
      A22[i][j] = matrix[i + half][j + half];
    }
  }
}

void merge_matrices(const std::vector<std::vector<double>>& C11, const std::vector<std::vector<double>>& C12,
                    const std::vector<std::vector<double>>& C21, const std::vector<std::vector<double>>& C22,
                    std::vector<std::vector<double>>& C) {
  int half = C11.size();
  int n = half * 2;
  C.resize(n, std::vector<double>(n));

  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      C[i][j] = C11[i][j];
      C[i][j + half] = C12[i][j];
      C[i + half][j] = C21[i][j];
      C[i + half][j + half] = C22[i][j];
    }
  }
}

std::vector<std::vector<double>> matrix_add(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
  int n = A.size();
  std::vector<std::vector<double>> result(n, std::vector<double>(n, 0.0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result[i][j] = A[i][j] + B[i][j];
    }
  }
  return result;
}

std::vector<std::vector<double>> matrix_subtract(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
  int n = A.size();
  std::vector<std::vector<double>> result(n, std::vector<double>(n, 0.0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result[i][j] = A[i][j] - B[i][j];
    }
  }
  return result;
}

std::vector<std::vector<double>> strassen_multiply(const std::vector<std::vector<double>>& A,
                                                   const std::vector<std::vector<double>>& B) {
  int n = A.size();
  if (n <= 2) {

    std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return C;
  }

  std::vector<std::vector<double>> A11, A12, A21, A22;
  std::vector<std::vector<double>> B11, B12, B21, B22;
  split_matrix(A, A11, A12, A21, A22);
  split_matrix(B, B11, B12, B21, B22);

  std::vector<std::vector<double>> P1 = strassen_multiply(matrix_add(A11, A22), matrix_add(B11, B22));
  std::vector<std::vector<double>> P2 = strassen_multiply(matrix_add(A21, A22), B11);
  std::vector<std::vector<double>> P3 = strassen_multiply(A11, matrix_subtract(B12, B22));
  std::vector<std::vector<double>> P4 = strassen_multiply(A22, matrix_subtract(B21, B11));
  std::vector<std::vector<double>> P5 = strassen_multiply(matrix_add(A11, A12), B22);
  std::vector<std::vector<double>> P6 = strassen_multiply(matrix_subtract(A21, A11), matrix_add(B11, B12));
  std::vector<std::vector<double>> P7 = strassen_multiply(matrix_subtract(A12, A22), matrix_add(B21, B22));

  std::vector<std::vector<double>> C11 = matrix_add(matrix_subtract(matrix_add(P1, P4), P5), P7);
  std::vector<std::vector<double>> C12 = matrix_add(P3, P5);
  std::vector<std::vector<double>> C21 = matrix_add(P2, P4);
  std::vector<std::vector<double>> C22 = matrix_add(matrix_subtract(matrix_add(P1, P3), P2), P6);

  std::vector<std::vector<double>> C;
  merge_matrices(C11, C12, C21, C22, C);
  return C;
}

bool StrassenAlgorithmMPI::pre_processing() {
  int size = taskData->inputs_count[0];
  generate_random_matrix(size, matrixA);
  generate_random_matrix(size, matrixB);

  int num_procs = world.size();
  int rows_per_proc = size / num_procs;

  local_A.resize(rows_per_proc, std::vector<double>(size));
  local_B.resize(rows_per_proc, std::vector<double>(size));

  std::vector<double> flat_A(size * size);
  std::vector<double> flat_B(size * size);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      flat_A[i * size + j] = matrixA[i][j];
      flat_B[i * size + j] = matrixB[i][j];
    }
  }

  boost::mpi::scatter(world, flat_A, local_A[0].data(), rows_per_proc * size, 0);
  boost::mpi::scatter(world, flat_B, local_B[0].data(), rows_per_proc * size, 0);

  return true;
}

bool StrassenAlgorithmMPI::validation() {

  int size = taskData->inputs_count[0];
  return size > 0 && (size & (size - 1)) == 0;
}

bool StrassenAlgorithmMPI::run() {

  int size = taskData->inputs_count[0];
  int num_procs = world.size();
  int rows_per_proc = size / num_procs;

  std::vector<std::vector<double>> local_C = strassen_multiply(local_A, local_B);

  std::vector<double> flat_C(size * size);
  boost::mpi::gather(world, local_C[0].data(), rows_per_proc * size, flat_C.data(), 0);

  if (world.rank() == 0) {
    matrixC.resize(size, std::vector<double>(size));
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrixC[i][j] = flat_C[i * size + j];
      }
    }
  }

  return true;
}

bool StrassenAlgorithmMPI::post_processing() {
  return true;
}

}  // namespace nasedkin_e_strassen_algorithm