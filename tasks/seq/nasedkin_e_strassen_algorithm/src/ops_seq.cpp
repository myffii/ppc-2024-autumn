#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"
#include <iostream>
#include <cmath>

namespace nasedkin_e_strassen_algorithm {

bool StrassenAlgorithmSeq::pre_processing() {
  if (!validation()) {
    return false;
  }
  result.resize(n, std::vector<double>(n, 0.0));
  return true;
}

bool StrassenAlgorithmSeq::validation() {
  if (taskData->inputs_count.empty()) {
    return false;
  }
  n = taskData->inputs_count[0];
  return n > 0 && (n & (n - 1)) == 0;
}

bool StrassenAlgorithmSeq::run() {
  strassen_multiply(A, B, result, n);
  return true;
}

bool StrassenAlgorithmSeq::post_processing() { return true; }

void StrassenAlgorithmSeq::set_matrices(const std::vector<std::vector<double>>& matrixA, const std::vector<std::vector<double>>& matrixB) {
  A = matrixA;
  B = matrixB;
}

void StrassenAlgorithmSeq::generate_random_matrix(int size, std::vector<std::vector<double>>& matrix) {
  matrix.resize(size, std::vector<double>(size));
  for (auto& row : matrix) {
    for (auto& elem : row) {
      elem = static_cast<double>(rand() % 100);
    }
  }
}

void StrassenAlgorithmSeq::strassen_multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size) {
  if (size <= 64) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        C[i][j] = 0;
        for (int k = 0; k < size; ++k) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return;
  }

  int new_size = size / 2;
  std::vector<std::vector<double>> A11(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> A12(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> A21(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> A22(new_size, std::vector<double>(new_size));

  std::vector<std::vector<double>> B11(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> B12(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> B21(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> B22(new_size, std::vector<double>(new_size));

  split_matrix(A, A11, A12, A21, A22, new_size);
  split_matrix(B, B11, B12, B21, B22, new_size);

  std::vector<std::vector<double>> M1(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> tempA(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> tempB(new_size, std::vector<double>(new_size));

  add_matrices(A11, A22, tempA, new_size);
  add_matrices(B11, B22, tempB, new_size);
  strassen_multiply(tempA, tempB, M1, new_size);

  join_matrices(A11, A12, A21, A22, C, new_size);
}

}  // namespace nasedkin_e_strassen_algorithm
