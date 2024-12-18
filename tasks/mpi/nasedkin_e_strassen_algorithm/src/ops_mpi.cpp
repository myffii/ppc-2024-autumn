#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <cmath>
#include <iostream>
#include <boost/mpi/collectives.hpp>

namespace nasedkin_e_strassen_algorithm {

bool StrassenAlgorithmMPI::pre_processing() {
  if (!validation()) return false;

  result.resize(n, std::vector<double>(n, 0.0));
  distribute_matrix(A, A);
  distribute_matrix(B, B);
  return true;
}

bool StrassenAlgorithmMPI::validation() {
  if (taskData->inputs_count.empty()) return false;

  n = taskData->inputs_count[0];
  if (n <= 0 || (n & (n - 1)) != 0) return false;

  A.resize(n, std::vector<double>(n, 0.0));
  B.resize(n, std::vector<double>(n, 0.0));

  return true;
}

bool StrassenAlgorithmMPI::run() {
  std::vector<std::vector<double>> local_result(n / world.size(), std::vector<double>(n, 0.0));
  strassen_multiply(A, B, local_result, n / world.size());
  gather_result(local_result, result);
  return true;
}

bool StrassenAlgorithmMPI::post_processing() {
  return true;
}
void StrassenAlgorithmMPI::strassen_multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size) {
  if (size <= 2) {
    for (int i = 0; i < size; ++i)
      for (int j = 0; j < size; ++j)
        for (int k = 0; k < size; ++k)
          C[i][j] += A[i][k] * B[k][j];
    return;
  }

  int new_size = size / 2;
  std::vector<std::vector<double>> A11(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> A12(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> A21(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> A22(new_size, std::vector<double>(new_size));

  split_matrix(A, A11, A12, A21, A22, new_size);

  std::vector<std::vector<double>> B11(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> B12(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> B21(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> B22(new_size, std::vector<double>(new_size));

  split_matrix(B, B11, B12, B21, B22, new_size);

  std::vector<std::vector<double>> C11(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> C12(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> C21(new_size, std::vector<double>(new_size));
  std::vector<std::vector<double>> C22(new_size, std::vector<double>(new_size));

  join_matrices(C11, C12, C21, C22, C, size);
}

    void StrassenAlgorithmMPI::add_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
    }

    void StrassenAlgorithmMPI::subtract_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
    }

    void StrassenAlgorithmMPI::split_matrix(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& A11, std::vector<std::vector<double>>& A12, std::vector<std::vector<double>>& A21, std::vector<std::vector<double>>& A22, int size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                A11[i][j] = A[i][j];
                A12[i][j] = A[i][j + size];
                A21[i][j] = A[i + size][j];
                A22[i][j] = A[i + size][j + size];
            }
        }
    }

    void StrassenAlgorithmMPI::join_matrices(const std::vector<std::vector<double>>& C11, const std::vector<std::vector<double>>& C12, const std::vector<std::vector<double>>& C21, const std::vector<std::vector<double>>& C22, std::vector<std::vector<double>>& C, int size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                C[i][j] = C11[i][j];
                C[i][j + size] = C12[i][j];
                C[i + size][j] = C21[i][j];
                C[i + size][j + size] = C22[i][j];
            }
        }
    }

    void StrassenAlgorithmMPI::set_matrices(const std::vector<std::vector<double>>& matrixA, const std::vector<std::vector<double>>& matrixB) {
        A = matrixA;
        B = matrixB;
        n = static_cast<int>(matrixA.size());
    }

    void StrassenAlgorithmMPI::generate_random_matrix(int size, std::vector<std::vector<double>>& matrix) {
        matrix.resize(size, std::vector<double>(size, 0.0));

        std::srand(static_cast<unsigned>(std::time(nullptr)));

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                matrix[i][j] = static_cast<double>(std::rand() % 10 + 1);
            }
        }
    }

    void StrassenAlgorithmMPI::distribute_matrix(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& distributed_matrix) {
      int total_size = n * n;
      std::vector<double> flat_matrix(total_size);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          flat_matrix[i * n + j] = matrix[i][j];
      boost::mpi::scatter(world, flat_matrix, distributed_matrix, 0);
    }

    void StrassenAlgorithmMPI::gather_result(const std::vector<std::vector<double>>& local_result, std::vector<std::vector<double>>& gathered_result) {
      std::vector<double> flat_local_result(local_result.size() * local_result[0].size());
      for (int i = 0; i < local_result.size(); ++i)
        for (int j = 0; j < local_result[0].size(); ++j)
          flat_local_result[i * local_result[0].size() + j] = local_result[i][j];

      boost::mpi::gather(world, flat_local_result, gathered_result, 0);
    }

    void StrassenAlgorithmMPI::flatten_matrix(const std::vector<std::vector<double>>& matrix, std::vector<double>& flat_matrix) {
        flat_matrix.clear();
        for (const auto& row : matrix) {
            flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
        }
    }

    void StrassenAlgorithmMPI::unflatten_matrix(const std::vector<double>& flat_matrix, std::vector<std::vector<double>>& matrix, int rows, int cols) {
        matrix.clear();
        matrix.resize(rows, std::vector<double>(cols, 0.0));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix[i][j] = flat_matrix[i * cols + j];
            }
        }
    }

}  // namespace nasedkin_e_strassen_algorithm