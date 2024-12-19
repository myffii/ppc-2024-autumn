#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <mpi.h>
#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

namespace nasedkin_e_strassen_algorithm_mpi {

void StrassenAlgorithmSequential::strassen(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
  if (n <= 64) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        C[i * n + j] = 0;
        for (size_t k = 0; k < n; ++k) {
          C[i * n + j] += A[i * n + k] * B[k * n + j];
        }
      }
    }
    return;
  }

  size_t new_n = n / 2;
  std::vector<double> A11(new_n * new_n);
  std::vector<double> A12(new_n * new_n);
  std::vector<double> A21(new_n * new_n);
  std::vector<double> A22(new_n * new_n);
  std::vector<double> B11(new_n * new_n);
  std::vector<double> B12(new_n * new_n);
  std::vector<double> B21(new_n * new_n);
  std::vector<double> B22(new_n * new_n);
  std::vector<double> C11(new_n * new_n);
  std::vector<double> C12(new_n * new_n);
  std::vector<double> C21(new_n * new_n);
  std::vector<double> C22(new_n * new_n);
  std::vector<double> M1(new_n * new_n);
  std::vector<double> M2(new_n * new_n);
  std::vector<double> M3(new_n * new_n);
  std::vector<double> M4(new_n * new_n);
  std::vector<double> M5(new_n * new_n);
  std::vector<double> M6(new_n * new_n);
  std::vector<double> M7(new_n * new_n);
  std::vector<double> temp1(new_n * new_n);
  std::vector<double> temp2(new_n * new_n);

  for (size_t i = 0; i < new_n; ++i) {
    for (size_t j = 0; j < new_n; ++j) {
      A11[i * new_n + j] = A[i * n + j];
      A12[i * new_n + j] = A[i * n + j + new_n];
      A21[i * new_n + j] = A[(i + new_n) * n + j];
      A22[i * new_n + j] = A[(i + new_n) * n + j + new_n];

      B11[i * new_n + j] = B[i * n + j];
      B12[i * new_n + j] = B[i * n + j + new_n];
      B21[i * new_n + j] = B[(i + new_n) * n + j];
      B22[i * new_n + j] = B[(i + new_n) * n + j + new_n];
    }
  }

  add(A11, A22, temp1, new_n);
  add(B11, B22, temp2, new_n);
  strassen(temp1, temp2, M1, new_n);

  add(A21, A22, temp1, new_n);
  strassen(temp1, B11, M2, new_n);

  subtract(B12, B22, temp1, new_n);
  strassen(A11, temp1, M3, new_n);

  subtract(B21, B11, temp1, new_n);
  strassen(A22, temp1, M4, new_n);

  add(A11, A12, temp1, new_n);
  strassen(temp1, B22, M5, new_n);

  subtract(A21, A11, temp1, new_n);
  add(B11, B12, temp2, new_n);
  strassen(temp1, temp2, M6, new_n);

  subtract(A12, A22, temp1, new_n);
  add(B21, B22, temp2, new_n);
  strassen(temp1, temp2, M7, new_n);

  add(M1, M4, temp1, new_n);
  subtract(temp1, M5, temp2, new_n);
  add(temp2, M7, C11, new_n);

  add(M3, M5, C12, new_n);

  add(M2, M4, C21, new_n);

  add(M1, M3, temp1, new_n);
  subtract(temp1, M2, temp2, new_n);
  add(temp2, M6, C22, new_n);

  for (size_t i = 0; i < new_n; ++i) {
    for (size_t j = 0; j < new_n; ++j) {
      C[i * n + j] = C11[i * new_n + j];
      C[i * n + j + new_n] = C12[i * new_n + j];
      C[(i + new_n) * n + j] = C21[i * new_n + j];
      C[(i + new_n) * n + j + new_n] = C22[i * new_n + j];
    }
  }
}

void StrassenAlgorithmSequential::add(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C[i * n + j] = A[i * n + j] + B[i * n + j];
    }
  }
}

void StrassenAlgorithmSequential::subtract(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C[i * n + j] = A[i * n + j] - B[i * n + j];
    }
  }
}

bool StrassenAlgorithmSequential::pre_processing() {
  internal_order_test();
  n = *reinterpret_cast<size_t*>(taskData->inputs[0]);

  A_.assign(n * n, 0.0);
  B_.assign(n * n, 0.0);
  C_.assign(n * n, 0.0);

  auto* A_input = reinterpret_cast<double*>(taskData->inputs[1]);
  auto* B_input = reinterpret_cast<double*>(taskData->inputs[2]);

  std::copy(A_input, A_input + n * n, A_.begin());
  std::copy(B_input, B_input + n * n, B_.begin());

  return true;
}

bool StrassenAlgorithmSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
    return false;
  }

  n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  return n > 0 && (n & (n - 1)) == 0;
}

bool StrassenAlgorithmSequential::run() {
  internal_order_test();
  strassen(A_, B_, C_, n);
  return true;
}

bool StrassenAlgorithmSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < C_.size(); ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = C_[i];
  }
  return true;
}

bool StrassenAlgorithmParallel::pre_processing() {
  internal_order_test();
  sizes_a.resize(world.size());
  displs_a.resize(world.size());

  if (world.rank() == 0) {
    n = *reinterpret_cast<size_t*>(taskData->inputs[0]);

    A_.assign(n * n, 0.0);
    B_.assign(n * n, 0.0);
    C_.assign(n * n, 0.0);

    auto* A_input = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* B_input = reinterpret_cast<double*>(taskData->inputs[2]);

    std::copy(A_input, A_input + n * n, A_.begin());
    std::copy(B_input, B_input + n * n, B_.begin());

    calculate_distribution(n, world.size(), sizes_a, displs_a);
  }
  return true;
}

bool StrassenAlgorithmParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
      return false;
    }

    n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    return n > 0 && (n & (n - 1)) == 0;
  }
  return true;
}

bool StrassenAlgorithmParallel::run() {
  internal_order_test();
  boost::mpi::broadcast(world, n, 0);

  int loc_mat_size = sizes_a[world.rank()];
  local_A.resize(loc_mat_size);
  local_B.resize(loc_mat_size);
  local_C.resize(loc_mat_size);

  if (world.rank() == 0) {
    boost::mpi::scatterv(world, A_.data(), sizes_a, displs_a, local_A.data(), loc_mat_size, 0);
    boost::mpi::scatterv(world, B_.data(), sizes_a, displs_a, local_B.data(), loc_mat_size, 0);
  } else {
    boost::mpi::scatterv(world, nullptr, sizes_a, displs_a, local_A.data(), loc_mat_size, 0);
    boost::mpi::scatterv(world, nullptr, sizes_a, displs_a, local_B.data(), loc_mat_size, 0);
  }

  strassen(local_A, local_B, local_C, n / world.size());

  if (world.rank() == 0) {
    boost::mpi::gatherv(world, local_C.data(), sizes_a[world.rank()], C_.data(), sizes_a, displs_a, 0);
  } else {
    boost::mpi::gatherv(world, local_C.data(), sizes_a[world.rank()], 0);
  }

  return true;
}

bool StrassenAlgorithmParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < C_.size(); ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = C_[i];
    }
  }
  return true;
}

void StrassenAlgorithmParallel::calculate_distribution(int rows, int num_proc, std::vector<int>& sizes, std::vector<int>& displs) {
  sizes.resize(num_proc, 0);
  displs.resize(num_proc, -1);

  if (num_proc > rows) {
    for (int i = 0; i < rows; ++i) {
      sizes[i] = rows;
      displs[i] = i * rows;
    }
  } else {
    int a = rows / num_proc;
    int b = rows % num_proc;

    int offset = 0;
    for (int i = 0; i < num_proc; ++i) {
      if (b-- > 0) {
        sizes[i] = (a + 1) * rows;
      } else {
        sizes[i] = a * rows;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}

void StrassenAlgorithmParallel::strassen(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
  if (n <= 64) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        C[i * n + j] = 0;
        for (size_t k = 0; k < n; ++k) {
          C[i * n + j] += A[i * n + k] * B[k * n + j];
        }
      }
    }
    return;
  }

  size_t new_n = n / 2;
  std::vector<double> A11(new_n * new_n);
  std::vector<double> A12(new_n * new_n);
  std::vector<double> A21(new_n * new_n);
  std::vector<double> A22(new_n * new_n);
  std::vector<double> B11(new_n * new_n);
  std::vector<double> B12(new_n * new_n);
  std::vector<double> B21(new_n * new_n);
  std::vector<double> B22(new_n * new_n);
  std::vector<double> C11(new_n * new_n);
  std::vector<double> C12(new_n * new_n);
  std::vector<double> C21(new_n * new_n);
  std::vector<double> C22(new_n * new_n);
  std::vector<double> M1(new_n * new_n);
  std::vector<double> M2(new_n * new_n);
  std::vector<double> M3(new_n * new_n);
  std::vector<double> M4(new_n * new_n);
  std::vector<double> M5(new_n * new_n);
  std::vector<double> M6(new_n * new_n);
  std::vector<double> M7(new_n * new_n);
  std::vector<double> temp1(new_n * new_n);
  std::vector<double> temp2(new_n * new_n);

  for (size_t i = 0; i < new_n; ++i) {
    for (size_t j = 0; j < new_n; ++j) {
      A11[i * new_n + j] = A[i * n + j];
      A12[i * new_n + j] = A[i * n + j + new_n];
      A21[i * new_n + j] = A[(i + new_n) * n + j];
      A22[i * new_n + j] = A[(i + new_n) * n + j + new_n];

      B11[i * new_n + j] = B[i * n + j];
      B12[i * new_n + j] = B[i * n + j + new_n];
      B21[i * new_n + j] = B[(i + new_n) * n + j];
      B22[i * new_n + j] = B[(i + new_n) * n + j + new_n];
    }
  }

  add(A11, A22, temp1, new_n);
  add(B11, B22, temp2, new_n);
  strassen(temp1, temp2, M1, new_n);

  add(A21, A22, temp1, new_n);
  strassen(temp1, B11, M2, new_n);

  subtract(B12, B22, temp1, new_n);
  strassen(A11, temp1, M3, new_n);

  subtract(B21, B11, temp1, new_n);
  strassen(A22, temp1, M4, new_n);

  add(A11, A12, temp1, new_n);
  strassen(temp1, B22, M5, new_n);

  subtract(A21, A11, temp1, new_n);
  add(B11, B12, temp2, new_n);
  strassen(temp1, temp2, M6, new_n);

  subtract(A12, A22, temp1, new_n);
  add(B21, B22, temp2, new_n);
  strassen(temp1, temp2, M7, new_n);

  add(M1, M4, temp1, new_n);
  subtract(temp1, M5, temp2, new_n);
  add(temp2, M7, C11, new_n);

  add(M3, M5, C12, new_n);

  add(M2, M4, C21, new_n);

  add(M1, M3, temp1, new_n);
  subtract(temp1, M2, temp2, new_n);
  add(temp2, M6, C22, new_n);

  for (size_t i = 0; i < new_n; ++i) {
    for (size_t j = 0; j < new_n; ++j) {
      C[i * n + j] = C11[i * new_n + j];
      C[i * n + j + new_n] = C12[i * new_n + j];
      C[(i + new_n) * n + j] = C21[i * new_n + j];
      C[(i + new_n) * n + j + new_n] = C22[i * new_n + j];
    }
  }
}

void StrassenAlgorithmParallel::add(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C[i * n + j] = A[i * n + j] + B[i * n + j];
    }
  }
}

void StrassenAlgorithmParallel::subtract(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C[i * n + j] = A[i * n + j] - B[i * n + j];
    }
  }
}

}  // namespace nasedkin_e_strassen_algorithm_mpi