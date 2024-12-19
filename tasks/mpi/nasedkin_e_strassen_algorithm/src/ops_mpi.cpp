#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

bool nasedkin_e_strassen_algorithm_mpi::StrassenSequential::pre_processing() {
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

bool nasedkin_e_strassen_algorithm_mpi::StrassenSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
    return false;
  }

  n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  return n > 0;  // Исправлено: упрощено условие
}

bool nasedkin_e_strassen_algorithm_mpi::StrassenSequential::run() {
  internal_order_test();
  strassen_multiply(A_, B_, C_, n);
  return true;
}

bool nasedkin_e_strassen_algorithm_mpi::StrassenSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < C_.size(); ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = C_[i];
  }
  return true;
}

void nasedkin_e_strassen_algorithm_mpi::StrassenSequential::strassen_multiply(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
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

  size_t m = n / 2;
  std::vector<double> A11(m * m);  // Исправлено: отдельные объявления
  std::vector<double> A12(m * m);
  std::vector<double> A21(m * m);
  std::vector<double> A22(m * m);

  std::vector<double> B11(m * m);  // Исправлено: отдельные объявления
  std::vector<double> B12(m * m);
  std::vector<double> B21(m * m);
  std::vector<double> B22(m * m);

  std::vector<double> C11(m * m);  // Исправлено: отдельные объявления
  std::vector<double> C12(m * m);
  std::vector<double> C21(m * m);
  std::vector<double> C22(m * m);

  std::vector<double> M1(m * m);  // Исправлено: отдельные объявления
  std::vector<double> M2(m * m);
  std::vector<double> M3(m * m);
  std::vector<double> M4(m * m);
  std::vector<double> M5(m * m);
  std::vector<double> M6(m * m);
  std::vector<double> M7(m * m);

  std::vector<double> temp1(m * m);  // Исправлено: отдельные объявления
  std::vector<double> temp2(m * m);

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < m; ++j) {
      A11[i * m + j] = A[i * n + j];
      A12[i * m + j] = A[i * n + j + m];
      A21[i * m + j] = A[(i + m) * n + j];
      A22[i * m + j] = A[(i + m) * n + j + m];

      B11[i * m + j] = B[i * n + j];
      B12[i * m + j] = B[i * n + j + m];
      B21[i * m + j] = B[(i + m) * n + j];
      B22[i * m + j] = B[(i + m) * n + j + m];
    }
  }

  // M1 = (A11 + A22) * (B11 + B22)
  std::vector<double> A11_plus_A22(m * m);  // Исправлено: отдельные объявления
  std::vector<double> B11_plus_B22(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    A11_plus_A22[i] = A11[i] + A22[i];
    B11_plus_B22[i] = B11[i] + B22[i];
  }
  strassen_multiply(A11_plus_A22, B11_plus_B22, M1, m);

  // M2 = (A21 + A22) * B11
  std::vector<double> A21_plus_A22(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    A21_plus_A22[i] = A21[i] + A22[i];
  }
  strassen_multiply(A21_plus_A22, B11, M2, m);

  // M3 = A11 * (B12 - B22)
  std::vector<double> B12_minus_B22(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    B12_minus_B22[i] = B12[i] - B22[i];
  }
  strassen_multiply(A11, B12_minus_B22, M3, m);

  // M4 = A22 * (B21 - B11)
  std::vector<double> B21_minus_B11(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    B21_minus_B11[i] = B21[i] - B11[i];
  }
  strassen_multiply(A22, B21_minus_B11, M4, m);

  // M5 = (A11 + A12) * B22
  std::vector<double> A11_plus_A12(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    A11_plus_A12[i] = A11[i] + A12[i];
  }
  strassen_multiply(A11_plus_A12, B22, M5, m);

  // M6 = (A21 - A11) * (B11 + B12)
  std::vector<double> A21_minus_A11(m * m);
  std::vector<double> B11_plus_B12(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    A21_minus_A11[i] = A21[i] - A11[i];
    B11_plus_B12[i] = B11[i] + B12[i];
  }
  strassen_multiply(A21_minus_A11, B11_plus_B12, M6, m);

  // M7 = (A12 - A22) * (B21 + B22)
  std::vector<double> A12_minus_A22(m * m);
  std::vector<double> B21_plus_B22(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    A12_minus_A22[i] = A12[i] - A22[i];
    B21_plus_B22[i] = B21[i] + B22[i];
  }
  strassen_multiply(A12_minus_A22, B21_plus_B22, M7, m);

  // C11 = M1 + M4 - M5 + M7
  for (size_t i = 0; i < m * m; ++i) {
    C11[i] = M1[i] + M4[i] - M5[i] + M7[i];
  }

  // C12 = M3 + M5
  for (size_t i = 0; i < m * m; ++i) {
    C12[i] = M3[i] + M5[i];
  }

  // C21 = M2 + M4
  for (size_t i = 0; i < m * m; ++i) {
    C21[i] = M2[i] + M4[i];
  }

  // C22 = M1 - M2 + M3 + M6
  for (size_t i = 0; i < m * m; ++i) {
    C22[i] = M1[i] - M2[i] + M3[i] + M6[i];
  }

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < m; ++j) {
      C[i * n + j] = C11[i * m + j];
      C[i * n + j + m] = C12[i * m + j];
      C[(i + m) * n + j] = C21[i * m + j];
      C[(i + m) * n + j + m] = C22[i * m + j];
    }
  }
}

bool nasedkin_e_strassen_algorithm_mpi::StrassenParallel::pre_processing() {
  internal_order_test();
  sizes_a.resize(world.size());
  displs_a.resize(world.size());

  sizes_b.resize(world.size());
  displs_b.resize(world.size());

  if (world.rank() == 0) {
    n = *reinterpret_cast<size_t*>(taskData->inputs[0]);

    A_.assign(n * n, 0.0);
    B_.assign(n * n, 0.0);
    C_.assign(n * n, 0.0);

    auto* A_input = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* B_input = reinterpret_cast<double*>(taskData->inputs[2]);

    std::copy(A_input, A_input + n * n, A_.begin());
    std::copy(B_input, B_input + n * n, B_.begin());

    calculate_distribution_a(n, world.size(), sizes_a, displs_a);
    calculate_distribution_b(n, world.size(), sizes_b, displs_b);
  }
  return true;
}

bool nasedkin_e_strassen_algorithm_mpi::StrassenParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
      return false;
    }

    n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    return n > 0;  // Исправлено: упрощено условие
  }
  return true;
}

bool nasedkin_e_strassen_algorithm_mpi::StrassenParallel::run() {
  internal_order_test();
  std::vector<double> local_C;

  boost::mpi::broadcast(world, sizes_a, 0);
  boost::mpi::broadcast(world, sizes_b, 0);
  boost::mpi::broadcast(world, displs_b, 0);
  boost::mpi::broadcast(world, n, 0);

  int loc_mat_size = sizes_a[world.rank()];
  int loc_vec_size = sizes_b[world.rank()];

  local_A.resize(loc_mat_size);
  local_B.resize(loc_vec_size);
  local_C.resize(sizes_b[world.rank()]);

  if (world.rank() == 0) {
    boost::mpi::scatterv(world, A_.data(), sizes_a, displs_a, local_A.data(), loc_mat_size, 0);
    boost::mpi::scatterv(world, B_.data(), sizes_b, displs_b, local_B.data(), loc_vec_size, 0);
  } else {
    boost::mpi::scatterv(world, local_A.data(), loc_mat_size, 0);
    boost::mpi::scatterv(world, local_B.data(), loc_vec_size, 0);
  }

  strassen_multiply_parallel(local_A, local_B, local_C, n);

  if (world.rank() == 0) {
    boost::mpi::gatherv(world, local_C.data(), sizes_b[world.rank()], C_.data(), sizes_b, displs_b, 0);
  } else {
    boost::mpi::gatherv(world, local_C.data(), sizes_b[world.rank()], 0);
  }

  return true;
}

bool nasedkin_e_strassen_algorithm_mpi::StrassenParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < C_.size(); ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = C_[i];
    }
  }
  return true;
}

void nasedkin_e_strassen_algorithm_mpi::StrassenParallel::strassen_multiply_parallel(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
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

  size_t m = n / 2;
  std::vector<double> A11(m * m);  // Исправлено: отдельные объявления
  std::vector<double> A12(m * m);
  std::vector<double> A21(m * m);
  std::vector<double> A22(m * m);

  std::vector<double> B11(m * m);  // Исправлено: отдельные объявления
  std::vector<double> B12(m * m);
  std::vector<double> B21(m * m);
  std::vector<double> B22(m * m);

  std::vector<double> C11(m * m);  // Исправлено: отдельные объявления
  std::vector<double> C12(m * m);
  std::vector<double> C21(m * m);
  std::vector<double> C22(m * m);

  std::vector<double> M1(m * m);  // Исправлено: отдельные объявления
  std::vector<double> M2(m * m);
  std::vector<double> M3(m * m);
  std::vector<double> M4(m * m);
  std::vector<double> M5(m * m);
  std::vector<double> M6(m * m);
  std::vector<double> M7(m * m);

  std::vector<double> temp1(m * m);  // Исправлено: отдельные объявления
  std::vector<double> temp2(m * m);

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < m; ++j) {
      A11[i * m + j] = A[i * n + j];
      A12[i * m + j] = A[i * n + j + m];
      A21[i * m + j] = A[(i + m) * n + j];
      A22[i * m + j] = A[(i + m) * n + j + m];

      B11[i * m + j] = B[i * n + j];
      B12[i * m + j] = B[i * n + j + m];
      B21[i * m + j] = B[(i + m) * n + j];
      B22[i * m + j] = B[(i + m) * n + j + m];
    }
  }

  // M1 = (A11 + A22) * (B11 + B22)
  std::vector<double> A11_plus_A22(m * m);  // Исправлено: отдельные объявления
  std::vector<double> B11_plus_B22(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    A11_plus_A22[i] = A11[i] + A22[i];
    B11_plus_B22[i] = B11[i] + B22[i];
  }
  strassen_multiply_parallel(A11_plus_A22, B11_plus_B22, M1, m);

  // M2 = (A21 + A22) * B11
  std::vector<double> A21_plus_A22(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    A21_plus_A22[i] = A21[i] + A22[i];
  }
  strassen_multiply_parallel(A21_plus_A22, B11, M2, m);

  // M3 = A11 * (B12 - B22)
  std::vector<double> B12_minus_B22(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    B12_minus_B22[i] = B12[i] - B22[i];
  }
  strassen_multiply_parallel(A11, B12_minus_B22, M3, m);

  // M4 = A22 * (B21 - B11)
  std::vector<double> B21_minus_B11(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    B21_minus_B11[i] = B21[i] - B11[i];
  }
  strassen_multiply_parallel(A22, B21_minus_B11, M4, m);

  // M5 = (A11 + A12) * B22
  std::vector<double> A11_plus_A12(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    A11_plus_A12[i] = A11[i] + A12[i];
  }
  strassen_multiply_parallel(A11_plus_A12, B22, M5, m);

  // M6 = (A21 - A11) * (B11 + B12)
  std::vector<double> A21_minus_A11(m * m);
  std::vector<double> B11_plus_B12(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    A21_minus_A11[i] = A21[i] - A11[i];
    B11_plus_B12[i] = B11[i] + B12[i];
  }
  strassen_multiply_parallel(A21_minus_A11, B11_plus_B12, M6, m);

  // M7 = (A12 - A22) * (B21 + B22)
  std::vector<double> A12_minus_A22(m * m);
  std::vector<double> B21_plus_B22(m * m);
  for (size_t i = 0; i < m * m; ++i) {
    A12_minus_A22[i] = A12[i] - A22[i];
    B21_plus_B22[i] = B21[i] + B22[i];
  }
  strassen_multiply_parallel(A12_minus_A22, B21_plus_B22, M7, m);

  // C11 = M1 + M4 - M5 + M7
  for (size_t i = 0; i < m * m; ++i) {
    C11[i] = M1[i] + M4[i] - M5[i] + M7[i];
  }

  // C12 = M3 + M5
  for (size_t i = 0; i < m * m; ++i) {
    C12[i] = M3[i] + M5[i];
  }

  // C21 = M2 + M4
  for (size_t i = 0; i < m * m; ++i) {
    C21[i] = M2[i] + M4[i];
  }

  // C22 = M1 - M2 + M3 + M6
  for (size_t i = 0; i < m * m; ++i) {
    C22[i] = M1[i] - M2[i] + M3[i] + M6[i];
  }

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < m; ++j) {
      C[i * n + j] = C11[i * m + j];
      C[i * n + j + m] = C12[i * m + j];
      C[(i + m) * n + j] = C21[i * m + j];
      C[(i + m) * n + j + m] = C22[i * m + j];
    }
  }
}

void nasedkin_e_strassen_algorithm_mpi::StrassenParallel::calculate_distribution_a(int rows, int num_proc, std::vector<int>& sizes, std::vector<int>& displs) {
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

void nasedkin_e_strassen_algorithm_mpi::StrassenParallel::calculate_distribution_b(int rows, int num_proc, std::vector<int>& sizes, std::vector<int>& displs) {
  sizes.resize(num_proc, 0);
  displs.resize(num_proc, -1);

  if (num_proc > rows) {
    for (int i = 0; i < rows; ++i) {
      sizes[i] = 1;
      displs[i] = i;
    }
  } else {
    int a = rows / num_proc;
    int b = rows % num_proc;

    int offset = 0;
    for (int i = 0; i < num_proc; ++i) {
      if (b-- > 0) {
        sizes[i] = (a + 1);
      } else {
        sizes[i] = a;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}