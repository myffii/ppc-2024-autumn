#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

namespace nasedkin_e_strassen_algorithm {

bool StrassenMatrixMultiplicationSequential::pre_processing() {
  internal_order_test();

  n_ = *reinterpret_cast<int*>(taskData->inputs[0]);
  A_.assign(reinterpret_cast<double*>(taskData->inputs[1]), reinterpret_cast<double*>(taskData->inputs[1]) + n_ * n_);
  B_.assign(reinterpret_cast<double*>(taskData->inputs[2]), reinterpret_cast<double*>(taskData->inputs[2]) + n_ * n_);

  return true;
}

bool StrassenMatrixMultiplicationSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == n_ * n_;
}

bool StrassenMatrixMultiplicationSequential::run() {
  internal_order_test();
  C_ = strassen(A_, B_, n_);
  return true;
}

bool StrassenMatrixMultiplicationSequential::post_processing() {
  internal_order_test();
  std::copy(C_.begin(), C_.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}

std::vector<double> StrassenMatrixMultiplicationSequential::strassen(const std::vector<double>& A, const std::vector<double>& B, int n) {
  if (n <= 2) {
    std::vector<double> C(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
          C[i * n + j] += A[i * n + k] * B[k * n + j];
        }
      }
    }
    return C;
  }

  int half = n / 2;
  std::vector<double> A11(half * half), A12(half * half), A21(half * half), A22(half * half);
  std::vector<double> B11(half * half), B12(half * half), B21(half * half), B22(half * half);

  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      A11[i * half + j] = A[i * n + j];
      A12[i * half + j] = A[i * n + j + half];
      A21[i * half + j] = A[(i + half) * n + j];
      A22[i * half + j] = A[(i + half) * n + j + half];

      B11[i * half + j] = B[i * n + j];
      B12[i * half + j] = B[i * n + j + half];
      B21[i * half + j] = B[(i + half) * n + j];
      B22[i * half + j] = B[(i + half) * n + j + half];
    }
  }

  auto P1 = strassen(add(A11, A22, half), add(B11, B22, half), half);
  auto P2 = strassen(add(A21, A22, half), B11, half);
  auto P3 = strassen(A11, subtract(B12, B22, half), half);
  auto P4 = strassen(A22, subtract(B21, B11, half), half);
  auto P5 = strassen(add(A11, A12, half), B22, half);
  auto P6 = strassen(subtract(A21, A11, half), add(B11, B12, half), half);
  auto P7 = strassen(subtract(A12, A22, half), add(B21, B22, half), half);

  auto C11 = add(subtract(add(P1, P4, half), P5, half), P7, half);
  auto C12 = add(P3, P5, half);
  auto C21 = add(P2, P4, half);
  auto C22 = add(subtract(add(P1, P3, half), P2, half), P6, half);

  std::vector<double> C(n * n, 0.0);
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      C[i * n + j] = C11[i * half + j];
      C[i * n + j + half] = C12[i * half + j];
      C[(i + half) * n + j] = C21[i * half + j];
      C[(i + half) * n + j + half] = C22[i * half + j];
    }
  }

  return C;
}

std::vector<double> StrassenMatrixMultiplicationSequential::add(const std::vector<double>& A, const std::vector<double>& B, int n) {
  std::vector<double> C(n * n);
  for (int i = 0; i < n * n; ++i) {
    C[i] = A[i] + B[i];
  }
  return C;
}

std::vector<double> StrassenMatrixMultiplicationSequential::subtract(const std::vector<double>& A, const std::vector<double>& B, int n) {
  std::vector<double> C(n * n);
  for (int i = 0; i < n * n; ++i) {
    C[i] = A[i] - B[i];
  }
  return C;
}

bool StrassenMatrixMultiplicationParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    n_ = *reinterpret_cast<int*>(taskData->inputs[0]);
    A_.assign(reinterpret_cast<double*>(taskData->inputs[1]), reinterpret_cast<double*>(taskData->inputs[1]) + n_ * n_);
    B_.assign(reinterpret_cast<double*>(taskData->inputs[2]), reinterpret_cast<double*>(taskData->inputs[2]) + n_ * n_);
  }

  boost::mpi::broadcast(world, n_, 0);
  boost::mpi::broadcast(world, A_, 0);
  boost::mpi::broadcast(world, B_, 0);

  return true;
}

bool StrassenMatrixMultiplicationParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == n_ * n_;
  }
  return true;
}

bool StrassenMatrixMultiplicationParallel::run() {
  internal_order_test();
  C_ = parallel_strassen(A_, B_, n_);
  return true;
}

bool StrassenMatrixMultiplicationParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(C_.begin(), C_.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  }
  return true;
}

std::vector<double> StrassenMatrixMultiplicationParallel::parallel_strassen(const std::vector<double>& A, const std::vector<double>& B, int n) {
  if (n <= 2) {
    return strassen(A, B, n);
  }

  int half = n / 2;
  std::vector<double> A11(half * half), A12(half * half), A21(half * half), A22(half * half);
  std::vector<double> B11(half * half), B12(half * half), B21(half * half), B22(half * half);

  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      A11[i * half + j] = A[i * n + j];
      A12[i * half + j] = A[i * n + j + half];
      A21[i * half + j] = A[(i + half) * n + j];
      A22[i * half + j] = A[(i + half) * n + j + half];

      B11[i * half + j] = B[i * n + j];
      B12[i * half + j] = B[i * n + j + half];
      B21[i * half + j] = B[(i + half) * n + j];
      B22[i * half + j] = B[(i + half) * n + j + half];
    }
  }

  std::vector<double> P1, P2, P3, P4, P5, P6, P7;

  if (world.size() >= 7) {
    std::vector<boost::mpi::request> requests;
    requests.push_back(world.isend(1, 0, add(A11, A22, half)));
    requests.push_back(world.isend(2, 0, add(B11, B22, half)));
    requests.push_back(world.isend(3, 0, add(A21, A22, half)));
    requests.push_back(world.isend(4, 0, B11));
    requests.push_back(world.isend(5, 0, A11));
    requests.push_back(world.isend(6, 0, subtract(B12, B22, half)));
    requests.push_back(world.isend(7, 0, A22));
    requests.push_back(world.isend(8, 0, subtract(B21, B11, half)));
    requests.push_back(world.isend(9, 0, add(A11, A12, half)));
    requests.push_back(world.isend(10, 0, B22));
    requests.push_back(world.isend(11, 0, subtract(A21, A11, half)));
    requests.push_back(world.isend(12, 0, add(B11, B12, half)));
    requests.push_back(world.isend(13, 0, subtract(A12, A22, half)));
    requests.push_back(world.isend(14, 0, add(B21, B22, half)));

    P1 = world.recv<std::vector<double>>(1, 1);
    P2 = world.recv<std::vector<double>>(2, 1);
    P3 = world.recv<std::vector<double>>(3, 1);
    P4 = world.recv<std::vector<double>>(4, 1);
    P5 = world.recv<std::vector<double>>(5, 1);
    P6 = world.recv<std::vector<double>>(6, 1);
    P7 = world.recv<std::vector<double>>(7, 1);
  } else {
    P1 = strassen(add(A11, A22, half), add(B11, B22, half), half);
    P2 = strassen(add(A21, A22, half), B11, half);
    P3 = strassen(A11, subtract(B12, B22, half), half);
    P4 = strassen(A22, subtract(B21, B11, half), half);
    P5 = strassen(add(A11, A12, half), B22, half);
    P6 = strassen(subtract(A21, A11, half), add(B11, B12, half), half);
    P7 = strassen(subtract(A12, A22, half), add(B21, B22, half), half);
  }

  auto C11 = add(subtract(add(P1, P4, half), P5, half), P7, half);
  auto C12 = add(P3, P5, half);
  auto C21 = add(P2, P4, half);
  auto C22 = add(subtract(add(P1, P3, half), P2, half), P6, half);

  std::vector<double> C(n * n, 0.0);
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      C[i * n + j] = C11[i * half + j];
      C[i * n + j + half] = C12[i * half + j];
      C[(i + half) * n + j] = C21[i * half + j];
      C[(i + half) * n + j + half] = C22[i * half + j];
    }
  }

  return C;
}

std::vector<double> StrassenMatrixMultiplicationParallel::add(const std::vector<double>& A, const std::vector<double>& B, int n) {
  std::vector<double> C(n * n);
  for (int i = 0; i < n * n; ++i) {
    C[i] = A[i] + B[i];
  }
  return C;
}

std::vector<double> StrassenMatrixMultiplicationParallel::subtract(const std::vector<double>& A, const std::vector<double>& B, int n) {
  std::vector<double> C(n * n);
  for (int i = 0; i < n * n; ++i) {
    C[i] = A[i] - B[i];
  }
  return C;
}

}  // namespace nasedkin_e_strassen_algorithm