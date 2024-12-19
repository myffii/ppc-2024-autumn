#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

namespace nasedkin_e_strassen_algorithm_mpi {

// Метод для последовательной версии алгоритма Штрассена
bool StrassenAlgorithmMPISequential::pre_processing() {
  internal_order_test();
  n = *reinterpret_cast<size_t*>(taskData->inputs[0]);

  A_.assign(n, std::vector<double>(n, 0.0));
  B_.assign(n, std::vector<double>(n, 0.0));
  C_.assign(n, std::vector<double>(n, 0.0));

  auto* A_input = reinterpret_cast<double*>(taskData->inputs[1]);
  auto* B_input = reinterpret_cast<double*>(taskData->inputs[2]);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      A_[i][j] = A_input[i * n + j];
      B_[i][j] = B_input[i * n + j];
    }
  }

  return true;
}

bool StrassenAlgorithmMPISequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
    return false;
  }

  n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  return n > 0;
}

bool StrassenAlgorithmMPISequential::run() {
  internal_order_test();
  C_ = strassen(A_, B_);
  return true;
}

bool StrassenAlgorithmMPISequential::post_processing() {
  internal_order_test();
  auto* C_output = reinterpret_cast<double*>(taskData->outputs[0]);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C_output[i * n + j] = C_[i][j];
    }
  }
  return true;
}

void StrassenAlgorithmMPISequential::split(const std::vector<std::vector<double>>& matrix,
                                           std::vector<std::vector<double>>& a,
                                           std::vector<std::vector<double>>& b,
                                           std::vector<std::vector<double>>& c,
                                           std::vector<std::vector<double>>& d) {
  size_t n = matrix.size();
  size_t half = n / 2;

  a.assign(half, std::vector<double>(half, 0.0));
  b.assign(half, std::vector<double>(half, 0.0));
  c.assign(half, std::vector<double>(half, 0.0));
  d.assign(half, std::vector<double>(half, 0.0));

  for (size_t i = 0; i < half; ++i) {
    for (size_t j = 0; j < half; ++j) {
      a[i][j] = matrix[i][j];
      b[i][j] = matrix[i][j + half];
      c[i][j] = matrix[i + half][j];
      d[i][j] = matrix[i + half][j + half];
    }
  }
}

std::vector<std::vector<double>> StrassenAlgorithmMPISequential::strassen(const std::vector<std::vector<double>>& A,
                                                                          const std::vector<std::vector<double>>& B) {
  size_t n = A.size();
  if (n <= 2) {
    return brute_force(A, B);
  }

  std::vector<std::vector<double>> a;
  std::vector<std::vector<double>> b;
  std::vector<std::vector<double>> c;
  std::vector<std::vector<double>> d;
  std::vector<std::vector<double>> e;
  std::vector<std::vector<double>> f;
  std::vector<std::vector<double>> g;
  std::vector<std::vector<double>> h;

  split(A, a, b, c, d);
  split(B, e, f, g, h);

  std::vector<std::vector<double>> p1 = strassen(a, subtract(f, h));
  std::vector<std::vector<double>> p2 = strassen(add(a, b), h);
  std::vector<std::vector<double>> p3 = strassen(add(c, d), e);
  std::vector<std::vector<double>> p4 = strassen(d, subtract(g, e));
  std::vector<std::vector<double>> p5 = strassen(add(a, d), add(e, h));
  std::vector<std::vector<double>> p6 = strassen(subtract(b, d), add(g, h));
  std::vector<std::vector<double>> p7 = strassen(subtract(a, c), add(e, f));

  std::vector<std::vector<double>> C11 = add(subtract(add(p5, p4), p2), p6);
  std::vector<std::vector<double>> C12 = add(p1, p2);
  std::vector<std::vector<double>> C21 = add(p3, p4);
  std::vector<std::vector<double>> C22 = subtract(subtract(add(p5, p1), p3), p7);

  std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < n / 2; ++i) {
    for (size_t j = 0; j < n / 2; ++j) {
      C[i][j] = C11[i][j];
      C[i][j + n / 2] = C12[i][j];
      C[i + n / 2][j] = C21[i][j];
      C[i + n / 2][j + n / 2] = C22[i][j];
    }
  }

  return C;
}

std::vector<std::vector<double>> StrassenAlgorithmMPISequential::brute_force(const std::vector<std::vector<double>>& A,
                                                                             const std::vector<std::vector<double>>& B) {
  size_t n = A.size();
  std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < n; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return C;
}

std::vector<std::vector<double>> StrassenAlgorithmMPISequential::add(const std::vector<std::vector<double>>& A,
                                                                     const std::vector<std::vector<double>>& B) {
  size_t n = A.size();
  std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }

  return C;
}

std::vector<std::vector<double>> StrassenAlgorithmMPISequential::subtract(const std::vector<std::vector<double>>& A,
                                                                          const std::vector<std::vector<double>>& B) {
  size_t n = A.size();
  std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C[i][j] = A[i][j] - B[i][j];
    }
  }

  return C;
}

// Метод для параллельной версии алгоритма Штрассена
bool StrassenAlgorithmMPIParallel::pre_processing() {
  internal_order_test();
  sizes_a.resize(world.size());
  displs_a.resize(world.size());

  if (world.rank() == 0) {
    n = *reinterpret_cast<size_t*>(taskData->inputs[0]);

    A_.assign(n, std::vector<double>(n, 0.0));
    B_.assign(n, std::vector<double>(n, 0.0));
    C_.assign(n, std::vector<double>(n, 0.0));

    auto* A_input = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* B_input = reinterpret_cast<double*>(taskData->inputs[2]);

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        A_[i][j] = A_input[i * n + j];
        B_[i][j] = B_input[i * n + j];
      }
    }

    calculate_distribution(n, world.size(), sizes_a, displs_a);
  }

  return true;
}

bool StrassenAlgorithmMPIParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
      return false;
    }

    n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    if (n <= 0) {
      return false;
    }
  }

  return true;
}

bool StrassenAlgorithmMPIParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, sizes_a, 0);
  boost::mpi::broadcast(world, displs_a, 0);
  boost::mpi::broadcast(world, n, 0);

  int loc_size = sizes_a[world.rank()];
  local_A.resize(loc_size, std::vector<double>(n, 0.0));
  local_B.resize(loc_size, std::vector<double>(n, 0.0));
  local_C.resize(loc_size, std::vector<double>(n, 0.0));

  // Отправка и получение данных с использованием объекта communicator
  if (world.rank() == 0) {
    for (int i = 0; i < world.size(); ++i) {
      for (int j = 0; j < sizes_a[i]; ++j) {
        for (size_t k = 0; k < n; ++k) {
          local_A[j][k] = A_[displs_a[i] + j][k];
          local_B[j][k] = B_[displs_a[i] + j][k];
        }
      }
      world.send(i, 0, local_A);  // Используйте world.send вместо boost::mpi::send
      world.send(i, 1, local_B);  // Используйте world.send вместо boost::mpi::send
    }
  } else {
    world.recv(0, 0, local_A);  // Используйте world.recv вместо boost::mpi::recv
    world.recv(0, 1, local_B);  // Используйте world.recv вместо boost::mpi::recv
  }

  // Выполнение локального умножения матриц
  for (int i = 0; i < loc_size; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < n; ++k) {
        local_C[i][j] += local_A[i][k] * local_B[k][j];
      }
    }
  }

  if (world.rank() == 0) {
    for (int i = 0; i < world.size(); ++i) {
      world.recv(i, 2, local_C);  // Используйте world.recv вместо boost::mpi::recv
      for (int j = 0; j < sizes_a[i]; ++j) {
        for (size_t k = 0; k < n; ++k) {
          C_[displs_a[i] + j][k] = local_C[j][k];
        }
      }
    }
  } else {
    world.send(0, 2, local_C);  // Используйте world.send вместо boost::mpi::send
  }

  return true;
}

bool StrassenAlgorithmMPIParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* C_output = reinterpret_cast<double*>(taskData->outputs[0]);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        C_output[i * n + j] = C_[i][j];
      }
    }
  }
  return true;
}

void StrassenAlgorithmMPIParallel::calculate_distribution(int rows, int num_proc,
                                                          std::vector<int>& sizes,
                                                          std::vector<int>& displs) {
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
        sizes[i] = a + 1;
      } else {
        sizes[i] = a;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}

}  // namespace nasedkin_e_strassen_algorithm_mpi