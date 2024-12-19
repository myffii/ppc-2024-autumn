#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

namespace nasedkin_e_strassen_algorithm {

StrassenMPITaskParallel::StrassenMPITaskParallel(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B)
    : A_(A), B_(B) {}

std::vector<std::vector<double>> StrassenMPITaskParallel::run() {
  return strassen_parallel(A_, B_);
}

std::vector<std::vector<double>> StrassenMPITaskParallel::strassen_parallel(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
  int n = A.size();
  if (n <= 2) {
    return brute_force(A, B);
  }

  // Разделение матриц на подматрицы
  auto a = split(A, 0, 0, n / 2);
  auto b = split(A, 0, n / 2, n / 2);
  auto c = split(A, n / 2, 0, n / 2);
  auto d = split(A, n / 2, n / 2, n / 2);

  auto e = split(B, 0, 0, n / 2);
  auto f = split(B, 0, n / 2, n / 2);
  auto g = split(B, n / 2, 0, n / 2);
  auto h = split(B, n / 2, n / 2, n / 2);

  // Распределение задач между процессами
  std::vector<std::vector<double>> p1, p2, p3, p4, p5, p6, p7;

  if (world.rank() == 0) {
    // Распределение задач между процессами
    std::vector<boost::mpi::request> requests;

    requests.push_back(world.isend(1, 0, add(a, d)));
    requests.push_back(world.isend(1, 1, add(e, h)));

    requests.push_back(world.isend(2, 0, add(c, d)));
    requests.push_back(world.isend(2, 1, e));

    requests.push_back(world.isend(3, 0, a));
    requests.push_back(world.isend(3, 1, subtract(f, h)));

    requests.push_back(world.isend(4, 0, d));
    requests.push_back(world.isend(4, 1, subtract(g, e)));

    requests.push_back(world.isend(5, 0, add(a, b)));
    requests.push_back(world.isend(5, 1, h));

    requests.push_back(world.isend(6, 0, subtract(b, d)));
    requests.push_back(world.isend(6, 1, add(g, h)));

    requests.push_back(world.isend(7, 0, subtract(a, c)));
    requests.push_back(world.isend(7, 1, add(e, f)));

    // Ожидание завершения отправки данных
    boost::mpi::wait_all(requests.begin(), requests.end());

    // Получение результатов от других процессов
    world.recv(1, 2, p1);
    world.recv(2, 2, p2);
    world.recv(3, 2, p3);
    world.recv(4, 2, p4);
    world.recv(5, 2, p5);
    world.recv(6, 2, p6);
    world.recv(7, 2, p7);
  } else {
    // Получение данных от процесса 0
    std::vector<std::vector<double>> local_A, local_B;
    world.recv(0, world.rank() - 1, local_A);
    world.recv(0, world.rank(), local_B);

    // Выполнение вычислений
    auto local_result = strassen_parallel(local_A, local_B);

    // Отправка результата обратно процессу 0
    world.send(0, 2, local_result);
  }

  // Сборка результата на процессе 0
  if (world.rank() == 0) {
    auto C11 = add(subtract(add(p1, p4), p5), p7);
    auto C12 = add(p3, p5);
    auto C21 = add(p2, p4);
    auto C22 = add(subtract(add(p1, p3), p2), p6);

    return combine(C11, C12, C21, C22);
  } else {
    return {}; // Возвращаем пустую матрицу для других процессов
  }
}

std::vector<std::vector<double>> StrassenMPITaskParallel::brute_force(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
  int n = A.size();
  int m = A[0].size();
  int p = B[0].size();
  std::vector<std::vector<double>> C(n, std::vector<double>(p, 0.0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < m; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

std::vector<std::vector<double>> StrassenMPITaskParallel::add(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
  int n = A.size();
  std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
  return C;
}

std::vector<std::vector<double>> StrassenMPITaskParallel::subtract(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
  int n = A.size();
  std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      C[i][j] = A[i][j] - B[i][j];
    }
  }
  return C;
}

std::vector<std::vector<double>> StrassenMPITaskParallel::combine(const std::vector<std::vector<double>>& C11, const std::vector<std::vector<double>>& C12, const std::vector<std::vector<double>>& C21, const std::vector<std::vector<double>>& C22) {
  int n = C11.size();
  std::vector<std::vector<double>> C(2 * n, std::vector<double>(2 * n, 0.0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      C[i][j] = C11[i][j];
      C[i][j + n] = C12[i][j];
      C[i + n][j] = C21[i][j];
      C[i + n][j + n] = C22[i][j];
    }
  }
  return C;
}

std::vector<std::vector<double>> StrassenMPITaskParallel::split(const std::vector<std::vector<double>>& matrix, int row_start, int col_start, int size) {
  std::vector<std::vector<double>> result(size, std::vector<double>(size, 0.0));
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      result[i][j] = matrix[row_start + i][col_start + j];
    }
  }
  return result;
}

}  // namespace nasedkin_e_strassen_algorithm