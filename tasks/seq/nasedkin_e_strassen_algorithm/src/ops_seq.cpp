#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

#include <cmath>
#include <cstdlib>
#include <ctime>

namespace nasedkin_e_strassen_algorithm {

    bool StrassenAlgorithmSeq::pre_processing() {
        if (!validation()) {
            return false;
        }

        result.resize(n, std::vector<double>(n, 0.0));
        return true;
    }

    bool StrassenAlgorithmSeq::validation() {
        if (n <= 0 || (n & (n - 1)) != 0) {
            return false;
        }

        A.resize(n, std::vector<double>(n, 0.0));
        B.resize(n, std::vector<double>(n, 0.0));

        return true;
    }

    bool StrassenAlgorithmSeq::run() {
        strassen_multiply(A, B, result, n);
        return true;
    }

    bool StrassenAlgorithmSeq::post_processing() { return true; }

    void StrassenAlgorithmSeq::strassen_multiply(const std::vector<std::vector<double>>& local_A, const std::vector<std::vector<double>>& local_B, std::vector<std::vector<double>>& local_result, int size) {
        if (size <= 64) {
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    local_result[i][j] = 0;
                    for (int k = 0; k < size; ++k) {
                        local_result[i][j] += local_A[i][k] * local_B[k][j];
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

        std::vector<std::vector<double>> C11(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> C12(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> C21(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> C22(new_size, std::vector<double>(new_size));

        split_matrix(local_A, A11, A12, A21, A22, new_size);
        split_matrix(local_B, B11, B12, B21, B22, new_size);

        std::vector<std::vector<double>> M1(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> M2(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> M3(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> M4(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> M5(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> M6(new_size, std::vector<double>(new_size));
        std::vector<std::vector<double>> M7(new_size, std::vector<double>(new_size));

        add_matrices(A11, A22, M1, new_size);
        add_matrices(B11, B22, M2, new_size);
        strassen_multiply(M1, M2, M3, new_size);

        add_matrices(A21, A22, M1, new_size);
        strassen_multiply(M1, B11, M4, new_size);

        subtract_matrices(B12, B22, M1, new_size);
        strassen_multiply(A11, M1, M5, new_size);

        subtract_matrices(B21, B11, M1, new_size);
        strassen_multiply(A22, M1, M6, new_size);

        add_matrices(A11, A12, M1, new_size);
        strassen_multiply(M1, B22, M7, new_size);

        add_matrices(M3, M6, M1, new_size);
        subtract_matrices(M5, M4, M2, new_size);
        add_matrices(M1, M2, C11, new_size);

        add_matrices(M5, M7, C12, new_size);

        add_matrices(M6, M4, C21, new_size);

        subtract_matrices(M3, M7, M1, new_size);
        add_matrices(M5, M1, C22, new_size);

        join_matrices(C11, C12, C21, C22, local_result, size);
    }

    void StrassenAlgorithmSeq::add_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
    }

    void StrassenAlgorithmSeq::subtract_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
    }

    void StrassenAlgorithmSeq::split_matrix(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& A11, std::vector<std::vector<double>>& A12, std::vector<std::vector<double>>& A21, std::vector<std::vector<double>>& A22, int size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                A11[i][j] = A[i][j];
                A12[i][j] = A[i][j + size];
                A21[i][j] = A[i + size][j];
                A22[i][j] = A[i + size][j + size];
            }
        }
    }

    void StrassenAlgorithmSeq::join_matrices(const std::vector<std::vector<double>>& C11, const std::vector<std::vector<double>>& C12, const std::vector<std::vector<double>>& C21, const std::vector<std::vector<double>>& C22, std::vector<std::vector<double>>& C, int size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                C[i][j] = C11[i][j];
                C[i][j + size] = C12[i][j];
                C[i + size][j] = C21[i][j];
                C[i + size][j + size] = C22[i][j];
            }
        }
    }

    void StrassenAlgorithmSeq::set_matrices(const std::vector<std::vector<double>>& matrixA, const std::vector<std::vector<double>>& matrixB) {
        A = matrixA;
        B = matrixB;
        n = static_cast<int>(matrixA.size());
    }

    void StrassenAlgorithmSeq::generate_random_matrix(int size, std::vector<std::vector<double>>& matrix) {
        matrix.resize(size, std::vector<double>(size, 0.0));

        std::srand(static_cast<unsigned>(std::time(nullptr)));

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                matrix[i][j] = static_cast<double>(std::rand() % 10 + 1);
            }
        }
    }

}  // namespace nasedkin_e_strassen_algorithm