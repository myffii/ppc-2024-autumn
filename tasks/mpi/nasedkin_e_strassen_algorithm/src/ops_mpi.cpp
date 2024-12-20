#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

namespace nasedkin_e_strassen_algorithm_mpi {

    void StrassenAlgorithmSequential::strassenMultiply(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
        if (n <= 64) {  // Base case: use standard matrix multiplication for small matrices
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

        size_t half = n / 2;

        std::vector<double> A11(half * half);
        std::vector<double> A12(half * half);
        std::vector<double> A21(half * half);
        std::vector<double> A22(half * half);

        std::vector<double> B11(half * half);
        std::vector<double> B12(half * half);
        std::vector<double> B21(half * half);
        std::vector<double> B22(half * half);

        std::vector<double> C11(half * half);
        std::vector<double> C12(half * half);
        std::vector<double> C21(half * half);
        std::vector<double> C22(half * half);

        std::vector<double> M1(half * half);
        std::vector<double> M2(half * half);
        std::vector<double> M3(half * half);
        std::vector<double> M4(half * half);
        std::vector<double> M5(half * half);
        std::vector<double> M6(half * half);
        std::vector<double> M7(half * half);

        std::vector<double> temp1(half * half);
        std::vector<double> temp2(half * half);

        // Divide matrices A and B into 4 submatrices each
        for (size_t i = 0; i < half; ++i) {
            for (size_t j = 0; j < half; ++j) {
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

        // Compute intermediate matrices
        addMatrices(A11, A22, temp1, half);
        addMatrices(B11, B22, temp2, half);
        strassenMultiply(temp1, temp2, M1, half);  // M1 = (A11 + A22) * (B11 + B22)

        addMatrices(A21, A22, temp1, half);
        strassenMultiply(temp1, B11, M2, half);  // M2 = (A21 + A22) * B11

        subtractMatrices(B12, B22, temp2, half);
        strassenMultiply(A11, temp2, M3, half);  // M3 = A11 * (B12 - B22)

        subtractMatrices(B21, B11, temp2, half);
        strassenMultiply(A22, temp2, M4, half);  // M4 = A22 * (B21 - B11)

        addMatrices(A11, A12, temp1, half);
        strassenMultiply(temp1, B22, M5, half);  // M5 = (A11 + A12) * B22

        subtractMatrices(A21, A11, temp1, half);
        addMatrices(B11, B12, temp2, half);
        strassenMultiply(temp1, temp2, M6, half);  // M6 = (A21 - A11) * (B11 + B12)

        subtractMatrices(A12, A22, temp1, half);
        addMatrices(B21, B22, temp2, half);
        strassenMultiply(temp1, temp2, M7, half);  // M7 = (A12 - A22) * (B21 + B22)

        // Combine results to get C11, C12, C21, C22
        addMatrices(M1, M4, temp1, half);
        subtractMatrices(temp1, M5, temp2, half);
        addMatrices(temp2, M7, C11, half);  // C11 = M1 + M4 - M5 + M7

        addMatrices(M3, M5, C12, half);  // C12 = M3 + M5

        addMatrices(M2, M4, C21, half);  // C21 = M2 + M4

        addMatrices(M1, M3, temp1, half);
        subtractMatrices(temp1, M2, temp2, half);
        addMatrices(temp2, M6, C22, half);  // C22 = M1 + M3 - M2 + M6

        // Combine submatrices into the result matrix C
        for (size_t i = 0; i < half; ++i) {
            for (size_t j = 0; j < half; ++j) {
                C[i * n + j] = C11[i * half + j];
                C[i * n + j + half] = C12[i * half + j];
                C[(i + half) * n + j] = C21[i * half + j];
                C[(i + half) * n + j + half] = C22[i * half + j];
            }
        }
    }

     void addMatrices(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                C[i * n + j] = A[i * n + j] + B[i * n + j];
            }
        }
    }

     void subtractMatrices(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
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
        if (n <= 0 || (n & (n - 1)) != 0) {  // Ensure n is a power of 2
            return false;
        }

        return true;
    }

    bool StrassenAlgorithmSequential::run() {
        internal_order_test();
        strassenMultiply(A_, B_, C_, n);
        return true;
    }

    bool StrassenAlgorithmSequential::post_processing() {
        internal_order_test();
        for (size_t i = 0; i < n * n; ++i) {
            reinterpret_cast<double*>(taskData->outputs[0])[i] = C_[i];
        }
        return true;
    }

    void StrassenAlgorithmParallel::calculate_distribution(int len, int num_proc, std::vector<int>& sizes, std::vector<int>& displs) {
        sizes.resize(num_proc, 0);
        displs.resize(num_proc, -1);

        if (num_proc > len) {
            for (int i = 0; i < len; ++i) {
                sizes[i] = 1;
                displs[i] = i;
            }
        } else {
            int a = len / num_proc;
            int b = len % num_proc;

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

    void StrassenAlgorithmParallel::strassenMultiplyParallel(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
        if (n <= 64) {  // Base case: use standard matrix multiplication for small matrices
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

        size_t half = n / 2;

        std::vector<double> A11(half * half);
        std::vector<double> A12(half * half);
        std::vector<double> A21(half * half);
        std::vector<double> A22(half * half);

        std::vector<double> B11(half * half);
        std::vector<double> B12(half * half);
        std::vector<double> B21(half * half);
        std::vector<double> B22(half * half);

        std::vector<double> C11(half * half);
        std::vector<double> C12(half * half);
        std::vector<double> C21(half * half);
        std::vector<double> C22(half * half);

        std::vector<double> M1(half * half);
        std::vector<double> M2(half * half);
        std::vector<double> M3(half * half);
        std::vector<double> M4(half * half);
        std::vector<double> M5(half * half);
        std::vector<double> M6(half * half);
        std::vector<double> M7(half * half);

        std::vector<double> temp1(half * half);
        std::vector<double> temp2(half * half);

        // Divide matrices A and B into 4 submatrices each
        for (size_t i = 0; i < half; ++i) {
            for (size_t j = 0; j < half; ++j) {
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

        // Compute intermediate matrices
        addMatrices(A11, A22, temp1, half);
        addMatrices(B11, B22, temp2, half);
        strassenMultiplyParallel(temp1, temp2, M1, half);  // M1 = (A11 + A22) * (B11 + B22)

        addMatrices(A21, A22, temp1, half);
        strassenMultiplyParallel(temp1, B11, M2, half);  // M2 = (A21 + A22) * B11

        subtractMatrices(B12, B22, temp2, half);
        strassenMultiplyParallel(A11, temp2, M3, half);  // M3 = A11 * (B12 - B22)

        subtractMatrices(B21, B11, temp2, half);
        strassenMultiplyParallel(A22, temp2, M4, half);  // M4 = A22 * (B21 - B11)

        addMatrices(A11, A12, temp1, half);
        strassenMultiplyParallel(temp1, B22, M5, half);  // M5 = (A11 + A12) * B22

        subtractMatrices(A21, A11, temp1, half);
        addMatrices(B11, B12, temp2, half);
        strassenMultiplyParallel(temp1, temp2, M6, half);  // M6 = (A21 - A11) * (B11 + B12)

        subtractMatrices(A12, A22, temp1, half);
        addMatrices(B21, B22, temp2, half);
        strassenMultiplyParallel(temp1, temp2, M7, half);  // M7 = (A12 - A22) * (B21 + B22)

        // Combine results to get C11, C12, C21, C22
        addMatrices(M1, M4, temp1, half);
        subtractMatrices(temp1, M5, temp2, half);
        addMatrices(temp2, M7, C11, half);  // C11 = M1 + M4 - M5 + M7

        addMatrices(M3, M5, C12, half);  // C12 = M3 + M5

        addMatrices(M2, M4, C21, half);  // C21 = M2 + M4

        addMatrices(M1, M3, temp1, half);
        subtractMatrices(temp1, M2, temp2, half);
        addMatrices(temp2, M6, C22, half);  // C22 = M1 + M3 - M2 + M6

        // Combine submatrices into the result matrix C
        for (size_t i = 0; i < half; ++i) {
            for (size_t j = 0; j < half; ++j) {
                C[i * n + j] = C11[i * half + j];
                C[i * n + j + half] = C12[i * half + j];
                C[(i + half) * n + j] = C21[i * half + j];
                C[(i + half) * n + j + half] = C22[i * half + j];
            }
        }
    }

     void addMatrices(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                C[i * n + j] = A[i * n + j] + B[i * n + j];
            }
        }
    }

     void subtractMatrices(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                C[i * n + j] = A[i * n + j] - B[i * n + j];
            }
        }
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

            calculate_distribution(n * n, world.size(), sizes_a, displs_a);
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
            if (n <= 0 || (n & (n - 1)) != 0) {  // Ensure n is a power of 2
                return false;
            }
        }
        return true;
    }

    bool StrassenAlgorithmParallel::run() {
        internal_order_test();

        boost::mpi::broadcast(world, sizes_a, 0);
        boost::mpi::broadcast(world, displs_a, 0);
        boost::mpi::broadcast(world, n, 0);

        int loc_size = sizes_a[world.rank()];

        local_A.resize(loc_size);
        local_B.resize(loc_size);
        local_C.resize(loc_size);

        if (world.rank() == 0) {
            boost::mpi::scatterv(world, A_.data(), sizes_a, displs_a, local_A.data(), loc_size, 0);
            boost::mpi::scatterv(world, B_.data(), sizes_a, displs_a, local_B.data(), loc_size, 0);
        } else {
            boost::mpi::scatterv(world, local_A.data(), loc_size, 0);
            boost::mpi::scatterv(world, local_B.data(), loc_size, 0);
        }

        strassenMultiplyParallel(local_A, local_B, local_C, n);

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
            for (size_t i = 0; i < n * n; ++i) {
                reinterpret_cast<double*>(taskData->outputs[0])[i] = C_[i];
            }
        }
        return true;
    }

}  // namespace nasedkin_e_strassen_algorithm_mpi