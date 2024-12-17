#pragma once

#include <vector>
#include <memory>

namespace nasedkin_e_strassen_algorithm {

    class StrassenAlgorithmSeq {
    public:
        explicit StrassenAlgorithmSeq(int size) : n(size) {}

        bool pre_processing();
        bool validation();
        bool run();
        bool post_processing();

        void set_matrices(const std::vector<std::vector<double>>& matrixA, const std::vector<std::vector<double>>& matrixB);
        static void generate_random_matrix(int size, std::vector<std::vector<double>>& matrix);
        const std::vector<std::vector<double>>& get_result() const { return result; }

    private:
        std::vector<std::vector<double>> A;
        std::vector<std::vector<double>> B;
        std::vector<std::vector<double>> result;
        int n;

        void strassen_multiply(const std::vector<std::vector<double>>& local_A, const std::vector<std::vector<double>>& local_B, std::vector<std::vector<double>>& local_result, int size);
        static void add_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size);
        static void subtract_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int size);
        static void split_matrix(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& A11, std::vector<std::vector<double>>& A12, std::vector<std::vector<double>>& A21, std::vector<std::vector<double>>& A22, int size);
        static void join_matrices(const std::vector<std::vector<double>>& C11, const std::vector<std::vector<double>>& C12, const std::vector<std::vector<double>>& C21, const std::vector<std::vector<double>>& C22, std::vector<std::vector<double>>& C, int size);
    };

}  // namespace nasedkin_e_strassen_algorithm