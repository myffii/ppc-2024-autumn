#include <gtest/gtest.h>

#include "seq/nasedkin_e_strassen_algorithm/src/ops_seq.cpp"

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_2x2) {
    int size = 2;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq strassen_task(size);

    std::vector<std::vector<double>> matrixA;
    std::vector<std::vector<double>> matrixB;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(size, matrixA);
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(size, matrixB);
    strassen_task.set_matrices(matrixA, matrixB);

    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_4x4) {
    int size = 4;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq strassen_task(size);

    std::vector<std::vector<double>> matrixA;
    std::vector<std::vector<double>> matrixB;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(size, matrixA);
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(size, matrixB);
    strassen_task.set_matrices(matrixA, matrixB);

    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_8x8) {
    int size = 8;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq strassen_task(size);

    std::vector<std::vector<double>> matrixA;
    std::vector<std::vector<double>> matrixB;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(size, matrixA);
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(size, matrixB);
    strassen_task.set_matrices(matrixA, matrixB);

    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_64x64) {
    int size = 64;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq strassen_task(size);

    std::vector<std::vector<double>> matrixA;
    std::vector<std::vector<double>> matrixB;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(size, matrixA);
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(size, matrixB);
    strassen_task.set_matrices(matrixA, matrixB);

    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_128x128) {
    int size = 128;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq strassen_task(size);

    std::vector<std::vector<double>> matrixA;
    std::vector<std::vector<double>> matrixB;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(size, matrixA);
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(size, matrixB);
    strassen_task.set_matrices(matrixA, matrixB);

    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_256x256) {
    int size = 256;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq strassen_task(size);

    std::vector<std::vector<double>> matrixA;
    std::vector<std::vector<double>> matrixB;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(size, matrixA);
    nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq::generate_random_matrix(size, matrixB);
    strassen_task.set_matrices(matrixA, matrixB);

    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}