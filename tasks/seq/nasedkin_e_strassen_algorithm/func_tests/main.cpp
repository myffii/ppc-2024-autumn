#include <gtest/gtest.h>

#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"
#include "seq/nasedkin_e_strassen_algorithm/src/ops_seq.cpp"

TEST(StrassenAlgorithmSeq, test_random_matrix_2x2) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(2);

  nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq task(taskData);

  std::vector<std::vector<double>> A, B;
  task.generate_random_matrix(2, A);
  task.generate_random_matrix(2, B);
  task.set_matrices(A, B);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(StrassenAlgorithmSeq, test_random_matrix_4x4) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(4);

  nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq task(taskData);

  std::vector<std::vector<double>> A, B;
  task.generate_random_matrix(4, A);
  task.generate_random_matrix(4, B);
  task.set_matrices(A, B);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(StrassenAlgorithmSeq, test_random_matrix_8x8) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(8);

  nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq task(taskData);

  std::vector<std::vector<double>> A, B;
  task.generate_random_matrix(8, A);
  task.generate_random_matrix(8, B);
  task.set_matrices(A, B);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(StrassenAlgorithmSeq, test_random_matrix_64x64) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(64);

  nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq task(taskData);

  std::vector<std::vector<double>> A, B;
  task.generate_random_matrix(64, A);
  task.generate_random_matrix(64, B);
  task.set_matrices(A, B);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(StrassenAlgorithmSeq, test_random_matrix_128x128) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(128);

  nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq task(taskData);

  std::vector<std::vector<double>> A, B;
  task.generate_random_matrix(128, A);
  task.generate_random_matrix(128, B);
  task.set_matrices(A, B);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(StrassenAlgorithmSeq, test_random_matrix_256x256) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(256);

  nasedkin_e_strassen_algorithm::StrassenAlgorithmSeq task(taskData);

  std::vector<std::vector<double>> A, B;
  task.generate_random_matrix(256, A);
  task.generate_random_matrix(256, B);
  task.set_matrices(A, B);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}
