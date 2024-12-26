#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

std::vector<double> generateRandomMatrix(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0);
  std::vector<double> matrix(size * size);

  if (size <= 0) {
    std::cout << "generateRandomMatrix: Invalid size: " << size << std::endl;
    return matrix;
  }

  for (int i = 0; i < size * size; i++) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

TEST(nasedkin_e_strassen_algorithm_seq, Test_InvalidInputSize) {
int matrixSizeA = 32;
int matrixSizeB = 64;  // Неверный размер второй матрицы

std::vector<double> matrixA = generateRandomMatrix(matrixSizeA);
std::vector<double> matrixB = generateRandomMatrix(matrixSizeB);
std::vector<double> resultSeq(matrixSizeA * matrixSizeA, 0.0);

std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
taskDataSeq->inputs_count.emplace_back(matrixA.size());
taskDataSeq->inputs_count.emplace_back(matrixB.size());
taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
taskDataSeq->outputs_count.emplace_back(resultSeq.size());

nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testSeqTask(taskDataSeq);
ASSERT_FALSE(testSeqTask.validation());
}

TEST(nasedkin_e_strassen_algorithm_seq, Test_IdentityMatrix) {
int matrixSize = 16;
std::vector<double> matrixA(matrixSize * matrixSize, 0.0);
std::vector<double> matrixB(matrixSize * matrixSize, 0.0);
std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

// Генерация единичной матрицы
for (int i = 0; i < matrixSize; ++i) {
matrixA[i * matrixSize + i] = 1.0;
matrixB[i * matrixSize + i] = 1.0;
}

std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
taskDataSeq->inputs_count.emplace_back(matrixA.size());
taskDataSeq->inputs_count.emplace_back(matrixB.size());
taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
taskDataSeq->outputs_count.emplace_back(resultSeq.size());

nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testSeqTask(taskDataSeq);
ASSERT_TRUE(testSeqTask.validation());
ASSERT_TRUE(testSeqTask.pre_processing());
ASSERT_TRUE(testSeqTask.run());
ASSERT_TRUE(testSeqTask.post_processing());

// Проверяем, что результат тоже является единичной матрицей
for (int i = 0; i < matrixSize; ++i) {
for (int j = 0; j < matrixSize; ++j) {
if (i == j) {
ASSERT_NEAR(resultSeq[i * matrixSize + j], 1.0, 1e-9);
} else {
ASSERT_NEAR(resultSeq[i * matrixSize + j], 0.0, 1e-9);
}
}
}
}


TEST(nasedkin_e_strassen_algorithm_seq, Test_2x2) {
  int matrixSize = 2;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA.size());
  taskDataSeq->inputs_count.emplace_back(matrixB.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resultSeq.size());

  std::cout << "Test: TaskData inputs_count[0] = " << taskDataSeq->inputs_count[0]
            << ", inputs_count[1] = " << taskDataSeq->inputs_count[1] << std::endl;

  nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testSeqTask(taskDataSeq);
  ASSERT_TRUE(testSeqTask.validation());
  ASSERT_TRUE(testSeqTask.pre_processing());
  ASSERT_TRUE(testSeqTask.run());
  ASSERT_TRUE(testSeqTask.post_processing());
  std::cout << "SEQ Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;
}

TEST(nasedkin_e_strassen_algorithm_seq, Test_4x4) {
  int matrixSize = 4;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA.size());
  taskDataSeq->inputs_count.emplace_back(matrixB.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resultSeq.size());

  std::cout << "Test: TaskData inputs_count[0] = " << taskDataSeq->inputs_count[0]
            << ", inputs_count[1] = " << taskDataSeq->inputs_count[1] << std::endl;

  nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testSeqTask(taskDataSeq);
  ASSERT_TRUE(testSeqTask.validation());
  ASSERT_TRUE(testSeqTask.pre_processing());
  ASSERT_TRUE(testSeqTask.run());
  ASSERT_TRUE(testSeqTask.post_processing());
  std::cout << "SEQ Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;
}

TEST(nasedkin_e_strassen_algorithm_seq, Test_8x8) {
  int matrixSize = 8;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA.size());
  taskDataSeq->inputs_count.emplace_back(matrixB.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resultSeq.size());

  std::cout << "Test: TaskData inputs_count[0] = " << taskDataSeq->inputs_count[0]
            << ", inputs_count[1] = " << taskDataSeq->inputs_count[1] << std::endl;

  nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testSeqTask(taskDataSeq);
  ASSERT_TRUE(testSeqTask.validation());
  ASSERT_TRUE(testSeqTask.pre_processing());
  ASSERT_TRUE(testSeqTask.run());
  ASSERT_TRUE(testSeqTask.post_processing());
  std::cout << "SEQ Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;
}

TEST(nasedkin_e_strassen_algorithm_seq, Test_16x16) {
  int matrixSize = 16;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA.size());
  taskDataSeq->inputs_count.emplace_back(matrixB.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resultSeq.size());

  std::cout << "Test: TaskData inputs_count[0] = " << taskDataSeq->inputs_count[0]
            << ", inputs_count[1] = " << taskDataSeq->inputs_count[1] << std::endl;

  nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testSeqTask(taskDataSeq);
  ASSERT_TRUE(testSeqTask.validation());
  ASSERT_TRUE(testSeqTask.pre_processing());
  ASSERT_TRUE(testSeqTask.run());
  ASSERT_TRUE(testSeqTask.post_processing());
  std::cout << "SEQ Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;
}

TEST(nasedkin_e_strassen_algorithm_seq, Test_32x32) {
  int matrixSize = 32;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA.size());
  taskDataSeq->inputs_count.emplace_back(matrixB.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resultSeq.size());

  std::cout << "Test: TaskData inputs_count[0] = " << taskDataSeq->inputs_count[0]
            << ", inputs_count[1] = " << taskDataSeq->inputs_count[1] << std::endl;

  nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testSeqTask(taskDataSeq);
  ASSERT_TRUE(testSeqTask.validation());
  ASSERT_TRUE(testSeqTask.pre_processing());
  ASSERT_TRUE(testSeqTask.run());
  ASSERT_TRUE(testSeqTask.post_processing());
  std::cout << "SEQ Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;
}