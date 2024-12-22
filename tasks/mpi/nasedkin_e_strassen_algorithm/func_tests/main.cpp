std::vector<double> generate_random_matrix(size_t size) {
    std::vector<double> matrix(size * size);
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < size * size; ++i) {
        matrix[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }
    return matrix;
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_2x2) {
auto taskData = std::make_shared<ppc::core::TaskData>();
taskData->inputs.push_back(generate_random_matrix(2));
taskData->inputs.push_back(generate_random_matrix(2));

nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_4x4) {
auto taskData = std::make_shared<ppc::core::TaskData>();
taskData->inputs.push_back(generate_random_matrix(4));
taskData->inputs.push_back(generate_random_matrix(4));

nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_8x8) {
auto taskData = std::make_shared<ppc::core::TaskData>();
taskData->inputs.push_back(generate_random_matrix(8));
taskData->inputs.push_back(generate_random_matrix(8));

nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_16x16) {
auto taskData = std::make_shared<ppc::core::TaskData>();
taskData->inputs.push_back(generate_random_matrix(16));
taskData->inputs.push_back(generate_random_matrix(16));

nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}