#include <Eigen/Dense>
#include <random>

Eigen::MatrixXf make_batch_mnist(int batch_size,
                                 std::mt19937 &rng,
                                 bool use_train);

bool write_png_grid_mnist(const Eigen::MatrixXf &batch,
                          int gridCols,
                          int gridRows,
                          const std::string &outPath);