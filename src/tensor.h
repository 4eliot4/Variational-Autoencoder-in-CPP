#ifndef TENSOR_H
#define TENSOR_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using Tensor4D = Eigen::Tensor<float, 4, Eigen::RowMajor>;
Tensor4D matrixToTensor(const Eigen::MatrixXf& X);


#endif // TENSOR_H