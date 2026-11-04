#ifndef TENSOR_H
#define TENSOR_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using Tensor4D = Eigen::Tensor<float, 4, Eigen::RowMajor>;
Tensor4D matrixToTensor(const Eigen::MatrixXf& X);

Eigen::MatrixXf flatten(const Tensor4D& X);

Tensor4D unflatten(const Eigen::MatrixXf& X,int channels,int height,int width);


#endif // TENSOR_H