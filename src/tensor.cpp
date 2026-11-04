#include "tensor.h"
#include <stdexcept>


/**
* @brief Converts MNIST input image to 4D tensor
* @param X Input image in Matrix format
*/

Tensor4D matrixToTensor(
    const Eigen::MatrixXf& X,
    int C,
    int H,
    int W)
{
    if (C <= 0 || H <= 0 || W <= 0 || X.cols() != C * H * W)
        throw std::invalid_argument("Invalid matrix-to-tensor dimensions");

    const int batch = X.rows();

    Tensor4D T(batch, C, H, W);

    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                {
                    const int col = (c * H + i) * W + j;
                    T(b, c, i, j) = X(b, col);
                }

    return T;
}




Eigen::MatrixXf flatten(const Tensor4D& X)
{
    const int B = X.dimension(0);
    const int C = X.dimension(1);
    const int H = X.dimension(2);
    const int W = X.dimension(3);

    Eigen::MatrixXf result(B, C * H * W);

    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    result(b, (c * H + i) * W + j)
                        = X(b, c, i, j);

    return result;
}

Tensor4D unflatten(const Eigen::MatrixXf& X,int C, int H, int W)
{
    if (X.cols() != C * H * W)
        throw std::invalid_argument("Invalid unflatten dimensions");

    Tensor4D result(X.rows(), C, H, W);

    for (int b = 0; b < X.rows(); ++b)
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    result(b, c, i, j)
                        = X(b, (c * H + i) * W + j);

    return result;
}
