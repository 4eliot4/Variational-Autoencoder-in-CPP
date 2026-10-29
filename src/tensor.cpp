#include "tensor.h"
#include <stdexcept>


/**
* @brief Converts MNIST input image to 4D tensor
* @param X Input image in Matrix format
*/
Tensor4D matrixToTensor(const Eigen::MatrixXf& X)
{
    const int B = X.rows();

    if (X.cols() != 784)
        throw std::runtime_error("Not Loading the MNIST dataset");

    Tensor4D result(B, 1, 28, 28); // MNIST contains 1 channel (grayscale)

    for (int b = 0; b < B; ++b)
    {
        for (int i = 0; i < 28; ++i)
        {
            for (int j = 0; j < 28; ++j)
            {
                result(b, 0, i, j) = X(b, i * 28 + j); // flat index i * 28 + j
            }
        }
    }

    return result;
}