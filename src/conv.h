#ifndef CONV_H
#define CONV_H
#include "tensor.h"
#include <Eigen/Dense>



struct Conv2D
{
    Tensor4D W;        // (Cout, Cin, K, K)
    Eigen::VectorXf b; // (Cout)

    int stride;
    int padding;

    Conv2D(int Cin, int Cout, int K, int S, int P);
};

Tensor4D convForward(const Tensor4D& X, const Conv2D& layer);



#endif