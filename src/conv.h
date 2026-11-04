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

struct ConvTranspose2D
{
    Tensor4D W;        // (Cout, Cin, K, K)
    Eigen::VectorXf b; // (Cout)

    int stride;
    int padding;

    ConvTranspose2D(int Cin, int Cout, int K, int S, int P);
};

struct ConvGradients
{
    Tensor4D dW;
    Eigen::VectorXf db;
    Tensor4D dX;
};


Tensor4D convForward(const Tensor4D& X, const Conv2D& layer);
Tensor4D reluForward(const Tensor4D &X);
ConvGradients convBackward(const Tensor4D& X,const Conv2D& layer,const Tensor4D& dY);
Tensor4D reluBackward(const Tensor4D &A, const Tensor4D &dH);

Tensor4D convTransposeForward(const Tensor4D& X,const ConvTranspose2D& layer);

ConvGradients convTransposeBackward(const Tensor4D& X,const ConvTranspose2D& layer,const Tensor4D& dY);


#endif