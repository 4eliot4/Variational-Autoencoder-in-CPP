#include "conv.h"

#include <cmath>
#include <stdexcept>
#include <algorithm>



Conv2D::Conv2D(int Cin, int Cout, int K, int S, int P)
    : W(Cout, Cin, K, K),
      b(Eigen::VectorXf::Zero(Cout)),
      stride(S),
      padding(P)
{
    if (Cin <= 0 || Cout <= 0 || K <= 0 || S <= 0 || P < 0)
        throw std::invalid_argument("Invalid Conv2D parameters");

    const float scale = std::sqrt(2.0f / (Cin * K * K));    

    W.setRandom();

    for (int o = 0; o < Cout; ++o)
        for (int c = 0; c < Cin; ++c)
            for (int u = 0; u < K; ++u)
                for (int v = 0; v < K; ++v)
                    W(o, c, u, v) *= scale;
}


Tensor4D convForward(const Tensor4D& X, const Conv2D& layer)
{
    const int B    = X.dimension(0);
    const int Cin  = X.dimension(1);
    const int H    = X.dimension(2);
    const int Win  = X.dimension(3);

    const int Cout = layer.W.dimension(0);
    const int K    = layer.W.dimension(2);

    const int S = layer.stride;
    const int P = layer.padding;

    if (S <= 0 || P < 0 ||
        layer.W.dimension(1) != Cin ||
        layer.W.dimension(3) != K ||
        layer.b.size() != Cout ||
        H + 2 * P < K || Win + 2 * P < K)
    {
        throw std::invalid_argument("Invalid convolution dimensions");
    }

    const int Hout = (H + 2 * P - K) / S + 1;
    const int Wout = (Win + 2 * P - K) / S + 1;

    Tensor4D Y(B, Cout, Hout, Wout);

    for (int b = 0; b < B; ++b)
    {
        for (int o = 0; o < Cout; ++o)
        {
            for (int i = 0; i < Hout; ++i)
            {
                for (int j = 0; j < Wout; ++j)
                {
                    float sum = layer.b(o);

                    for (int c = 0; c < Cin; ++c)
                    {
                        for (int u = 0; u < K; ++u)
                        {
                            for (int v = 0; v < K; ++v)
                            {
                                const int h = i * S + u - P;
                                const int w = j * S + v - P;

                                if (h < 0 || h >= H || w < 0 || w >= Win)
                                    continue;

                                sum += X(b, c, h, w)* layer.W(o, c, u, v);
                            }
                        }
                    }

                    Y(b, o, i, j) = sum;
                }
            }
        }
    }

    return Y;
}

Tensor4D reluForward(const Tensor4D& X)
{
    Tensor4D Y(
        X.dimension(0),
        X.dimension(1),
        X.dimension(2),
        X.dimension(3)
    );

    for (int b = 0; b < X.dimension(0); ++b)
        for (int c = 0; c < X.dimension(1); ++c)
            for (int h = 0; h < X.dimension(2); ++h)
                for (int w = 0; w < X.dimension(3); ++w)
                    Y(b, c, h, w) =
                        std::max(0.0f, X(b, c, h, w));

    return Y;
}



ConvGradients convBackward(const Tensor4D& X,const Conv2D& layer,const Tensor4D& dY)
{
    const int B    = X.dimension(0);
    const int Cin  = X.dimension(1);
    const int H    = X.dimension(2);
    const int Win  = X.dimension(3);

    const int Cout = layer.W.dimension(0);
    const int K    = layer.W.dimension(2);

    const int S = layer.stride;
    const int P = layer.padding;

    if (S <= 0 || P < 0 ||
        layer.W.dimension(1) != Cin ||
        layer.W.dimension(3) != K ||
        layer.b.size() != Cout ||
        H + 2 * P < K || Win + 2 * P < K)
    {
        throw std::invalid_argument("Invalid convolution dimensions");
    }

    const int Hout = (H + 2 * P - K) / S + 1;
    const int Wout = (Win + 2 * P - K) / S + 1;

    if (dY.dimension(0) != B ||
        dY.dimension(1) != Cout ||
        dY.dimension(2) != Hout ||
        dY.dimension(3) != Wout)
    {
        throw std::invalid_argument("Invalid upstream gradient dimensions");
    }

    ConvGradients G{
        Tensor4D(Cout, Cin, K, K),
        Eigen::VectorXf::Zero(Cout),
        Tensor4D(B, Cin, H, Win)
    };

    G.dW.setZero();
    G.dX.setZero();

    for (int b = 0; b < B; ++b)
    {
        for (int o = 0; o < Cout; ++o)
        {
            for (int i = 0; i < Hout; ++i)
            {
                for (int j = 0; j < Wout; ++j)
                {
                    const float g = dY(b, o, i, j);

                    G.db(o) += g;  // Gradient of the bias

                    for (int c = 0; c < Cin; ++c)
                    {
                        for (int u = 0; u < K; ++u)
                        {
                            for (int v = 0; v < K; ++v)
                            {
                                const int h = i * S + u - P;
                                const int w = j * S + v - P;

                                if (h < 0 || h >= H ||
                                    w < 0 || w >= Win)
                                    continue;

                                
                                G.dW(o, c, u, v) += X(b, c, h, w) * g;// Gradient of kernel weights

                                
                                G.dX(b, c, h, w) += layer.W(o, c, u, v) * g; // Gradient of input
                            }
                        }
                    }
                }
            }
        }
    }

    return G;
}


Tensor4D reluBackward(const Tensor4D& A,const Tensor4D& dH)
{
    for (int d = 0; d < 4; ++d)
    {
        if (A.dimension(d) != dH.dimension(d))
            throw std::invalid_argument("ReLU gradient shape mismatch");
    }

    Tensor4D dA(
        A.dimension(0),
        A.dimension(1),
        A.dimension(2),
        A.dimension(3)
    );

    for (int b = 0; b < A.dimension(0); ++b)
        for (int c = 0; c < A.dimension(1); ++c)
            for (int h = 0; h < A.dimension(2); ++h)
                for (int w = 0; w < A.dimension(3); ++w)
                {
                    dA(b,c,h,w) =
                        A(b,c,h,w) > 0.0f
                        ? dH(b,c,h,w)
                        : 0.0f;
                }

    return dA;
}
