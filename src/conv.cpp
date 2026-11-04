#include "conv.h"

#include <cmath>
#include <stdexcept>
#include <algorithm>


using DeconvRowMatrix = Eigen::Matrix<
    float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
>;

// (B,Cin,H,W) -> (B*H*W,Cin)
static DeconvRowMatrix deconvInputToMatrix(const Tensor4D& X)
{
    const int B = X.dimension(0);
    const int Cin = X.dimension(1);
    const int H = X.dimension(2);
    const int W = X.dimension(3);

    DeconvRowMatrix A(B * H * W, Cin);

    for (int b = 0; b < B; ++b)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
            {
                const int row = (b * H + i) * W + j;

                for (int c = 0; c < Cin; ++c)
                    A(row, c) = X(b, c, i, j);
            }

    return A;
}

// (Cout,Cin,K,K) -> (Cin,Cout*K*K)
static Eigen::MatrixXf deconvWeightsToMatrix(const ConvTranspose2D& layer)
{
    const int Cout = layer.W.dimension(0);
    const int Cin = layer.W.dimension(1);
    const int K = layer.W.dimension(2);

    Eigen::MatrixXf Wmat(Cin, Cout * K * K);

    for (int c = 0; c < Cin; ++c)
        for (int o = 0; o < Cout; ++o)
            for (int u = 0; u < K; ++u)
                for (int v = 0; v < K; ++v)
                {
                    const int col = (o * K + u) * K + v;
                    Wmat(c, col) = layer.W(o, c, u, v);
                }

    return Wmat;
}


Conv2D::Conv2D(int Cin, int Cout, int K, int S, int P)
    : W(Cout, Cin, K, K),
      b(Eigen::VectorXf::Zero(Cout)),
      stride(S),
      padding(P)
{
    if (Cin <= 0 || Cout <= 0 || K <= 0 || S <= 0 || P < 0)
        throw std::invalid_argument("Invalid Conv2D parameters");

    
    const float scale = std::sqrt(6.0f / float(Cin * K * K));

    W.setRandom();

    for (Eigen::Index i = 0; i < W.size(); ++i)
    {
        W.data()[i] =
            (2.0f * W.data()[i] - 1.0f) * scale;
    }

}

ConvTranspose2D::ConvTranspose2D(int Cin, int Cout, int K, int S, int P)
    : W(Cout, Cin, K, K),
      b(Eigen::VectorXf::Zero(Cout)),
      stride(S),
      padding(P)
{
    if (Cin <= 0 || Cout <= 0 || K <= 0 ||
        S <= 0 || P < 0)
    {
        throw std::invalid_argument(
            "Invalid transposed convolution parameters"
        );
    }

    W.setRandom();

    // Variance-scaled uniform initialization
    const float scale =
        std::sqrt(6.0f / float(Cin * K * K));

    for (Eigen::Index i = 0; i < W.size(); ++i)
        W.data()[i] = (2.0f * W.data()[i] - 1.0f) * scale;
}



Tensor4D convTransposeForward(const Tensor4D& X,const ConvTranspose2D& layer)
{
    const int B = X.dimension(0);
    const int Cin = X.dimension(1);
    const int H = X.dimension(2);
    const int Win = X.dimension(3);

    const int Cout = layer.W.dimension(0);
    const int K = layer.W.dimension(2);
    const int S = layer.stride;
    const int P = layer.padding;

    if (S <= 0 || P < 0 || layer.W.dimension(1) != Cin || layer.W.dimension(3) != K || layer.b.size() != Cout)
        throw std::invalid_argument("Invalid transposed convolution dimensions");

    const int Hout = (H - 1) * S - 2 * P + K;
    const int Wout = (Win - 1) * S - 2 * P + K;

    if (Hout <= 0 || Wout <= 0)
        throw std::invalid_argument("Invalid output size");

    const int N = B * H * Win;
    const int F = Cout * K * K;

    // 1. Convert inputs and filters to matrices
    DeconvRowMatrix Xmat = deconvInputToMatrix(X);
    Eigen::MatrixXf Wmat = deconvWeightsToMatrix(layer);

    // 2. Perform the expensive computation with Eigen
    // (N,Cin) * (Cin,F) -> (N,F)
    DeconvRowMatrix patches(N, F);
    patches.noalias() = Xmat * Wmat;

    // 3. Initialize output with biases
    Tensor4D Y(B, Cout, Hout, Wout);

    for (int b = 0; b < B; ++b)
        for (int o = 0; o < Cout; ++o)
            for (int h = 0; h < Hout; ++h)
                for (int w = 0; w < Wout; ++w)
                    Y(b, o, h, w) = layer.b(o);

    // 4. Scatter patch contributions into the output
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < Win; ++j)
            {
                const int row = (b * H + i) * Win + j;

                for (int o = 0; o < Cout; ++o)
                    for (int u = 0; u < K; ++u)
                        for (int v = 0; v < K; ++v)
                        {
                            const int h = i * S + u - P;
                            const int w = j * S + v - P;

                            if (h < 0 || h >= Hout || w < 0 || w >= Wout)
                                continue;

                            const int col = (o * K + u) * K + v;

                            Y(b, o, h, w) += patches(row, col);
                        }
            }

    return Y;
}




ConvGradients convTransposeBackward(const Tensor4D& X,const ConvTranspose2D& layer,const Tensor4D& dY)
{
    const int B = X.dimension(0);
    const int Cin = X.dimension(1);
    const int H = X.dimension(2);
    const int Win = X.dimension(3);

    const int Cout = layer.W.dimension(0);
    const int K = layer.W.dimension(2);
    const int S = layer.stride;
    const int P = layer.padding;

    if (S <= 0 || P < 0 || layer.W.dimension(1) != Cin || layer.W.dimension(3) != K || layer.b.size() != Cout)
        throw std::invalid_argument("Invalid transposed convolution dimensions");

    const int Hout = (H - 1) * S - 2 * P + K;
    const int Wout = (Win - 1) * S - 2 * P + K;

    if (dY.dimension(0) != B || dY.dimension(1) != Cout || dY.dimension(2) != Hout || dY.dimension(3) != Wout)
        throw std::invalid_argument("Wrong transposed convolution gradient shape");

    const int N = B * H * Win;
    const int F = Cout * K * K;

    DeconvRowMatrix Xmat = deconvInputToMatrix(X);
    Eigen::MatrixXf Wmat = deconvWeightsToMatrix(layer);

    // (N,F)
    DeconvRowMatrix dP(N, F);
    dP.setZero();

    for (int b = 0; b < B; ++b)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < Win; ++j)
            {
                const int row = (b * H + i) * Win + j;

                for (int o = 0; o < Cout; ++o)
                    for (int u = 0; u < K; ++u)
                        for (int v = 0; v < K; ++v)
                        {
                            const int h = i * S + u - P;
                            const int w = j * S + v - P;

                            if (h < 0 || h >= Hout || w < 0 || w >= Wout)
                                continue;

                            const int col = (o * K + u) * K + v;

                            dP(row, col) = dY(b, o, h, w);
                        }
            }


    Eigen::MatrixXf dWmat(Cin, F);
    dWmat.noalias() = Xmat.transpose() * dP;
    DeconvRowMatrix dXmat(N, Cin);
    dXmat.noalias() = dP * Wmat.transpose();

    ConvGradients G{
        Tensor4D(Cout, Cin, K, K),
        Eigen::VectorXf::Zero(Cout),
        Tensor4D(B, Cin, H, Win)
    };

    for (int o = 0; o < Cout; ++o)
        for (int c = 0; c < Cin; ++c)
            for (int u = 0; u < K; ++u)
                for (int v = 0; v < K; ++v)
                {
                    const int col = (o * K + u) * K + v;
                    G.dW(o, c, u, v) = dWmat(c, col);
                }

    for (int b = 0; b < B; ++b)
        for (int c = 0; c < Cin; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < Win; ++j)
                {
                    const int row = (b * H + i) * Win + j;
                    G.dX(b, c, i, j) = dXmat(row, c);
                }

    for (int b = 0; b < B; ++b)
        for (int o = 0; o < Cout; ++o)
            for (int h = 0; h < Hout; ++h)
                for (int w = 0; w < Wout; ++w)
                    G.db(o) += dY(b, o, h, w);

    return G;
}



static Eigen::MatrixXf im2col(const Tensor4D& X,int K, int S, int P)
{
    const int B   = X.dimension(0);
    const int Cin = X.dimension(1);
    const int H   = X.dimension(2);
    const int W   = X.dimension(3);

    const int Hout = (H + 2 * P - K) / S + 1;
    const int Wout = (W + 2 * P - K) / S + 1;

    Eigen::MatrixXf cols(
        B * Hout * Wout,
        Cin * K * K
    );

    for (int b = 0; b < B; ++b)
        for (int i = 0; i < Hout; ++i)
            for (int j = 0; j < Wout; ++j)
            {
                int row = (b * Hout + i) * Wout + j;

                for (int c = 0; c < Cin; ++c)
                    for (int u = 0; u < K; ++u)
                        for (int v = 0; v < K; ++v)
                        {
                            int h = i * S + u - P;
                            int w = j * S + v - P;

                            int col = (c * K + u) * K + v;

                            cols(row, col) =
                                (h >= 0 && h < H &&
                                 w >= 0 && w < W)
                                ? X(b, c, h, w)
                                : 0.0f;
                        }
            }

    return cols;
}

static Eigen::MatrixXf kernelsToMatrix(const Conv2D& layer)
{
    int Cout = layer.W.dimension(0);
    int Cin  = layer.W.dimension(1);
    int K    = layer.W.dimension(2);

    Eigen::MatrixXf Wmat(Cin * K * K, Cout);

    for (int o = 0; o < Cout; ++o)
        for (int c = 0; c < Cin; ++c)
            for (int u = 0; u < K; ++u)
                for (int v = 0; v < K; ++v)
                {
                    int row = (c * K + u) * K + v;
                    Wmat(row, o) = layer.W(o, c, u, v);
                }

    return Wmat;
}



Tensor4D convForward(const Tensor4D& X,const Conv2D& layer)
{
    const int B = X.dimension(0);
    const int Cin = X.dimension(1);
    const int H = X.dimension(2);
    const int Win = X.dimension(3);

    const int Cout = layer.W.dimension(0);
    const int K = layer.W.dimension(2);

    const int S = layer.stride;
    const int P = layer.padding;

    if (S <= 0 || P < 0 || layer.W.dimension(1) != Cin || layer.W.dimension(3) != K || layer.b.size() != Cout || H + 2 * P < K || Win + 2 * P < K)
        throw std::invalid_argument("Invalid convolution dimensions");

    const int Hout = (H + 2 * P - K) / S + 1;
    const int Wout = (Win + 2 * P - K) / S + 1;

    Eigen::MatrixXf patches = im2col(X, K, S, P);
    Eigen::MatrixXf Wmat = kernelsToMatrix(layer);

    // All convolutions computed using one matrix multiplication
    Eigen::MatrixXf Ymat = patches * Wmat;

    Ymat.rowwise() += layer.b.transpose();

    // Return to tensor format (B,Cout,Hout,Wout)
    Tensor4D Y(B, Cout, Hout, Wout);

    for (int b = 0; b < B; ++b)
        for (int i = 0; i < Hout; ++i)
            for (int j = 0; j < Wout; ++j)
            {
                int row = (b * Hout + i) * Wout + j;

                for (int o = 0; o < Cout; ++o)
                    Y(b, o, i, j) = Ymat(row, o);
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
        throw std::invalid_argument("Invalid convolution dimensions");

    const int Hout = (H + 2 * P - K) / S + 1;
    const int Wout = (Win + 2 * P - K) / S + 1;

    if (dY.dimension(0) != B ||
        dY.dimension(1) != Cout ||
        dY.dimension(2) != Hout ||
        dY.dimension(3) != Wout)
        throw std::invalid_argument("Invalid dY dimensions");

    const int N = B * Hout * Wout;

    // Same matrices as in forward propagation
    Eigen::MatrixXf patches = im2col(X, K, S, P);
    Eigen::MatrixXf Wmat = kernelsToMatrix(layer);

    // Convert incoming gradient from tensor to matrix
    Eigen::MatrixXf dYmat(N, Cout);

    for (int b = 0; b < B; ++b)
        for (int i = 0; i < Hout; ++i)
            for (int j = 0; j < Wout; ++j)
            {
                int row = (b * Hout + i) * Wout + j;

                for (int o = 0; o < Cout; ++o)
                    dYmat(row, o) = dY(b, o, i, j);
            }

    // Matrix-based gradients
    Eigen::MatrixXf dWmat = patches.transpose() * dYmat;
    Eigen::MatrixXf dPatches = dYmat * Wmat.transpose();

    ConvGradients G{
        Tensor4D(Cout, Cin, K, K),
        dYmat.colwise().sum().transpose(),
        Tensor4D(B, Cin, H, Win)
    };

    G.dX.setZero();

    // Convert weight gradients back to tensor format
    for (int o = 0; o < Cout; ++o)
        for (int c = 0; c < Cin; ++c)
            for (int u = 0; u < K; ++u)
                for (int v = 0; v < K; ++v)
                {
                    int row = (c * K + u) * K + v;
                    G.dW(o, c, u, v) = dWmat(row, o);
                }

    // Scatter input gradients back to spatial coordinates
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < Hout; ++i)
            for (int j = 0; j < Wout; ++j)
            {
                int row = (b * Hout + i) * Wout + j;

                for (int c = 0; c < Cin; ++c)
                    for (int u = 0; u < K; ++u)
                        for (int v = 0; v < K; ++v)
                        {
                            int h = i * S + u - P;
                            int w = j * S + v - P;

                            if (h < 0 || h >= H ||
                                w < 0 || w >= Win)
                                continue;

                            int col = (c * K + u) * K + v;

                            G.dX(b, c, h, w) +=
                                dPatches(row, col);
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
