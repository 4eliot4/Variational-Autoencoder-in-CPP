#include "conv.h"

#include <cmath>
#include <stdexcept>




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