#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>

static int H = 32;
static int D = 256;

struct Weights
{
    // Eigen::MatrixXd W1(256,H); its not gonna call the constructor
    Eigen::MatrixXd W1;
    Eigen::MatrixXd b1;
    Eigen::MatrixXd W2;
    Eigen::MatrixXd b2;
    Weights() : W1(D, H), b1(1,H),W2(H,D),b2(1,256)  {};
};

int main()
{
    
    return 0;
}
