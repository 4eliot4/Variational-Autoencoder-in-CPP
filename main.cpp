#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <cmath>

static int H_size = 32;
static int D = 256;
static int B = 32;
static double lr = 0.01f;  // learning rate


struct Weights
{
    // Eigen::MatrixXd W1(256,H); wrong: its not gonna call the constructor
    Eigen::MatrixXd W1;
    Eigen::RowVectorXd b1;
    Eigen::MatrixXd W2;
    Eigen::RowVectorXd b2;
    Weights() : W1(Eigen::MatrixXd::Random(D,H_size)*0.05),
                b1(Eigen::MatrixXd::Zero(1,H_size)),
                W2(Eigen::MatrixXd::Random(H_size,D)*0.05),
                b2(Eigen::MatrixXd::Zero(1,D))
                {};
    void print()
    {
        std::cout << "W1 : " << W1 << std::endl;
        std::cout << "b1 : " << b1 << std::endl;
        std::cout << "W2 : " << W2 << std::endl;
        std::cout << "b2 : " << b2 << std::endl;
    }
};
struct ForwardOutput
{
    Eigen::MatrixXd Z;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Yhat;
    Eigen::MatrixXd sigmoid; // sigmoid of Y
    double loss;
    ForwardOutput() : Z(B, H_size), H(B, H_size), Yhat(B, D),sigmoid(B,D) {};
};

struct Gradients
{
    Eigen::MatrixXd Gy, Gw2, Gh, Gz, Gw1;
    Eigen::RowVectorXd Gb2, Gb1;
    Gradients() : Gy(B, D), Gw2(H_size, D), Gh(B, H_size), Gz(B, H_size), Gw1(D, H_size),Gb2(1, D), Gb1(1, D){};
};

void forwardPass(ForwardOutput &forward, const Weights &weights, const Eigen::MatrixXd &X);
void backPass(Gradients &gradients, const ForwardOutput &forward, const Weights &weights, const Eigen::MatrixXd &X);



int main()
{
    Weights weights;

    
    return 0;
}


/* FORWARD PASS
reference on forward !
*/
void forwardPass(ForwardOutput& forward,const Weights& weights, const Eigen::MatrixXd& X)
{
    forward.Z = X * weights.W1;
    forward.Z.rowwise() += weights.b1;
    forward.H = forward.Z.array().tanh(); // element wise
    forward.Yhat = forward.H * weights.W2;
    forward.Yhat.rowwise() += weights.b2;

    forward.sigmoid = 1.0 / (1.0 + (-forward.Yhat.array()).exp()); // sigmoid element wise
    Eigen::MatrixXd loss_per_entry = -(X.array() * forward.sigmoid.array().log() // every element compute -xlog(...)
                                     + (1 - X.array()) * (1 - forward.sigmoid.array()).log());

    forward.loss = loss_per_entry.mean(); // mean over all entries in batch, mean over B & D !
}

void backPass(Gradients& gradients, const ForwardOutput& forward, const Weights& weights,const Eigen::MatrixXd& X)
{
    gradients.Gy = (forward.sigmoid - X) / (B * D);
    gradients.Gw2 = forward.H.transpose() * gradients.Gy;
    gradients.Gb2 = gradients.Gy.colwise().sum();
    gradients.Gh = gradients.Gy * weights.W2.transpose();
    gradients.Gz = gradients.Gh.array() * (1 - forward.H.array() * forward.H.array());
    gradients.Gw1 = X.transpose() * gradients.Gz;
    gradients.Gb1 = gradients.Gz.rowwise().sum();
}