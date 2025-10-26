#include "network.h"
#include <cmath>

// ====== SETTINGS ======
int H_size = 32;
int D = 256;
int B = 32;
double lr = 0.01f;

Weights::Weights() : W1(Eigen::MatrixXd::Random(D,H_size)*0.05),
                     b1(Eigen::MatrixXd::Zero(1,H_size)),
                     W2(Eigen::MatrixXd::Random(H_size,D)*0.05),
                     b2(Eigen::MatrixXd::Zero(1,D))
                     {}
void Weights::print()
{
    std::cout << "W1 : " << W1 << std::endl;
    std::cout << "b1 : " << b1 << std::endl;
    std::cout << "W2 : " << W2 << std::endl;
    std::cout << "b2 : " << b2 << std::endl;
}

ForwardOutput::ForwardOutput() : Z(B, H_size),
                                 H(B, H_size), 
                                 Yhat(B, D),
                                 sigmoid(B,D) 
                                 {}

Gradients::Gradients() : Gy(B, D), 
                         Gw2(H_size, D), 
                         Gh(B, H_size), 
                         Gz(B, H_size), 
                         Gw1(D, H_size),
                         Gb2(1, D), 
                         Gb1(1, H_size)
                         {}


/**
 * @brief Performs the forward pass through the network.
 * @param forward REFERENCE : Struct containing intermediate results (Z, H, Yhat, sigmoid, loss).
 * @param weights const : Current model weights and biases.
 * @param X const : Input batch matrix of shape (B, D).
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

/**
 * @brief Computes all gradients for backpropagation
 * @param gradients REF : Output struct to store all computed gradients.
 * @param forward const : Forward pass results.
 * @param weights  const : Current model weights.
 * @param X const : Input batch.
 */
void backPass(Gradients& gradients, const ForwardOutput& forward, const Weights& weights,const Eigen::MatrixXd& X)
{
    gradients.Gy = (forward.sigmoid - X) / (B * D);
    gradients.Gw2 = forward.H.transpose() * gradients.Gy;
    gradients.Gb2 = gradients.Gy.colwise().sum();
    gradients.Gh = gradients.Gy * weights.W2.transpose();
    gradients.Gz = gradients.Gh.array() * (1 - forward.H.array() * forward.H.array());
    gradients.Gw1 = X.transpose() * gradients.Gz;
    gradients.Gb1 = gradients.Gz.colwise().sum();
}

/**
 * @brief Updates the network weights using gradient descent.
 * @param weights REF : Model weights to update.
 * @param gradients const : Gradients computed from backPass().
 */
void backProp(Weights& weights,const Gradients& gradients)
{
    weights.W1 -= lr * gradients.Gw1;
    weights.b1 -= lr * gradients.Gb1;
    weights.W2 -= lr * gradients.Gw2;
    weights.b2 -= lr * gradients.Gb2;
}

void training()
{

}