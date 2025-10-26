#include "network.h"
#include <cmath>
#include <stdexcept>

// ====== SETTINGS ======
int H_size = 128;
int D = 256;
int B = 50;
double lr = 0.01f;

Weights::Weights() : W1(Eigen::MatrixXf::Random(D,H_size)*0.05),
                     b1(Eigen::MatrixXf::Zero(1,H_size)),
                     W2(Eigen::MatrixXf::Random(H_size,H_size)*0.05),
                     b2(Eigen::MatrixXf::Zero(1,H_size)),
                     W3(Eigen::MatrixXf::Random(H_size,D)*0.05),
                     b3(Eigen::MatrixXf::Zero(1,D))
                     {}
/**
* @brief Print first 5X5 matrixes of W1 & W2 and print b1 & b2.
* @brief Throw exeption if too small
*/
void Weights::print()
{
    try {
        if (W1.rows() < 5 || W1.cols() < 5 ||
            W2.rows() < 5 || W2.cols() < 5) {
            throw std::runtime_error("Matrix too small to print 5x5 block.");
        }

        // If no exception, print submatrices
        std::cout << "small W1 :\n" << W1.block(0, 0, 5, 5) << std::endl;
        std::cout << "small b1 :\n" << b1 << std::endl;
        std::cout << "small W2 :\n" << W2.block(0, 0, 5, 5) << std::endl;
        std::cout << "small b2 :\n" << b2 << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Caught an exception: " << e.what() << std::endl;
    }
}

ForwardOutput::ForwardOutput() : Z(B, H_size),
                                 H(B, H_size), 
                                 Z2(B,H_size),
                                 A2(B,H_size),
                                 Yhat(B, D),
                                 sigmoid(B,D) 
                                 {}
void ForwardOutput::lossPrint()
{
    std::cout << "The loss is : " << this->loss << std::endl;
}

Gradients::Gradients() : Gy(B, D), 
                         Gw3(H_size,D),
                         Ga2(B,H_size),
                         Gz2(B,H_size),
                         Gw2(H_size, H_size), 
                         Gh(B, H_size), 
                         Gz(B, H_size), 
                         Gw1(D, H_size),
                         Gb3(1,D),
                         Gb2(1, H_size), 
                         Gb1(1, H_size)
                         {}


/**
 * @brief Performs the forward pass through the network.
 * @param forward REFERENCE : Struct containing intermediate results (Z, H, Yhat, sigmoid, loss).
 * @param weights const : Current model weights and biases.
 * @param X const : Input batch matrix of shape (B, D).
 */
void forwardPass(ForwardOutput& forward,const Weights& weights, const Eigen::MatrixXf& X)
{
    forward.Z = X * weights.W1;
    forward.Z.rowwise() += weights.b1;
    forward.H = forward.Z.array().tanh(); // element wise
    forward.Z2 = forward.H * weights.W2;
    forward.A2 = forward.Z2.array().tanh();
    forward.Yhat = forward.A2 * weights.W3;
    forward.Yhat.rowwise() += weights.b3;

    forward.sigmoid = 1.0 / (1.0 + (-forward.Yhat.array()).exp()); // sigmoid element wise
    Eigen::MatrixXf loss_per_entry = -(X.array() * forward.sigmoid.array().log() // every element compute -xlog(...)
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
void backPass(Gradients& gradients, const ForwardOutput& forward, const Weights& weights,const Eigen::MatrixXf& X)
{
    gradients.Gy = (forward.sigmoid - X) / (B * D);
    gradients.Gw3 = forward.A2.transpose() * gradients.Gy;
    gradients.Gb3 = gradients.Gy.colwise().sum();
    gradients.Ga2 = gradients.Gy * weights.W3.transpose();
    gradients.Gz2 = gradients.Ga2.array() * (1 - forward.A2.array() * forward.A2.array());
    gradients.Gw2 = forward.H.transpose() * gradients.Gz2;
    gradients.Gb2 = gradients.Gz2.colwise().sum();
    gradients.Gh = gradients.Gz2 * weights.W2.transpose();
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
    weights.W3 -= lr * gradients.Gw3;
    weights.b3 -= lr * gradients.Gb3;
}

void training()
{

}