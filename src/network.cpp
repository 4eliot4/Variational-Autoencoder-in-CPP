#include "network.h"
#include <cmath>
#include <stdexcept>
#include <iostream> // Required for std::cout, std::cerr

// ====== SETTINGS ======
// These global settings define the network's architecture and training parameters.
// For a more robust application, these might be moved to a configuration file or
// passed as parameters.
int H_size = 64; // Hidden layer size
int D = 256;     // Input/Output dimension (e.g., flattened image pixels)
int B = 50;      // Batch size for training
float lr = 0.01f; // Learning rate for gradient descent

// He initialization factors for ReLU activation functions.
// These scales are applied to randomly initialized weights to help
// prevent vanishing/exploding gradients in networks using ReLU.
const float he_scale_W1 = std::sqrt(2.0f / D);      // For W1: Input D, Output H_size
const float he_scale_W2 = std::sqrt(2.0f / H_size); // For W2: Input H_size, Output H_size
const float he_scale_W3 = std::sqrt(2.0f / H_size); // For W3: Input H_size, Output D

Weights::Weights() : W1(Eigen::MatrixXf::Random(D, H_size) * he_scale_W1), // He initialization for W1
                     b1(Eigen::MatrixXf::Zero(1, H_size)),
                     W2(Eigen::MatrixXf::Random(H_size, H_size) * he_scale_W2), // He initialization for W2
                     b2(Eigen::MatrixXf::Zero(1, H_size)),
                     W3(Eigen::MatrixXf::Random(H_size, D) * he_scale_W3), // He initialization for W3
                     b3(Eigen::MatrixXf::Zero(1, D)) {}

/**
* @brief Prints a 5x5 block of W1 & W2 and the full b1 & b2 matrices.
* @brief Throws an exception if matrices are too small to print a 5x5 block.
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
                                 Z2(B, H_size),
                                 A2(B, H_size),
                                 Yhat(B, D),
                                 sigmoid(B, D) {}

/**
 * @brief Prints the current loss value.
 */
void ForwardOutput::lossPrint()
{
    std::cout << "The loss is : " << this->loss << std::endl;
}

Gradients::Gradients() : Gy(B, D),
                         Gw3(H_size, D),
                         Ga2(B, H_size),
                         Gz2(B, H_size),
                         Gw2(H_size, H_size),
                         Gh(B, H_size),
                         Gz(B, H_size),
                         Gw1(D, H_size),
                         Gb3(1, D),
                         Gb2(1, H_size),
                         Gb1(1, H_size) {}


/**
 * @brief Performs the forward pass through the network.
 *        Hidden layers now use ReLU activation for better training dynamics.
 * @param forward REFERENCE : Struct containing intermediate results (Z, H, Yhat, sigmoid, loss).
 * @param weights const : Current model weights and biases.
 * @param X const : Input batch matrix of shape (B, D).
 */
void forwardPass(ForwardOutput& forward, const Weights& weights, const Eigen::MatrixXf& X)
{
    // First hidden layer
    forward.Z = X * weights.W1;
    forward.Z.rowwise() += weights.b1;
    forward.H = forward.Z.array().max(0.0f); // ReLU activation: max(0, Z)

    // Second hidden layer
    forward.Z2 = forward.H * weights.W2;
    forward.Z2.rowwise() += weights.b2;
    forward.A2 = forward.Z2.array().max(0.0f); // ReLU activation: max(0, Z2)

    // Output layer (linear transformation)
    forward.Yhat = forward.A2 * weights.W3;
    forward.Yhat.rowwise() += weights.b3;

    // Sigmoid activation for the output layer, suitable for binary classification (pixel values 0 or 1)
    forward.sigmoid = 1.0f / (1.0f + (-forward.Yhat.array()).exp()); // Sigmoid element-wise

    // Binary Cross-Entropy (BCE) Loss calculation
    // - (X * log(sigmoid) + (1 - X) * log(1 - sigmoid))
    Eigen::MatrixXf loss_per_entry = -(X.array() * forward.sigmoid.array().log()
                                     + (1.0f - X.array()) * (1.0f - forward.sigmoid.array()).log());

    forward.loss = loss_per_entry.mean(); // Mean loss over all entries in the batch (B * D)
}

/**
 * @brief Computes all gradients for backpropagation.
 *        Derivatives for ReLU activation functions are now correctly applied.
 * @param gradients REF : Output struct to store all computed gradients.
 * @param forward const : Forward pass results.
 * @param weights  const : Current model weights.
 * @param X const : Input batch.
 */
void backPass(Gradients& gradients, const ForwardOutput& forward, const Weights& weights, const Eigen::MatrixXf& X)
{
    // Gradient for the output layer's pre-activation (Yhat)
    // This combines the derivative of BCE loss with the derivative of Sigmoid.
    // The division by (B * D) scales the gradients to correspond to the mean loss.
    gradients.Gy = (forward.sigmoid - X) / (B * D);

    // Gradients for W3 and b3
    gradients.Gw3 = forward.A2.transpose() * gradients.Gy;
    gradients.Gb3 = gradients.Gy.colwise().sum();

    // Gradients for A2 (activation of the second hidden layer)
    gradients.Ga2 = gradients.Gy * weights.W3.transpose();

    // Gradients for Z2 (pre-activation of the second hidden layer)
    // Derivative of ReLU (d/dx max(0, x)) is 1 if x > 0, and 0 otherwise.
    gradients.Gz2 = gradients.Ga2.array() * (forward.Z2.array() > 0.0f).cast<float>();

    // Gradients for W2 and b2
    gradients.Gw2 = forward.H.transpose() * gradients.Gz2;
    gradients.Gb2 = gradients.Gz2.colwise().sum();

    // Gradients for H (activation of the first hidden layer)
    gradients.Gh = gradients.Gz2 * weights.W2.transpose();

    // Gradients for Z (pre-activation of the first hidden layer)
    // Derivative of ReLU (d/dx max(0, x)) is 1 if x > 0, and 0 otherwise.
    gradients.Gz = gradients.Gh.array() * (forward.Z.array() > 0.0f).cast<float>();

    // Gradients for W1 and b1
    gradients.Gw1 = X.transpose() * gradients.Gz;
    gradients.Gb1 = gradients.Gz.colwise().sum();
}

/**
 * @brief Updates the network weights using gradient descent.
 * @param weights REF : Model weights to update.
 * @param gradients const : Gradients computed from backPass().
 */
void backProp(Weights& weights, const Gradients& gradients)
{
    weights.W1 -= lr * gradients.Gw1;
    weights.b1 -= lr * gradients.Gb1;
    weights.W2 -= lr * gradients.Gw2;
    weights.b2 -= lr * gradients.Gb2;
    weights.W3 -= lr * gradients.Gw3;
    weights.b3 -= lr * gradients.Gb3;
}

/**
 * @brief Placeholder for the main training loop, which is typically
 *        orchestrated in `main.cpp`.
 */
void training()
{
    // This function remains empty as per the original file's structure.
}