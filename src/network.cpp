#include "network.h"
#include <cmath>
#include <stdexcept>
#include <fstream>


// ====== SETTINGS ======
int H_size = 64;
int D = 784;
int B = 64;
double lr = 0.0003f;

auto xavier = [](int fan_in, int fan_out){ return std::sqrt(2.0f / float(fan_in + fan_out)); };

Weights::Weights() : W1(Eigen::MatrixXf::Random(D,H_size) * xavier(D, H_size)),
                     b1(Eigen::MatrixXf::Zero(1,H_size)),
                     W2(Eigen::MatrixXf::Random(H_size,H_size) * xavier(H_size, H_size)),
                     b2(Eigen::MatrixXf::Zero(1,H_size)),
                     W3(Eigen::MatrixXf::Random(H_size,D) * xavier(H_size, D)),
                     b3(Eigen::MatrixXf::Zero(1,D)),
                     Wmu(Eigen::MatrixXf::Random(H_size,H_size) * xavier(H_size,H_size)),
                     bmu(Eigen::MatrixXf::Zero(1,H_size)),
                     Wv(Eigen::MatrixXf::Random(H_size,H_size) * xavier(H_size,H_size)),
                     bv(Eigen::MatrixXf::Zero(1,H_size))
                     {}
                     

static void adam_update(Eigen::MatrixXf& theta,
                        const Eigen::MatrixXf& grad,
                        Eigen::MatrixXf& m,
                        Eigen::MatrixXf& v,
                        int t,
                        float lr,
                        float beta1 = 0.9f, // default values
                        float beta2 = 0.999f,
                        float eps   = 1e-8f)
{
    // m = beta1*m + (1-beta1)*g
    m = beta1 * m + (1.0f - beta1) * grad;

    // v = beta2*v + (1-beta2)*g^2 (elementwise)
    v = beta2 * v + (1.0f - beta2) * grad.array().square().matrix();

    // bias correction
    const float b1t = 1.0f - std::pow(beta1, (float)t);
    const float b2t = 1.0f - std::pow(beta2, (float)t);

    Eigen::MatrixXf mhat = m / b1t;
    Eigen::MatrixXf vhat = v / b2t;

    // theta -= lr * mhat / (sqrt(vhat) + eps)
    theta.array() -= lr * mhat.array() / (vhat.array().sqrt() + eps);
}

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

ForwardOutput::ForwardOutput() : X(B, H_size),
                                 H(B, H_size),
                                 M( B,H_size),
                                 V(B,H_size), 
                                 E(B,H_size),
                                 Z(B,H_size),
                                 A2(B,H_size),
                                 H2(B,H_size),
                                 Yhat(B, D),
                                 sigmoid(B,D) 
                                 {}
void ForwardOutput::lossPrint()
{
    std::cout << "Total: " << loss
              << " | Reconstruction: " << lossRec
              << " | KL: " << lossKL
              << std::endl;
}

Gradients::Gradients() : Gy(B, D), 
                         Gw3(H_size,D),
                         Gh2(B,H_size),
                         Ga2(B,H_size),
                         Gm(B,H_size),
                         Gv(B,H_size),
                         Gwmu(H_size, H_size),   
                         Gwv(H_size, H_size),    
                         Gw2(H_size, H_size), 
                         Gh(B, H_size),
                         Ga1(B,H_size), 
                         Gz(B, H_size), 
                         Gw1(D, H_size),
                         Gb3(1,D),
                         Gb2(1, H_size), 
                         Gbv(1,H_size),
                         Gbmu(1,H_size),
                         Gb1(1, H_size)
                         {}

Eigen::MatrixXf Gy, Gw3,Gh2,Ga2,Gm,Gv,Gw2, Gh, Gz, Gw1;
    Eigen::MatrixXf Gb3,Gb2,Gbv,Gbmu,Gb1;
/**
 * @brief Performs the forward pass through the network.
 * @param forward REFERENCE : Struct containing intermediate results (Z, H, Yhat, sigmoid, loss).
 * @param weights const : Current model weights and biases.
 * @param X const : Input batch matrix of shape (B, D).
 */
void forwardPass(ForwardOutput& forward,const Weights& weights, const Eigen::MatrixXf& X,std::mt19937& rng)
{
    forward.X = X * weights.W1;
    forward.X.rowwise() += weights.b1.row(0); // same param b across all images
    forward.H = forward.X.array().cwiseMax(0.0); // element wise ReLU
    // VAE
    // Mean: M = H * Wmu + bmu
    forward.M = forward.H * weights.Wmu;
    forward.M.rowwise() += weights.bmu.row(0);
    // Log-variance: V = H * Wv + bv
    forward.V = forward.H * weights.Wv;
    forward.V.rowwise() += weights.bv.row(0);

    // SAMPLE
    std::normal_distribution<float> normal(0.0f, 1.0f);

    for (int i = 0; i < forward.E.rows(); ++i) {
        for (int j = 0; j < forward.E.cols(); ++j) {
            forward.E(i, j) = normal(rng);
        }
    }
    forward.Z = forward.M.array() + (0.5f * forward.V.array()).exp() * forward.E.array();

    // DECODER
    forward.A2 = forward.H * weights.W2;
    forward.A2.rowwise() += weights.b2.row(0);
    forward.H2 = forward.A2.array().cwiseMax(0.0);
    forward.Yhat = forward.H2 * weights.W3;
    forward.Yhat.rowwise() += weights.b3.row(0);

    forward.sigmoid = 1.0 / (1.0 + (-forward.Yhat.array()).exp()); // sigmoid element wise
    //Eigen::MatrixXf loss_per_entry = -(X.array() * forward.sigmoid.array().log() // every element compute -xlog(...)
                                  //   + (1 - X.array()) * (1 - forward.sigmoid.array()).log());
    
    const float eps = 1e-7f; // small safety
    Eigen::ArrayXXf s = forward.sigmoid.array().min(1.0f - eps).max(eps);

    Eigen::MatrixXf loss_per_entry = -(X.array() * s.log() + (1.0f - X.array()) * (1.0f - s).log()).matrix();
    forward.lossRec = loss_per_entry.sum() / B;
    forward.lossKL = 0.5f / B * (forward.M.array().square() + forward.V.array().exp() - 1.0f- forward.V.array()).sum();
    forward.loss = forward.lossRec + forward.lossKL;
}

/**
 * @brief Compute parameters from H only, used for interpolation for now
 */
void decoder(ForwardOutput& forward,const Weights& weights)
{
    forward.A2 = forward.H * weights.W2;
    forward.A2.rowwise() += weights.b2.row(0);
    forward.H2 = forward.A2.array().cwiseMax(0.0);
    forward.Yhat = forward.H2 * weights.W3;
    forward.Yhat.rowwise() += weights.b3.row(0);
    forward.sigmoid = 1.0 / (1.0 + (-forward.Yhat.array()).exp()); // sigmoid element wise
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
    gradients.Gy = (forward.sigmoid - X) / B;
    gradients.Gw3 = forward.H2.transpose() * gradients.Gy;
    gradients.Gb3 = gradients.Gy.colwise().sum();
    gradients.Gh2 = gradients.Gy * weights.W3.transpose();
    //gradients.Ga2 = gradients.Gh2.array() * (1 - forward.H2.array() * forward.H2.array()); //for tanh
    Eigen::MatrixXf relu2_mask = (forward.A2.array() > 0.0f).cast<float>(); // for ReLU
    gradients.Ga2 = gradients.Gh2.array() * relu2_mask.array();

    gradients.Gw2 = forward.Z.transpose() * gradients.Ga2;
    gradients.Gb2 = gradients.Ga2.colwise().sum();
    gradients.Gz = gradients.Ga2 * weights.W2.transpose();

    // VAEE, .array and * FOR ELEMENT WISE OPERATIONS
    gradients.Gm = gradients.Gz + forward.M / B;
    gradients.Gv = 0.5f * gradients.Gz.array() * (0.5f * forward.V.array()).exp() * forward.E.array() + (forward.V.array().exp() - 1.0f) / (2.0f * B);
    gradients.Gwmu = forward.H.transpose() * gradients.Gm;
    gradients.Gbmu = gradients.Gm.colwise().sum();
    gradients.Gwv = forward.H.transpose() * gradients.Gv;
    gradients.Gbv = gradients.Gv.colwise().sum();
    gradients.Gh = gradients.Gm * weights.Wmu.transpose() + gradients.Gv * weights.Wv.transpose();

    Eigen::MatrixXf relu1_mask = (forward.X.array() > 0.0f).cast<float>();
    gradients.Ga1 = gradients.Gh.array() * relu1_mask.array();
    
    gradients.Gw1 = X.transpose() * gradients.Ga1;
    gradients.Gb1 = gradients.Ga1.colwise().sum();
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
void backPropAdam(Weights& weights, const Gradients& gradients, AdamState& opt)
{
    opt.t += 1; // one step per minibatch

    adam_update(weights.W1, gradients.Gw1, opt.mW1, opt.vW1, opt.t, (float)lr);
    adam_update(weights.b1, gradients.Gb1, opt.mb1, opt.vb1, opt.t, (float)lr);

    adam_update(weights.W2, gradients.Gw2, opt.mW2, opt.vW2, opt.t, (float)lr);
    adam_update(weights.b2, gradients.Gb2, opt.mb2, opt.vb2, opt.t, (float)lr);

    adam_update(weights.W3, gradients.Gw3, opt.mW3, opt.vW3, opt.t, (float)lr);
    adam_update(weights.b3, gradients.Gb3, opt.mb3, opt.vb3, opt.t, (float)lr);

    adam_update(weights.Wmu, gradients.Gwmu,opt.mWmu, opt.vWmu, opt.t, (float)lr);
    adam_update(weights.bmu, gradients.Gbmu,opt.mbmu, opt.vbmu, opt.t, (float)lr);

    adam_update(weights.Wv, gradients.Gwv, opt.mWv, opt.vWv, opt.t, (float)lr);
    adam_update(weights.bv, gradients.Gbv, opt.mbv, opt.vbv, opt.t, (float)lr);
}






// Save a matrix as CSV: one row per line, comma-separated
void save_matrix_csv(const Eigen::MatrixXf& M, const std::string& path)
{
    std::ofstream out(path);
    if (!out) {
        std::cerr << "ERROR: cannot open file for writing: " << path << "\n";
        return;
    }

    const int rows = M.rows();
    const int cols = M.cols();
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j) {
            out << M(i, j);
            if (j + 1 < cols) out << ",";  // comma between columns
        }
        out << "\n";
    }
    out.close();
}



/** 
 @brief Save to csv the latent representation of images
 @param M matrix that is saved, often H
 @param path path of the csv
 @param a number of images to save 
*/
void save_matrix_images_csv(const Eigen::MatrixXf& M, const std::string& path, const int& a)
{
    std::ofstream out(path);
    if (!out) {
        std::cerr << "ERROR: cannot open file for writing: " << path << "\n";
        return;
    }
    const int rows = a;
    const int cols = M.cols();
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j) {
            out << M(i, j);
            if (j + 1 < cols) out << ",";  // comma between columns
        }
        out << "\n";
    }
    out.close();
}
