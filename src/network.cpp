#include "network.h"
#include <cmath>
#include <stdexcept>
#include <fstream>


// ====== SETTINGS ======
int H_size = 400;
int D = 784;
int B = 64;

constexpr float betaKL = 0.9f;

double lr = 0.0003f;

auto xavier = [](int fan_in, int fan_out){ return std::sqrt(2.0f / float(fan_in + fan_out)); };

Weights::Weights() : W1(Eigen::MatrixXf::Random(D,H_size) * xavier(D, H_size)),
                     b1(Eigen::MatrixXf::Zero(1,H_size)),
                     W2(Eigen::MatrixXf::Random(LATENT_SIZE, FEATURE_SIZE)* xavier(LATENT_SIZE, FEATURE_SIZE)),
                     b2(Eigen::MatrixXf::Zero(1, FEATURE_SIZE)),
                     W3(Eigen::MatrixXf::Random(H_size,D) * xavier(H_size, D)),
                     b3(Eigen::MatrixXf::Zero(1,D)),
                     Wmu(Eigen::MatrixXf::Random(FEATURE_SIZE, LATENT_SIZE)* xavier(FEATURE_SIZE, LATENT_SIZE)),
                     bmu(Eigen::MatrixXf::Zero(1, LATENT_SIZE)),
                     Wv(Eigen::MatrixXf::Random(FEATURE_SIZE, LATENT_SIZE) * xavier(FEATURE_SIZE, LATENT_SIZE)),
                     bv(Eigen::MatrixXf::Zero(1, LATENT_SIZE)),
                     conv1(channels_in, channels_1,kernel, stride, padding),
                     conv2(channels_1, channels_2,kernel, stride, padding),
                     deconv1(channels_2, channels_1,kernel, stride, padding),
                     deconv2(channels_1, channels_in,kernel, stride, padding)
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


ForwardOutput::ForwardOutput(): X(B, H_size),
                                H(B, FEATURE_SIZE),
                                M(B, LATENT_SIZE),
                                V(B, LATENT_SIZE),
                                E(B, LATENT_SIZE),
                                Z(B, LATENT_SIZE),

                                A2(B, FEATURE_SIZE),
                                H2(B, FEATURE_SIZE),
                                Yhat(B, INPUT_SIZE),
                                sigmoid(B, INPUT_SIZE),

                                // Encoder
                                convInput(B, channels_in, IMAGE_SIZE, IMAGE_SIZE),

                                convA1(B, channels_1, CONV1_SIZE, CONV1_SIZE),
                                convH1(B, channels_1, CONV1_SIZE, CONV1_SIZE),

                                convA2(B, channels_2, CONV2_SIZE, CONV2_SIZE),
                                convH2(B, channels_2, CONV2_SIZE, CONV2_SIZE),

                                // Decoder
                                decInput(B, channels_2, CONV2_SIZE, CONV2_SIZE),

                                decA1(B, channels_1, DECONV1_SIZE, DECONV1_SIZE),
                                decH1(B, channels_1, DECONV1_SIZE, DECONV1_SIZE),

                                decA2(B, channels_in, DECONV2_SIZE, DECONV2_SIZE)
{
}

void ForwardOutput::lossPrint()
{
    std::cout << "Total: " << loss
              << " | Reconstruction: " << lossRec
              << " | KL: " << lossKL
              << std::endl;
}

Gradients::Gradients() : Gy(B, D), 
                         Gw3(H_size,D),
                         Gh2(B,FEATURE_SIZE),
                         Ga2(B,FEATURE_SIZE),
                         Gm(B,LATENT_SIZE),
                         Gv(B,LATENT_SIZE),
                         Gwmu(FEATURE_SIZE, LATENT_SIZE),
                         Gwv(FEATURE_SIZE, LATENT_SIZE),  
                         Gw2(LATENT_SIZE, FEATURE_SIZE), 
                         Gh(B, FEATURE_SIZE),
                         Ga1(B,H_size), 
                         Gz(B, LATENT_SIZE), 
                         Gw1(D, H_size),
                         Gb3(1,D),
                         Gb2(1, FEATURE_SIZE), 
                         Gbv(1, LATENT_SIZE),
                         Gbmu(1, LATENT_SIZE),
                         Gb1(1, H_size),
                         // encoder
                         Gconv1W(channels_1, channels_in, kernel, kernel),
                         Gconv2W(channels_2, channels_1, kernel, kernel),
                         Gconv1b(Eigen::VectorXf::Zero(channels_1)),
                         Gconv2b(Eigen::VectorXf::Zero(channels_2)),

                         // Decoder
                         Gdeconv1W(channels_1, channels_2, kernel, kernel),
                         Gdeconv2W(channels_in, channels_1, kernel, kernel),
                         Gdeconv1b(Eigen::VectorXf::Zero(channels_1)),
                         Gdeconv2b(Eigen::VectorXf::Zero(channels_in))
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


    // CNN encoeder
    forward.convInput = matrixToTensor(X,channels_in,IMAGE_SIZE,IMAGE_SIZE);

    // First conv
    forward.convA1 = convForward(forward.convInput, weights.conv1);
    forward.convH1 = reluForward(forward.convA1);

    // Second conv
    forward.convA2 = convForward(forward.convH1, weights.conv2);
    forward.convH2 = reluForward(forward.convA2);

    // Flatten for the Gauss
    forward.H = flatten(forward.convH2);

    // VAE
    // Mean: M = H * Wmu + bmu
    forward.M = forward.H * weights.Wmu;
    forward.M.rowwise() += weights.bmu.row(0);
    // Log-variance: V = H * Wv + bv
    forward.V = forward.H * weights.Wv;
    forward.V.rowwise() += weights.bv.row(0);

    forward.E.resize(forward.M.rows(), forward.M.cols());
    // SAMPLE
    std::normal_distribution<float> normal(0.0f, 1.0f);

    for (int i = 0; i < forward.E.rows(); ++i) {
        for (int j = 0; j < forward.E.cols(); ++j) {
            forward.E(i, j) = normal(rng);
        }
    }
    forward.Z = forward.M.array() + (0.5f * forward.V.array()).exp() * forward.E.array();

    // convo decoder 
    decoder(forward, weights);
    
    const float eps = 1e-7f; // small safety
    Eigen::ArrayXXf s = forward.sigmoid.array().min(1.0f - eps).max(eps);

    Eigen::MatrixXf loss_per_entry = -(X.array() * s.log() + (1.0f - X.array()) * (1.0f - s).log()).matrix();
    forward.lossRec = loss_per_entry.sum() / B;
    forward.lossKL = 0.5f / B * (forward.M.array().square() + forward.V.array().exp() - 1.0f- forward.V.array()).sum();
    forward.loss = forward.lossRec + betaKL * forward.lossKL;
}

/**
 * @brief convolution decoder 
 */

void decoder(ForwardOutput& forward, const Weights& weights)
{

    forward.A2 = forward.Z * weights.W2;
    forward.A2.rowwise() += weights.b2.row(0);

    forward.H2 = forward.A2.array().cwiseMax(0.0f);


    forward.decInput = unflatten(forward.H2,channels_2,CONV2_SIZE,CONV2_SIZE);

    forward.decA1 = convTransposeForward(forward.decInput,weights.deconv1);
    forward.decH1 = reluForward(forward.decA1);
    forward.decA2 = convTransposeForward(forward.decH1,weights.deconv2);
    forward.Yhat = flatten(forward.decA2);

    forward.sigmoid = 1.0f / (1.0f + (-forward.Yhat.array()).exp());
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
    
    
    gradients.Gy = (forward.sigmoid - X) / B; // BCE gradient wrt output logits
    Tensor4D dLogits = unflatten(gradients.Gy,channels_in,DECONV2_SIZE,DECONV2_SIZE);  // Convert gradient back into image dimensions
    
    ConvGradients GD2 = convTransposeBackward(forward.decH1,weights.deconv2,dLogits);// Backprop through second transposed convolution
    Tensor4D dAdec1 = reluBackward(forward.decA1,GD2.dX);
    ConvGradients GD1 = convTransposeBackward(forward.decInput,weights.deconv1,dAdec1);
    gradients.Gh2 = flatten(GD1.dX);

    
    Eigen::MatrixXf mask =(forward.A2.array() > 0.0f).cast<float>();
    gradients.Ga2 = (gradients.Gh2.array() * mask.array()).matrix(); // Backprop through dense ReLU
    gradients.Gw2 = forward.Z.transpose() * gradients.Ga2;
    gradients.Gb2 = gradients.Ga2.colwise().sum();
    gradients.Gz = gradients.Ga2 * weights.W2.transpose();
    // Save decoder conv grad
    gradients.Gdeconv1W = GD1.dW;
    gradients.Gdeconv1b = GD1.db;

    gradients.Gdeconv2W = GD2.dW;
    gradients.Gdeconv2b = GD2.db;


    // VAEE, .array and * FOR ELEMENT WISE OPERATIONS
    gradients.Gm = gradients.Gz + betaKL * forward.M / B;
    gradients.Gv = 0.5f * gradients.Gz.array() * (0.5f * forward.V.array()).exp() * forward.E.array() + betaKL * (forward.V.array().exp() - 1.0f) / (2.0f * B);
    gradients.Gwmu = forward.H.transpose() * gradients.Gm;
    gradients.Gbmu = gradients.Gm.colwise().sum();
    gradients.Gwv = forward.H.transpose() * gradients.Gv;
    gradients.Gbv = gradients.Gv.colwise().sum();
    gradients.Gh = gradients.Gm * weights.Wmu.transpose() + gradients.Gv * weights.Wv.transpose();

    Tensor4D dHconv2 = unflatten(gradients.Gh, channels_2, CONV2_SIZE, CONV2_SIZE);
    Tensor4D dAconv2 = reluBackward(forward.convA2, dHconv2);
    ConvGradients G2 = convBackward(forward.convH1, weights.conv2, dAconv2);
    Tensor4D dAconv1 = reluBackward(forward.convA1, G2.dX);
    ConvGradients G1 = convBackward(forward.convInput, weights.conv1, dAconv1);

    gradients.Gconv2W = G2.dW;
    gradients.Gconv2b = G2.db;

    gradients.Gconv1W = G1.dW;
    gradients.Gconv1b = G1.db;

}

static void ensureMoments(Eigen::MatrixXf& m,Eigen::MatrixXf& v,const Eigen::MatrixXf& theta)
{
    if (m.rows() != theta.rows() ||
        m.cols() != theta.cols() ||
        v.rows() != theta.rows() ||
        v.cols() != theta.cols())
    {
        m.setZero(theta.rows(), theta.cols());
        v.setZero(theta.rows(), theta.cols());
    }
}
static void adam_update_tensor(Tensor4D& theta,const Tensor4D& grad,Tensor4D& m,Tensor4D& v,int t,float lr)
{
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps = 1e-8f;

    if (grad.size() != theta.size() ||
        m.size() != theta.size() ||
        v.size() != theta.size())
        throw std::invalid_argument("Adam tensor size mismatch");

    const float correction1 = 1.0f - std::pow(beta1, float(t));
    const float correction2 = 1.0f - std::pow(beta2, float(t));

    for (Eigen::Index i = 0; i < theta.size(); ++i)
    {
        const float g = grad.data()[i];

        m.data()[i] =
            beta1 * m.data()[i] + (1.0f - beta1) * g;

        v.data()[i] =
            beta2 * v.data()[i] + (1.0f - beta2) * g * g;

        const float mhat = m.data()[i] / correction1;
        const float vhat = v.data()[i] / correction2;

        theta.data()[i] -=
            lr * mhat / (std::sqrt(vhat) + eps);
    }
}


static void adam_update_vector(Eigen::VectorXf& theta,const Eigen::VectorXf& grad,Eigen::VectorXf& m,Eigen::VectorXf& v,int t,float lr)
{
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps = 1e-8f;

    const float c1 = 1.0f - std::pow(beta1, float(t));
    const float c2 = 1.0f - std::pow(beta2, float(t));

    m = beta1 * m + (1.0f - beta1) * grad;

    v.array() =
        beta2 * v.array()
        + (1.0f - beta2) * grad.array().square();

    theta.array() -=
        lr * (m.array() / c1)
        / ((v.array() / c2).sqrt() + eps);
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

    // ensure correct dim
    ensureMoments(opt.mWmu, opt.vWmu, weights.Wmu);
    ensureMoments(opt.mWv,  opt.vWv,  weights.Wv);
    ensureMoments(opt.mW2,  opt.vW2,  weights.W2);
    ensureMoments(opt.mb2, opt.vb2, weights.b2);
    ensureMoments(opt.mbmu, opt.vbmu, weights.bmu);
    ensureMoments(opt.mbv,  opt.vbv,  weights.bv);

    // First conv
    adam_update_tensor(weights.conv1.W, gradients.Gconv1W,opt.conv1.mW, opt.conv1.vW,opt.t, float(lr));
    adam_update_vector(weights.conv1.b, gradients.Gconv1b,opt.conv1.mb, opt.conv1.vb,opt.t, float(lr));

    // Second conv
    adam_update_tensor(weights.conv2.W, gradients.Gconv2W,opt.conv2.mW, opt.conv2.vW,opt.t, float(lr));
    adam_update_vector(weights.conv2.b, gradients.Gconv2b,opt.conv2.mb, opt.conv2.vb,opt.t, float(lr));



    adam_update(weights.W2, gradients.Gw2, opt.mW2, opt.vW2, opt.t, (float)lr);
    adam_update(weights.b2, gradients.Gb2, opt.mb2, opt.vb2, opt.t, (float)lr);

    

    adam_update(weights.Wmu, gradients.Gwmu,opt.mWmu, opt.vWmu, opt.t, (float)lr);
    adam_update(weights.bmu, gradients.Gbmu,opt.mbmu, opt.vbmu, opt.t, (float)lr);

    adam_update(weights.Wv, gradients.Gwv, opt.mWv, opt.vWv, opt.t, (float)lr);
    adam_update(weights.bv, gradients.Gbv, opt.mbv, opt.vbv, opt.t, (float)lr);

    adam_update_tensor(weights.deconv1.W,gradients.Gdeconv1W,opt.deconv1.mW,opt.deconv1.vW,opt.t,float(lr));
    adam_update_vector(weights.deconv1.b,gradients.Gdeconv1b,opt.deconv1.mb,opt.deconv1.vb,opt.t,float(lr));
    adam_update_tensor(weights.deconv2.W,gradients.Gdeconv2W,opt.deconv2.mW,opt.deconv2.vW,opt.t,float(lr));
    adam_update_vector(weights.deconv2.b,gradients.Gdeconv2b,opt.deconv2.mb,opt.deconv2.vb,opt.t,float(lr));

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
