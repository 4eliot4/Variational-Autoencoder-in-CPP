#ifndef NETWORK_H
#define NETWORK_H

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include "conv.h"
// ====== CONSTANTS ======
extern int H_size;
extern int D;
extern int B;
extern double lr;



inline constexpr int IMAGE_SIZE = 64;

inline constexpr int channels_in = 1;
inline constexpr int channels_1 = 32;
inline constexpr int channels_2 = 64;

inline constexpr int kernel = 4;
inline constexpr int stride = 2;
inline constexpr int padding = 1;


inline constexpr int CONV1_SIZE = (IMAGE_SIZE + 2 * padding - kernel) / stride + 1;
inline constexpr int CONV2_SIZE = (CONV1_SIZE + 2 * padding - kernel) / stride + 1;


inline constexpr int DECONV1_SIZE = (CONV2_SIZE - 1) * stride - 2 * padding + kernel;
inline constexpr int DECONV2_SIZE = (DECONV1_SIZE - 1) * stride - 2 * padding + kernel;
inline constexpr int LATENT_SIZE = 32;
inline constexpr int FEATURE_SIZE = channels_2 * CONV2_SIZE * CONV2_SIZE;
inline constexpr int INPUT_SIZE = channels_in * IMAGE_SIZE * IMAGE_SIZE;
static_assert(DECONV2_SIZE == IMAGE_SIZE,"Decoder output must match input image size");



struct Weights
{
    // Eigen::MatrixXf W1(256,H); wrong: its not gonna call the constructor
    Eigen::MatrixXf W1;
    Eigen::MatrixXf b1;
    Eigen::MatrixXf W2;
    Eigen::MatrixXf b2;
    Eigen::MatrixXf W3;
    Eigen::MatrixXf b3;
    Eigen::MatrixXf Wmu;
    Eigen::MatrixXf bmu;
    Eigen::MatrixXf Wv;
    Eigen::MatrixXf bv;
    Conv2D conv1;
    Conv2D conv2;
    ConvTranspose2D deconv1;
    ConvTranspose2D deconv2;
    Weights();
    void print();
};
struct ForwardOutput
{
    Eigen::MatrixXf X;
    Eigen::MatrixXf H;
    Eigen::MatrixXf M;
    Eigen::MatrixXf V;
    Eigen::MatrixXf E;
    Eigen::MatrixXf Z;
    Eigen::MatrixXf A2;
    Eigen::MatrixXf H2;
    Eigen::MatrixXf Yhat;
    Eigen::MatrixXf sigmoid; // sigmoid of Y
    Tensor4D convInput;
    Tensor4D convA1, convH1;
    Tensor4D convA2, convH2;
    Tensor4D decInput;  // (B,64,7,7)
    Tensor4D decA1;     // (B,32,14,14) 
    Tensor4D decH1;     // (B,32,14,14)
    Tensor4D decA2;     // (B,1,28,28)
    double loss;
    double lossRec;
    double lossKL;
    ForwardOutput();
    void lossPrint();
};

struct Gradients
{
    Eigen::MatrixXf Gy, Gw3,Gh2,Ga2,Gm,Gv,Gwmu,Gwv,Gw2, Gh,Ga1, Gz, Gw1;
    Eigen::MatrixXf Gb3,Gb2,Gbv,Gbmu,Gb1;
    Tensor4D Gconv1W;
    Tensor4D Gconv2W;

    Eigen::VectorXf Gconv1b;
    Eigen::VectorXf Gconv2b;
    Tensor4D Gdeconv1W;
    Tensor4D Gdeconv2W;

    Eigen::VectorXf Gdeconv1b;
    Eigen::VectorXf Gdeconv2b;
    Gradients();
};

struct ConvAdamState
{
    Tensor4D mW;
    Tensor4D vW;

    Eigen::VectorXf mb;
    Eigen::VectorXf vb;

    ConvAdamState(int Cin, int Cout, int K)
        : mW(Cout, Cin, K, K),
          vW(Cout, Cin, K, K),
          mb(Eigen::VectorXf::Zero(Cout)),
          vb(Eigen::VectorXf::Zero(Cout))
    {
        mW.setZero();
        vW.setZero();
    }
};

struct AdamState 
{
    // moments for each parameter
    Eigen::MatrixXf mW1, vW1;
    Eigen::MatrixXf mb1, vb1;

    Eigen::MatrixXf mW2, vW2;
    Eigen::MatrixXf mb2, vb2;

    Eigen::MatrixXf mW3, vW3;
    Eigen::MatrixXf mb3, vb3;

    Eigen::MatrixXf mWmu, vWmu;
    Eigen::MatrixXf mbmu, vbmu;

    Eigen::MatrixXf mWv, vWv;
    Eigen::MatrixXf mbv, vbv;

        
    ConvAdamState conv1{channels_in, channels_1, kernel};
    ConvAdamState conv2{channels_1, channels_2, kernel};

    ConvAdamState deconv1{channels_2, channels_1, kernel};
    ConvAdamState deconv2{channels_1, channels_in, kernel};


    int t = 0;

    AdamState()
      : 
        mW2(Eigen::MatrixXf::Zero(LATENT_SIZE, FEATURE_SIZE)),
        vW2(Eigen::MatrixXf::Zero(LATENT_SIZE, FEATURE_SIZE)),

        mb2(Eigen::MatrixXf::Zero(1, FEATURE_SIZE)),
        vb2(Eigen::MatrixXf::Zero(1, FEATURE_SIZE)),

        mWmu(Eigen::MatrixXf::Zero(FEATURE_SIZE, LATENT_SIZE)),
        vWmu(Eigen::MatrixXf::Zero(FEATURE_SIZE, LATENT_SIZE)),

        mbmu(Eigen::MatrixXf::Zero(1, LATENT_SIZE)),
        vbmu(Eigen::MatrixXf::Zero(1, LATENT_SIZE)),

        mWv(Eigen::MatrixXf::Zero(FEATURE_SIZE, LATENT_SIZE)),
        vWv(Eigen::MatrixXf::Zero(FEATURE_SIZE, LATENT_SIZE)),

        mbv(Eigen::MatrixXf::Zero(1, LATENT_SIZE)),
        vbv(Eigen::MatrixXf::Zero(1, LATENT_SIZE))
    {}
};

void decoder(ForwardOutput &forward, const Weights &weights);
void forwardPass(ForwardOutput &forward, const Weights &weights, const Eigen::MatrixXf &X, std::mt19937 &rng);
void backPass(Gradients &gradients, const ForwardOutput &forward, const Weights &weights, const Eigen::MatrixXf &X);
void backProp(Weights &weights, const Gradients &gradients);
void backPropAdam(Weights& weights, const Gradients& gradients, AdamState& opt);

void save_matrix_csv(const Eigen::MatrixXf &M, const std::string &path);
void save_matrix_images_csv(const Eigen::MatrixXf &M, const std::string &path, const int& a = 1);

#endif // NETWORK_H
