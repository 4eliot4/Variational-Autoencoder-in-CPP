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

    ConvAdamState conv1{1, 32, 4};
    ConvAdamState conv2{32, 64, 4};

    int t = 0;

    AdamState()
      : mW1(Eigen::MatrixXf::Zero(D, H_size)), vW1(Eigen::MatrixXf::Zero(D, H_size)),
        mb1(Eigen::MatrixXf::Zero(1, H_size)), vb1(Eigen::MatrixXf::Zero(1, H_size)),
        mW2(Eigen::MatrixXf::Zero(H_size, H_size)), vW2(Eigen::MatrixXf::Zero(H_size, H_size)),
        mb2(Eigen::MatrixXf::Zero(1, H_size)), vb2(Eigen::MatrixXf::Zero(1, H_size)),
        mW3(Eigen::MatrixXf::Zero(H_size, D)), vW3(Eigen::MatrixXf::Zero(H_size, D)),
        mb3(Eigen::MatrixXf::Zero(1, D)), vb3(Eigen::MatrixXf::Zero(1, D)),
        mWmu(Eigen::MatrixXf::Zero(H_size, H_size)), vWmu(Eigen::MatrixXf::Zero(H_size, H_size)),
        mbmu(Eigen::MatrixXf::Zero(1, H_size)), vbmu(Eigen::MatrixXf::Zero(1, H_size)),
        mWv(Eigen::MatrixXf::Zero(H_size, H_size)), vWv(Eigen::MatrixXf::Zero(H_size, H_size)),
        mbv(Eigen::MatrixXf::Zero(1, H_size)), vbv(Eigen::MatrixXf::Zero(1, H_size))
    {}
};



void forwardPass(ForwardOutput &forward, const Weights &weights, const Eigen::MatrixXf &X,std::mt19937& rng);
void decoder(ForwardOutput &forward, const Weights &weights);
void backPass(Gradients &gradients, const ForwardOutput &forward, const Weights &weights, const Eigen::MatrixXf &X);
void backProp(Weights &weights, const Gradients &gradients);
void backPropAdam(Weights& weights, const Gradients& gradients, AdamState& opt);

void save_matrix_csv(const Eigen::MatrixXf &M, const std::string &path);
void save_matrix_images_csv(const Eigen::MatrixXf &M, const std::string &path, const int& a = 1);

#endif // NETWORK_H
