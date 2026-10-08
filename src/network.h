#ifndef NETWORK_H
#define NETWORK_H

#include <Eigen/Dense>
#include <iostream>
#include <random>

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
    Weights();
    void print();
};
struct ForwardOutput
{
    Eigen::MatrixXf X;
    Eigen::MatrixXf H;
    Eigen::MatrixXf Z2;
    Eigen::MatrixXf A2;
    Eigen::MatrixXf Yhat;
    Eigen::MatrixXf sigmoid; // sigmoid of Y
    double loss;
    ForwardOutput();
    void lossPrint();
};

struct Gradients
{
    Eigen::MatrixXf Gy, Gw3,Ga2,Gz2,Gm,Gv,Gw2, Gh, Gz, Gw1;
    Eigen::MatrixXf Gb3,Gb2,Gbv,Gbmu,Gb1;
    Gradients();
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

void forwardPass(ForwardOutput &forward, const Weights &weights, const Eigen::MatrixXf &X);
void decoder(ForwardOutput &forward, const Weights &weights);
void backPass(Gradients &gradients, const ForwardOutput &forward, const Weights &weights, const Eigen::MatrixXf &X);
void backProp(Weights &weights, const Gradients &gradients);
void backPropAdam(Weights& weights, const Gradients& gradients, AdamState& opt);

void save_matrix_csv(const Eigen::MatrixXf &M, const std::string &path);
void save_matrix_images_csv(const Eigen::MatrixXf &M, const std::string &path, const int& a = 1);

#endif // NETWORK_H
