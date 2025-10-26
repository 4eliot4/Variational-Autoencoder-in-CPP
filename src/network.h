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
    Eigen::RowVectorXf b1;
    Eigen::MatrixXf W2;
    Eigen::RowVectorXf b2;
    Eigen::MatrixXf W3;
    Eigen::RowVectorXf b3;
    Weights();
    void print();
};
struct ForwardOutput
{
    Eigen::MatrixXf Z;
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
    Eigen::MatrixXf Gy, Gw3,Ga2,Gz2,Gw2, Gh, Gz, Gw1;
    Eigen::RowVectorXf Gb3,Gb2, Gb1;
    Gradients();
};

void forwardPass(ForwardOutput &forward, const Weights &weights, const Eigen::MatrixXf &X);
void backPass(Gradients &gradients, const ForwardOutput &forward, const Weights &weights, const Eigen::MatrixXf &X);
void backProp(Weights &weights, const Gradients &gradients);


#endif // NETWORK_H