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
    // Eigen::MatrixXd W1(256,H); wrong: its not gonna call the constructor
    Eigen::MatrixXd W1;
    Eigen::RowVectorXd b1;
    Eigen::MatrixXd W2;
    Eigen::RowVectorXd b2;

    Weights();
    void print();
};
struct ForwardOutput
{
    Eigen::MatrixXd Z;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Yhat;
    Eigen::MatrixXd sigmoid; // sigmoid of Y
    double loss;
    ForwardOutput();
};

struct Gradients
{
    Eigen::MatrixXd Gy, Gw2, Gh, Gz, Gw1;
    Eigen::RowVectorXd Gb2, Gb1;
    Gradients();
};

void forwardPass(ForwardOutput &forward, const Weights &weights, const Eigen::MatrixXd &X);
void backPass(Gradients &gradients, const ForwardOutput &forward, const Weights &weights, const Eigen::MatrixXd &X);
void backProp(Weights &weights, const Gradients &gradients);


#endif // NETWORK_H