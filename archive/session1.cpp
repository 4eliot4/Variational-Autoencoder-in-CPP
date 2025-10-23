#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <chrono>
using Eigen::MatrixXd;

static double epsilon = 1e-6;

double numGrad(int idi, int idj, const MatrixXd &A, const MatrixXd& B);
MatrixXd numGradMat(const MatrixXd &A, const MatrixXd &B);
MatrixXd analGradMat(const MatrixXd &A, const MatrixXd &B);

int main()
{
    MatrixXd A = MatrixXd::Random(3, 2);
    MatrixXd B = MatrixXd::Random(2, 3);
    MatrixXd M = A * B;

    double Loss = M.squaredNorm();
    std::cout << "Loss = " << Loss << std::endl;
    std::cout << "Grad loss :" << numGrad(1, 1, A, B) << std::endl;
    std::cout << "Grad loss Mat Numeric: " << numGradMat(A, B) << std::endl;
    std::cout << "Grad loss Mat Analytic: " << analGradMat(A, B) << std::endl;

    return 0;
}

double numGrad(int idi, int idj, const MatrixXd &A,const MatrixXd &B)
{
    MatrixXd Aplus = A;
    MatrixXd Aminus = A;
    Aplus(idi, idj)  += epsilon;
    Aminus(idi, idj) -= epsilon;
    MatrixXd Mplus = Aplus* B;
    MatrixXd Mminus = Aminus * B;
    
    return (Mplus.squaredNorm() - Mminus.squaredNorm()) / (2 * epsilon);
    
}
MatrixXd numGradMat(const MatrixXd &A,const MatrixXd &B)
{
    MatrixXd gradLossMat = MatrixXd(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); i++)
    {
        for (int j = 0; j < A.cols(); j++)
        {
            gradLossMat(i, j) = numGrad(i, j, A, B);
        }
    }

    return gradLossMat;
}

MatrixXd analGradMat(const MatrixXd &A, const MatrixXd &B)
{
    MatrixXd BBt = B * B.transpose(); 
    MatrixXd G(A.rows(), A.cols());
    G.noalias() = 2.0 * A * BBt;      
    return G;
}