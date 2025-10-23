#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>

int main() {
    Eigen::Matrix2d A; A << 1,2,3,4;
    Eigen::Vector2d b(5,6);
    std::cout << "A*b =\n" << A*b << "\n";
    
    return 0;
}
