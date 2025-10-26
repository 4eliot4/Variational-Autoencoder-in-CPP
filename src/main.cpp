#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>
#include "shapes.h"
#include "network.h"

int main()
{
    std::mt19937 rng(1337u); // random generator
    Weights weights;

    Eigen::MatrixXf X = make_batch_downsampled(16, rng, ShapeType::Any);
    bool ok = write_png_grid(
        X,
        /*gridCols=*/4,
        /*gridRows=*/4,
        "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/MAIN_Test.png"
    );

    if (!ok) {
        std::cerr << "png write failed\n";
    }

    weights.print();

    return 0;
}
