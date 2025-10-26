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
    Weights weights;
    ForwardOutput forward;
    Gradients gradients;
    std::cout << "Loss no pass : ";forward.lossPrint();
    
    forwardPass(forward, weights, X);
    backPass(gradients, forward, weights, X);
    backProp(weights,gradients);
    // std::cout << "first pass : "; weights.print();

    std::cout << "Loss first pass : ";forward.lossPrint();

    for (size_t i = 0; i <= 10000; i++)
    {
        X = make_batch_downsampled(16, rng, ShapeType::Any);
        forwardPass(forward, weights, X);
        backPass(gradients, forward, weights, X);
        backProp(weights,gradients);
        if ( i % 100 == 0)
        {
            std::cout << "loss after :" << i << "iterations : "; forward.lossPrint();
        }
            
    }

    // std::cout << "After 100 iterations :"; weights.print();
    std::cout << "Loss after 100 iterations : ";forward.lossPrint();
    return 0;
}
