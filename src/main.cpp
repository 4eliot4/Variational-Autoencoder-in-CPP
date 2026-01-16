#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "shape.h"
#include "network.h"

void generateOutput(std::__1::mt19937 &rng, ForwardOutput &forward, const Weights &weights, int iteration);
size_t iterations = 500000;


int main()
{
    std::mt19937 rng(1337u); // random generator
    

    Eigen::MatrixXf X = make_batch_mnist(B, rng, true);
    Weights weights;
    ForwardOutput forward;
    Gradients gradients;
    AdamState opt;
    for (size_t i = 0; i <= iterations; i++)
    {
        X = make_batch_mnist(B, rng, true);
        // X = make_single_image_batch(1234, B); // overfit test
        //X = make_two_images_batch(1234, 1235, B);
        X = (X.array() > 0.5f).cast<float>();
        forwardPass(forward, weights, X);
        backPass(gradients, forward, weights, X);
        backProp(weights,gradients);
        backPropAdam(weights, gradients, opt);
        if (i == 0) {
            std::cout << "X(0) mean=" << X.row(0).mean() << "  X(1) mean=" << X.row(1).mean() << "\n";
        }
        if ( i % 100 == 0)
        {
            std::cout << "loss after :" << i << "iterations : "; forward.lossPrint();
        }
        if(i % 100 == 0)
        {
            generateOutput(rng, forward, weights, i);
            float d01 = (forward.H.row(0) - forward.H.row(1)).norm();
            std::cout << "latent distance ||H0-H1|| = " << d01 << "\n";
            std::ostringstream hPath;
            hPath << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"
                  << "H_latent_iter_" << std::setw(5) << std::setfill('0') << i << ".csv";
            save_matrix_csv(forward.H,hPath.str());
        }
    }
    std::cout << "Loss after 100 iterations : ";forward.lossPrint();
    return 0;
}


void generateOutput(std::mt19937 &rng, ForwardOutput& forward, const Weights& weights, int iteration)
{
    // Load an image
    Eigen::MatrixXf X_test = make_batch_mnist(B, rng, true);
    //Eigen::MatrixXf X_test = make_single_image_batch(1234, B);
    //Eigen::MatrixXf X_test = make_two_images_batch(1234, 1235, B);
    X_test = (X_test.array() > 0.5f).cast<float>();
    std::ostringstream inputPath;
    inputPath << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"<< "INPUT_After_" << std::setw(5) << std::setfill('0') << iteration << ".png";

    forwardPass(forward, weights, X_test);
    std::cout << "sigmoid: min=" << forward.sigmoid.minCoeff()
          << " max=" << forward.sigmoid.maxCoeff()
          << " mean=" << forward.sigmoid.mean() << "\n";
    // Save
    std::ostringstream path;
    path << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"<< "OUTPUT_After_" << std::setw(5) << std::setfill('0') << iteration << ".png";

    // Sauvegarde de la reconstruction
    bool input = write_png_grid_mnist(X_test, 4, 4, inputPath.str()) ;
    bool ok = write_png_grid_mnist(forward.sigmoid, 4, 4, path.str());

    if (!input) {
        std::cerr << " Failed to write " << inputPath.str() << "\n";
    } else {
        std::cout << "Saved " << inputPath.str() << "\n";
    }
    if (!ok) {
        std::cerr << " Failed to write " << path.str() << "\n";
    } else {
        std::cout << "Saved " << path.str() << "\n";
    }
}