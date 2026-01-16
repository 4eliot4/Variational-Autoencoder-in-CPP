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
void interpolation(Eigen::MatrixXf &X1, Eigen::MatrixXf &X2, const Weights &weights, ForwardOutput &forward, const double &t);
size_t iterations = 1000;

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
        
        if ( i % 100 == 0)
        {
            std::cout << "loss after :" << i << "iterations : "; forward.lossPrint();
        }
        if(i % 1000 == 0)
        {
            generateOutput(rng, forward, weights, i);
            std::ostringstream hPath;
            hPath << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"
                  << "H_latent_iter" << std::setw(5) << std::setfill('0') << i << ".csv";
            save_matrix_csv(forward.H,hPath.str());
            //save_matrix_images_csv(forward.H, hPath.str());
        }
    }
    Eigen::MatrixXf X1 = make_batch_mnist(B, rng, true);
    Eigen::MatrixXf X2 = make_batch_mnist(B, rng, true);
    double t = 0.5;
    interpolation(X1, X2, weights, forward, t);
    return 0;
}


void generateOutput(std::mt19937 &rng, ForwardOutput& forward, const Weights& weights, int iteration)
{
    // Load an image
    std::mt19937 rng1(1020u);
    Eigen::MatrixXf X_test = make_batch_mnist(B, rng, true);
    //Eigen::MatrixXf X_test = make_single_image_batch(1234, B);
    //Eigen::MatrixXf X_test = make_two_images_batch(1234, 1235, B);
    X_test = (X_test.array() > 0.5f).cast<float>();
    std::ostringstream inputPath;
    inputPath << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"<< "INPUT_After_" << std::setw(5) << std::setfill('0') << iteration << ".png";

    forwardPass(forward, weights, X_test);
    // Save
    std::ostringstream path;
    path << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"<< "OUTPUT_After_" << std::setw(5) << std::setfill('0') << iteration << ".png";

    // Sauvegarde de la reconstruction
    bool input = write_png_grid_mnist(X_test, 4, 4, inputPath.str()) ;
    bool ok = write_png_grid_mnist(forward.sigmoid, 4, 4, path.str());

    if (!input) {
        std::cerr << " Failed to write " << inputPath.str() << "\n";
    } else {
        std::cout << "Saved input no. " << iteration<< "\n";
    }
    if (!ok) {
        std::cerr << " Failed to write " << path.str() << "\n";
    } else {
        std::cout << "Saved output no." << iteration << "\n";
    }
}

void interpolation(Eigen::MatrixXf& X1,Eigen::MatrixXf& X2,const Weights& weights, ForwardOutput& forward, const double& t)
{
    forwardPass(forward, weights, X1);
    Eigen::MatrixXf h1 = forward.H;
    forwardPass(forward, weights, X2);
    Eigen::MatrixXf h2 = forward.H;
    forward.H = (1.0 - t) * h1 + t * h2;

    decoder(forward, weights);

    std::cout << "interpolation is : " << std::endl;

    std::ostringstream inputPath1;
    inputPath1 << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"<< "Interpol_INPUT_1"  << std::setfill('0') << ".png";    
    std::ostringstream inputPath2;
    inputPath2 << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"<< "Interpol_INPUT_2"  << std::setfill('0') << ".png";
    std::ostringstream outputPath;
    outputPath << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"<< "Interpol_OUTPUT"  << std::setfill('0') << ".png";

    write_png_grid_mnist(X1, 4, 4, inputPath1.str()) ;
    write_png_grid_mnist(X2, 4, 4, inputPath2.str()) ;
    write_png_grid_mnist(forward.sigmoid, 4, 4, outputPath.str());
}