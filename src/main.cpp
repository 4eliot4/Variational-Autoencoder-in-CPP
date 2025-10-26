#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "shapes.h"
#include "network.h"

void generateOutput(std::__1::mt19937 &rng, ForwardOutput &forward, const Weights &weights, int iteration);

int main()
{
    std::mt19937 rng(1337u); // random generator
    

    Eigen::MatrixXf X = make_batch_downsampled(/*Batch size*/B, rng, ShapeType::Any);
    Weights weights;
    ForwardOutput forward;
    Gradients gradients;
    std::cout << "Loss no pass : ";forward.lossPrint();
    generateOutput(rng, forward, weights, 0);
    forwardPass(forward, weights, X);
    backPass(gradients, forward, weights, X);
    backProp(weights,gradients);
    // std::cout << "first pass : "; weights.print();

    std::cout << "Loss first pass : ";forward.lossPrint();

    for (size_t i = 0; i <= 15000; i++)
    {
        X = make_batch_downsampled(/*Batch size*/B, rng, ShapeType::Any);
        forwardPass(forward, weights, X);
        backPass(gradients, forward, weights, X);
        backProp(weights,gradients);
        if ( i % 100 == 0)
        {
            std::cout << "loss after :" << i << "iterations : "; forward.lossPrint();
        }
        if(i % 500 == 0)
        {
            generateOutput(rng, forward, weights, i);
        }
    }

    // std::cout << "After 100 iterations :"; weights.print();
    std::cout << "Loss after 100 iterations : ";forward.lossPrint();
    return 0;
}


void generateOutput(std::mt19937 &rng, ForwardOutput& forward, const Weights& weights, int iteration)
{
    // === Génération d'une image de sortie après 100 itérations ===
    Eigen::MatrixXf X_test = make_batch_downsampled(/*Batch size*/B, rng, ShapeType::Any);
    std::ostringstream inputPath;
    inputPath << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"
         << "INPUT_After_" << std::setw(5) << std::setfill('0') << iteration << ".png";

    // Passage forward final
    forwardPass(forward, weights, X_test);

    // Sauvegarde de la reconstruction
    std::ostringstream path;
    path << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"
         << "OUTPUT_After_" << std::setw(5) << std::setfill('0') << iteration << ".png";

    // Sauvegarde de la reconstruction
    bool input = write_png_grid(X_test, 4, 4, inputPath.str()) ;
    bool ok = write_png_grid(forward.sigmoid, 4, 4, path.str());

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