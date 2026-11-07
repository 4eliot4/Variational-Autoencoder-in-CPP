#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "shape.h"
#include "network.h"

void generateOutput(std::__1::mt19937 &rng, ForwardOutput &forward, const Weights &weights, int iteration);
void interpolation(Eigen::MatrixXf &X1, Eigen::MatrixXf &X2, const Weights &weights, ForwardOutput &forward, const double &t);
void generateRandomDigits(ForwardOutput &forward, const Weights &weights, std::mt19937 &rng);
void generateFromPosterior(ForwardOutput &forward, const Weights &weights, std::mt19937 &rng);
void generateFromCentroids(ForwardOutput &forward, const Weights &weights, std::mt19937 &rng);
std::vector<unsigned char> loadMnistLabels(const std::string &path);
size_t iterations = 3000;

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
        forwardPass(forward, weights, X,rng);
        backPass(gradients, forward, weights, X);
        //backProp(weights,gradients);
        backPropAdam(weights, gradients, opt);
        
        if ( i % 100 == 0)
        {
            std::cout << "loss after :" << i << "iterations : "; forward.lossPrint();
        }
        if(i % 500 == 0)
        {
            generateOutput(rng, forward, weights, i);
            std::ostringstream hPath;
            hPath << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"
                  << "H_latent_iter" << std::setw(5) << std::setfill('0') << i << ".csv";
            save_matrix_csv(forward.H,hPath.str());
            //save_matrix_images_csv(forward.H, hPath.str());
        }
    }

    // export for PCA Plotting
    auto labels = loadMnistLabels("/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/MNIST/train-labels.idx1-ubyte");

    std::ofstream latentFile("latent_means.csv");
    std::ofstream labelFile("latent_labels.csv");

    if (!latentFile || !labelFile)
        throw std::runtime_error("Cannot create CSV files");

    // Export 160 batches = 10,240 images
    for (int batch = 0; batch < 160; ++batch)
    {
        std::vector<int> indices;

        Eigen::MatrixXf X = make_batch_mnist(
            B, rng, true, &indices
        );

        X = (X.array() > 0.5f).cast<float>();

        forwardPass(forward, weights, X, rng);

        for (int i = 0; i < forward.M.rows(); ++i)
        {
            
            for (int j = 0; j < LATENT_SIZE; ++j)
            {
                latentFile << forward.M(i, j);

                if (j + 1 < LATENT_SIZE)
                    latentFile << ",";// Save the 32 latent means
            }

            latentFile << "\n";

            
            labelFile << int(labels.at(indices.at(i))) << "\n"; /// Save the matching MNIST label
        }
    }

    std::cout << "Latent CSV files saved\n";
    generateFromCentroids(forward, weights, rng);


    generateFromPosterior(forward, weights, rng);
    generateRandomDigits(forward, weights, rng);
    return 0;
}


void generateOutput(std::mt19937 &rng, ForwardOutput& forward, const Weights& weights, int iteration)
{
    // Load an image
    std::mt19937 rng1(1020u);
    Eigen::MatrixXf X_test = make_batch_mnist(B, rng1, true);
    //Eigen::MatrixXf X_test = make_single_image_batch(1234, B);
    //Eigen::MatrixXf X_test = make_two_images_batch(1234, 1235, B);
    X_test = (X_test.array() > 0.5f).cast<float>();
    std::ostringstream inputPath;
    inputPath << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"<< "INPUT_After_" << std::setw(5) << std::setfill('0') << iteration << ".png";

    forwardPass(forward, weights, X_test,rng);
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
/*
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
*/


void generateRandomDigits(ForwardOutput& forward,const Weights& weights,std::mt19937& rng)
{
    std::normal_distribution<float> normal(0.0f, 1.0f);

    forward.Z.resize(B, LATENT_SIZE);

    for (int i = 0; i < B; ++i)
    {
        for (int j = 0; j < LATENT_SIZE; ++j)
        {
            forward.Z(i, j) = normal(rng);
        }
    }

    
    decoder(forward, weights); // Generate images  from latent vectors

    // Save predicted pixel probabilities
    write_png_grid_mnist(forward.sigmoid, 8, 8, "generated_digits.png");
    Eigen::MatrixXf binary = (forward.sigmoid.array() > 0.5f).cast<float>();

    write_png_grid_mnist(binary, 8, 8, "generated_binary.png");
}


void generateFromPosterior(ForwardOutput& forward,const Weights& weights,std::mt19937& rng)
{

    Eigen::MatrixXf X = make_batch_mnist(B, rng, true);
    X = (X.array() > 0.5f).cast<float>();


    forwardPass(forward, weights, X, rng);

    write_png_grid_mnist(forward.sigmoid, 8, 8,"generated_from_posterior.png");
}


std::vector<unsigned char> loadMnistLabels(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file)
        throw std::runtime_error("Cannot open MNIST labels");

    auto read32 = [&file]() -> uint32_t
    {
        unsigned char b[4];

        if (!file.read(reinterpret_cast<char*>(b), 4))
            throw std::runtime_error("Invalid label header");

        return (uint32_t(b[0]) << 24)
             | (uint32_t(b[1]) << 16)
             | (uint32_t(b[2]) << 8)
             | uint32_t(b[3]);
    };

    uint32_t magic = read32();
    uint32_t count = read32();

    if (magic != 2049)
        throw std::runtime_error("Invalid MNIST label magic");

    std::vector<unsigned char> labels(count);

    if (!file.read(
        reinterpret_cast<char*>(labels.data()),
        count))
        throw std::runtime_error("Incomplete MNIST labels");

    return labels;
}



void generateFromCentroids(ForwardOutput& forward,const Weights& weights,std::mt19937& rng)
{
    std::ifstream file("digit_centroids.txt");

    if (!file)
        throw std::runtime_error("Cannot open centroids");

    Eigen::MatrixXf centroids(10, LATENT_SIZE);

    for (int k = 0; k < 10; ++k)
        for (int j = 0; j < LATENT_SIZE; ++j)
            if (!(file >> centroids(k, j)))
                throw std::runtime_error("Invalid centroids file");

    forward.Z.resize(B, LATENT_SIZE);

    std::normal_distribution<float> normal(0.0f, 1.0f);

    const float variation = 0.3f;

    for (int i = 0; i < B; ++i)
    {
        int digit = i % 10;

        for (int j = 0; j < LATENT_SIZE; ++j)
        {
            forward.Z(i, j) = centroids(digit, j) + variation * normal(rng);
        }
    }

    decoder(forward, weights);

    write_png_grid_mnist(
        forward.sigmoid,
        8, 8,
        "generated_centroids.png"
    );
}
