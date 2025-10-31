#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>
#include <sstream>
#include <iomanip> // Required for std::setw, std::setfill

#include "shapes.h"
#include "network.h"

// Forward declaration of the output generation function.
// Changed std::__1::mt19937 to std::mt19937 for standard compliance.
void generateOutput(std::mt19937 &rng, ForwardOutput &forward, const Weights &weights, int iteration);

int main()
{
    // === Hyperparameters ===
    // Learning rate for gradient descent. Adjust to control the step size of weight updates.
    // A high learning rate can cause divergence; a low one can cause slow convergence or getting stuck in local minima.
    const float LEARNING_RATE = 0.001f; // Initial value, common starting point for neural networks. Reduced from typical 0.01 for stability.
    // Total number of training iterations. Each iteration processes one batch of data.
    const int NUM_ITERATIONS = 50000; // Increased iterations for extended training and observation.
    // Interval (in iterations) for printing training loss to the console.
    const int LOG_LOSS_INTERVAL = 100;
    // Interval (in iterations) for generating and saving sample input/output images.
    const int GENERATE_OUTPUT_INTERVAL = 1000; // Less frequent saving to manage file output and observe trends.

    std::mt19937 rng(1337u); // Random number generator, seeded for reproducibility.

    // Initialize a batch of input data. 'B' (Batch size) is assumed to be a globally defined constant
    // (e.g., from network.h or shapes.h). It determines the number of samples in one training step.
    Eigen::MatrixXf X = make_batch_downsampled(B, rng, ShapeType::Any);
    Weights weights;     // Stores the neural network's parameters (weights and biases).
    ForwardOutput forward; // Stores results from the forward pass (activations, loss).
    Gradients gradients; // Stores gradients computed during the backward pass.

    std::cout << "Starting training with the following hyperparameters:\n"
              << "  Learning Rate: " << LEARNING_RATE << "\n"
              << "  Batch Size (B): " << B << "\n" // Assuming B is visible and defined.
              << "  Number of Iterations: " << NUM_ITERATIONS << "\n"
              << "  Loss Logging Interval: " << LOG_LOSS_INTERVAL << "\n"
              << "  Output Generation Interval: " << GENERATE_OUTPUT_INTERVAL << "\n"
              << "---------------------------------------------------------" << std::endl;

    // Main training loop
    for (int i = 0; i < NUM_ITERATIONS; ++i)
    {
        // Generate a new batch of data for the current iteration.
        X = make_batch_downsampled(B, rng, ShapeType::Any);

        // Perform the forward pass: compute network output and intermediate activations.
        forwardPass(forward, weights, X);

        // Perform the backward pass: calculate gradients of the loss with respect to all weights.
        backPass(gradients, forward, weights, X);

        // Update network weights using the calculated gradients and the specified learning rate.
        // NOTE: This assumes the 'backProp' function in 'network.cpp' will be updated
        // to accept a learning rate parameter, as per the overall implementation plan.
        backProp(weights, gradients, LEARNING_RATE);

        // Log training loss and current iteration periodically.
        if (i % LOG_LOSS_INTERVAL == 0)
        {
            std::cout << "Iteration " << std::setw(6) << std::setfill('0') << i
                      << " | Current Loss: ";
            forward.lossPrint(); // Assuming lossPrint() outputs the loss value and potentially a newline.
        }

        // Generate and save sample input and reconstructed output images periodically.
        if (i > 0 && i % GENERATE_OUTPUT_INTERVAL == 0)
        {
            generateOutput(rng, forward, weights, i);
        }
    }

    // Final loss reporting after the training loop completes.
    std::cout << "\n---------------------------------------------------------\n"
              << "Training finished. Final loss after " << NUM_ITERATIONS << " iterations: ";
    // To report the final loss accurately, perform one last forward pass on a fresh batch.
    X = make_batch_downsampled(B, rng, ShapeType::Any);
    forwardPass(forward, weights, X);
    forward.lossPrint();
    std::cout << std::endl; // Ensure a newline after the final loss output.

    return 0;
}

// Function to generate and save sample input and reconstructed output images.
void generateOutput(std::mt19937 &rng, ForwardOutput& forward, const Weights& weights, int iteration)
{
    // Generate a new batch of test data.
    Eigen::MatrixXf X_test = make_batch_downsampled(B, rng, ShapeType::Any);
    std::ostringstream inputPath;

    // Construct the file path for the input image. Using original hardcoded path for consistency with existing style.
    inputPath << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"
              << "INPUT_After_" << std::setw(5) << std::setfill('0') << iteration << ".png";

    // Perform a forward pass with the test data to get the network's reconstruction.
    forwardPass(forward, weights, X_test);

    std::ostringstream outputPath;
    // Construct the file path for the reconstructed output image.
    outputPath << "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/"
               << "OUTPUT_After_" << std::setw(5) << std::setfill('0') << iteration << ".png";

    // Save the input batch as a grid image.
    bool input_save_ok = write_png_grid(X_test, 4, 4, inputPath.str());
    // Save the network's sigmoid output (reconstruction) as a grid image.
    bool output_save_ok = write_png_grid(forward.sigmoid, 4, 4, outputPath.str());

    // Report on the success or failure of saving the input image.
    if (!input_save_ok) {
        std::cerr << " Failed to write input image: " << inputPath.str() << "\n";
    } else {
        std::cout << "Saved input image: " << inputPath.str() << "\n";
    }

    // Report on the success or failure of saving the output image.
    if (!output_save_ok) {
        std::cerr << " Failed to write output image: " << outputPath.str() << "\n";
    } else {
        std::cout << "Saved output image: " << outputPath.str() << "\n";
    }
}