#pragma once
#include <Eigen/Dense>
#include <random>
#include <string>

// Which shape to draw
enum class ShapeType { Circle = 0, Square = 1, Triangle = 2, Any = 3 };

// Simple QA stats on a batch
struct BatchStats {
    float mean_pixel;
    int   ones_total;
};

// Generate a batch of synthetic 16x16 grayscale images.
// Each row of the returned matrix is one flattened 16x16 image (256 floats in [0,1]).
// Arguments:
//   batch_size : number of samples to generate
//   rng        : PRNG state (std::mt19937) passed by reference so it's reproducible + advances
//   force      : if not ShapeType::Any, force that specific shape for all rows (debug/visualization)
// Returns:
//   Eigen::MatrixXf of shape (batch_size, 256)
Eigen::MatrixXf make_batch_downsampled(
    int batch_size,
    std::mt19937& rng,
    ShapeType force = ShapeType::Any
);

// Write a batch (B x 256) into a PNG grid image for inspection.
//   batch     : (B x 256) in [0,1]
//   gridCols  : how many tiles horizontally
//   gridRows  : how many tiles vertically
//   outPath   : filesystem path for the PNG
// returns true if write succeeded
bool write_png_grid(const Eigen::MatrixXf& batch,
                    int gridCols,
                    int gridRows,
                    const std::string& outPath);

// Compute quick stats about a batch (mean brightness, approx "area")
// returns struct BatchStats { mean_pixel, ones_total }
BatchStats compute_stats(const Eigen::MatrixXf& X);

// MNIST READER
Eigen::MatrixXf make_batch_mnist(int batch_size, std::mt19937 &rng, bool use_training_set);

Eigen::RowVectorXf compute_dataset_mean(int N, std::mt19937& rng, ShapeType force = ShapeType::Any);