#ifndef SHAPES_DATASET_H_
#define SHAPES_DATASET_H_

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

// Public aliases for convenience
using Eigen::MatrixXf;
using Eigen::RowVectorXf;

// ------------------------------------------------------------
// Shape kind
enum class ShapeType {
    Circle   = 0,
    Square   = 1,
    Triangle = 2,
    Any      = 3
};

// Parameters describing one shape instance
struct ShapeParams {
    ShapeType type;

    // Common center
    float cx, cy;

    // Circle
    float radius;

    // Square
    float side;

    // Triangle (isosceles, pointing "up"):
    // base width and height; vertices are derived from (cx, cy)
    float tri_bw;
    float tri_h;
};

// ------------------------------------------------------------
// Simple stats for a batch of images
struct BatchStats {
    float mean_pixel = 0.0f; // average grayscale value over batch
    int   ones_total = 0;    // count of pixels >= 0.5 across batch
};

// ------------------------------------------------------------
// Utility helpers (no allocation-heavy work, can be inline if desired)

// Clamp float to [0,1]
float clamp01(float v);

// ------------------------------------------------------------
// Parameter sampling / shape generation

// Randomly sample ShapeParams. If 'force' != ShapeType::Any,
// always generate that specific shape. Keeps shapes mostly on-canvas.
ShapeParams sample_params(std::mt19937& rng,
                          ShapeType force = ShapeType::Any);

// ------------------------------------------------------------
// Coverage / rasterization

// Compute pixel coverage of each primitive at (px, py).
// supersample = 1 (no AA), 2 (2x2), 4 (4x4).
// Return value is grayscale in [0,1].
//
// These are declared in case callers want per-pixel queries, but you
// can omit them from the public API if you want them private.
float coverage_circle   (const ShapeParams& sp,
                         float px, float py,
                         int supersample);

float coverage_square   (const ShapeParams& sp,
                         float px, float py,
                         int supersample);

float coverage_triangle (const ShapeParams& sp,
                         float px, float py,
                         int supersample);

// Rasterize one shape directly at 16x16 with optional supersampling,
// flattened row-major (size 256). Values are in [0,1].
RowVectorXf rasterize_one(const ShapeParams& sp,
                          int supersample = 1);

// Rasterize one shape on a 64x64 hi-res canvas, then average-pool
// down to 16x16 (anti-aliased style). Returns 256-vector in [0,1].
RowVectorXf rasterize_one_downsampled(const ShapeParams& sp);

// ------------------------------------------------------------
// Downsampling helper

// Take a 64x64 grayscale image (row-major, values in [0,1])
// and average-pool 4x4 → 16x16. Returns 256-float RowVectorXf.
RowVectorXf downsample_64_to_16(const std::vector<float>& hr);

// ------------------------------------------------------------
// Batch generation

// Generate a batch of B images (B x 256) using direct rasterization.
// 'supersample' forwarded to rasterize_one().
// If 'force' != Any, restricts to that shape.
MatrixXf make_batch(int batch_size,
                    std::mt19937& rng,
                    ShapeType force = ShapeType::Any,
                    int supersample = 1);

// Generate a batch of B images (B x 256) using hi-res→low-res pipeline.
MatrixXf make_batch_downsampled(int batch_size,
                                std::mt19937& rng,
                                ShapeType force = ShapeType::Any);

// ------------------------------------------------------------
// Batch statistics

// Compute quick sanity stats for a batch:
//  - mean grayscale over all pixels in all images
//  - number of "on" pixels (>=0.5 threshold)
BatchStats compute_stats(const MatrixXf& X);

// ------------------------------------------------------------
// I/O helpers

// Write a batch (B x 256 floats in [0,1]) to a single grayscale PNG grid.
// gridCols * gridRows must be >= B.
// Each tile is interpreted as a 16x16 image.
// Returns true on success.
// NOTE: The implementation will use stb_image_write; only declare here.
bool write_png_grid(const MatrixXf& batch,
                    int gridCols,
                    int gridRows,
                    const std::string& outPath);

#endif  // SHAPES_DATASET_H_