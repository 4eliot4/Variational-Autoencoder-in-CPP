#include "shapes.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// stb_image_write MUST be implemented in exactly one .cpp file.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/third_party/stb/stb_image_write.h"

using Eigen::MatrixXf;
using Eigen::RowVectorXf;

// ------------------------------------------------------------
// Utility: clamp to [0,1]
static inline float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

// ------------------------------------------------------------
// Geometry helpers

// Same-side test for triangle: sign of cross product
static inline float edge_sign(float px, float py,
                              float ax, float ay,
                              float bx, float by)
{
    // (p - b) x (a - b)
    return (px - bx) * (ay - by) - (ax - bx) * (py - by);
}

// Internal shape parameter struct (not exposed in header)
struct ShapeParams {
    ShapeType type;
    // Common center
    float cx, cy;
    // Circle
    float radius;
    // Square
    float side;
    // Triangle (isosceles): base width & height; vertices derived from (cx, cy)
    float tri_bw, tri_h;
};

// Sample random shape parameters with mild jitter.
// Keeps most shapes on-canvas.
static ShapeParams sample_params(std::mt19937& rng,
                                 ShapeType force = ShapeType::Any)
{
    std::uniform_real_distribution<float> U01(0.f, 1.f);

    ShapeParams p{};
    if (force == ShapeType::Any) {
        int r = static_cast<int>(std::floor(U01(rng) * 3.0f)) % 3;
        p.type = static_cast<ShapeType>(r);
    } else {
        p.type = force;
    }

    

    if (p.type == ShapeType::Circle) {
        std::uniform_real_distribution<float> Ur(3.8f, 4.2f);
        p.radius = Ur(rng);

        std::uniform_real_distribution<float> Ucx(-p.radius, 16.0f + p.radius);
        std::uniform_real_distribution<float> Ucy(-p.radius, 16.0f + p.radius);
        p.cx = Ucx(rng);
        p.cy = Ucy(rng);

    } else if (p.type == ShapeType::Square) {
        std::uniform_real_distribution<float> Us(6.0f, 10.0f);
        p.side = Us(rng);
        float half = 0.5f * p.side;
        std::uniform_real_distribution<float> Ucx(-half, 16.0f + half);
        std::uniform_real_distribution<float> Ucy(-half, 16.0f + half);
        p.cx = Ucx(rng);
        p.cy = Ucy(rng);

    } else if (p.type == ShapeType::Triangle) {
        std::uniform_real_distribution<float> Ubw(6.0f, 10.0f);
        std::uniform_real_distribution<float> Uh(6.0f, 10.0f);
        p.tri_bw = Ubw(rng);
        p.tri_h  = Uh(rng);

        float halfW = 0.5f * p.tri_bw;
        float halfH = 0.5f * p.tri_h;
        std::uniform_real_distribution<float> Ucx(-halfW, 16.0f + halfW);
        std::uniform_real_distribution<float> Ucy(-halfH, 16.0f + halfH);
        p.cx = Ucx(rng);
        p.cy = Ucy(rng);
    }
    return p;
}

// ------------------------------------------------------------
// Inside tests at a pixel center (px,py). Return coverage in [0,1].
// supersample = 1 (no AA), 2 (2x2), or 4 (4x4).

static float coverage_circle(const ShapeParams& sp,
                             float px, float py,
                             int supersample)
{
    auto inside = [&](float x, float y) {
        float dx = x - sp.cx;
        float dy = y - sp.cy;
        return (dx*dx + dy*dy) <= (sp.radius * sp.radius) ? 1.0f : 0.0f;
    };

    if (supersample <= 1) return inside(px, py);

    int S = supersample;
    float step = 1.0f / static_cast<float>(S + 1); // jitter across pixel
    float base = -0.5f + step;
    float acc = 0.0f;
    for (int sy = 0; sy < S; ++sy) {
        for (int sx = 0; sx < S; ++sx) {
            float ox = base + sx * step * 2.0f;
            float oy = base + sy * step * 2.0f;
            acc += inside(px + ox * 0.5f, py + oy * 0.5f);
        }
    }
    return acc / float(S * S);
}

static float coverage_square(const ShapeParams& sp,
                             float px, float py,
                             int supersample)
{
    float h = sp.side * 0.5f; // half size
    auto inside = [&](float x, float y) {
        return (std::fabs(x - sp.cx) <= h && std::fabs(y - sp.cy) <= h)
            ? 1.0f : 0.0f;
    };

    if (supersample <= 1) return inside(px, py);

    int S = supersample;
    float step = 1.0f / static_cast<float>(S + 1);
    float base = -0.5f + step;
    float acc = 0.0f;
    for (int sy = 0; sy < S; ++sy) {
        for (int sx = 0; sx < S; ++sx) {
            float ox = base + sx * step * 2.0f;
            float oy = base + sy * step * 2.0f;
            acc += inside(px + ox * 0.5f, py + oy * 0.5f);
        }
    }
    return acc / float(S * S);
}

static float coverage_triangle(const ShapeParams& sp,
                               float px, float py,
                               int supersample)
{
    // Upward isosceles triangle:
    // v1 = top, v2 = bottom-left, v3 = bottom-right
    float vx1 = sp.cx;
    float vy1 = sp.cy - sp.tri_h * 0.5f;
    float vx2 = sp.cx - sp.tri_bw * 0.5f;
    float vy2 = sp.cy + sp.tri_h * 0.5f;
    float vx3 = sp.cx + sp.tri_bw * 0.5f;
    float vy3 = sp.cy + sp.tri_h * 0.5f;

    auto inside = [&](float x, float y) {
        float s1 = edge_sign(x, y, vx1, vy1, vx2, vy2);
        float s2 = edge_sign(x, y, vx2, vy2, vx3, vy3);
        float s3 = edge_sign(x, y, vx3, vy3, vx1, vy1);
        bool has_neg = (s1 < 0.0f) || (s2 < 0.0f) || (s3 < 0.0f);
        bool has_pos = (s1 > 0.0f) || (s2 > 0.0f) || (s3 > 0.0f);
        // inside if all cross products have the same sign
        return !(has_neg && has_pos) ? 1.0f : 0.0f;
    };

    if (supersample <= 1) return inside(px, py);

    int S = supersample;
    float step = 1.0f / static_cast<float>(S + 1);
    float base = -0.5f + step;
    float acc = 0.0f;
    for (int sy = 0; sy < S; ++sy) {
        for (int sx = 0; sx < S; ++sx) {
            float ox = base + sx * step * 2.0f;
            float oy = base + sy * step * 2.0f;
            acc += inside(px + ox * 0.5f, py + oy * 0.5f);
        }
    }
    return acc / float(S * S);
}

// ------------------------------------------------------------
// Downsample 64x64 -> 16x16 by average pooling, row-major.
static inline RowVectorXf downsample_64_to_16(const std::vector<float>& hr)
{
    const int HR = 64, LR = 16, R = HR / LR; // 4
    RowVectorXf lr(LR * LR);
    int idx = 0;
    for (int y = 0; y < LR; ++y) {
        for (int x = 0; x < LR; ++x) {
            float acc = 0.0f;
            for (int yy = 0; yy < R; ++yy) {
                for (int xx = 0; xx < R; ++xx) {
                    int hx = x * R + xx;
                    int hy = y * R + yy;
                    acc += hr[hy * HR + hx];
                }
            }
            lr(idx++) = acc / float(R * R);
        }
    }
    return lr;
}

// Render one shape at 64x64, then average-pool down to 16x16.
static RowVectorXf rasterize_one_downsampled(const ShapeParams& sp)
{
    const int HR = 64;
    const float scale = 16.0f / float(HR); // map HR pixel centers into [0,16)
    std::vector<float> hr(HR * HR);

    for (int y = 0; y < HR; ++y) {
        float py = (y + 0.5f) * scale; // pixel center in 16x16 space
        for (int x = 0; x < HR; ++x) {
            float px = (x + 0.5f) * scale;
            float v = 0.0f;
            switch (sp.type) {
                case ShapeType::Circle:
                    v = coverage_circle(sp, px, py, /*supersample=*/2);
                    break;
                case ShapeType::Square:
                    v = coverage_square(sp, px, py, /*supersample=*/2);
                    break;
                case ShapeType::Triangle:
                    v = coverage_triangle(sp, px, py, /*supersample=*/2);
                    break;
                default:
                    break;
            }
            hr[y * HR + x] = clamp01(v);
        }
    }
    return downsample_64_to_16(hr);
}

// ------------------------------------------------------------
// Public API implementations
// ------------------------------------------------------------

MatrixXf make_batch_downsampled(int batch_size,
                                std::mt19937& rng,
                                ShapeType force)
{
    MatrixXf X(batch_size, 16 * 16);
    const float MIN_VISIBLE = 12.0f;  // ~12 pixels worth; tune 8–30
    for (int i = 0; i < batch_size; ++i) 
    {
        RowVectorXf img;
        int tries = 0;
        while (true) {
            ShapeParams sp = sample_params(rng, force);     // your current sampler (cropping allowed)
            img = rasterize_one_downsampled(sp);
            float visible = img.sum();  // [0,256], approximates visible area
            if (visible >= MIN_VISIBLE || tries++ > 20) break;  // retry a few times
    }
    X.row(i) = img;
    }
    
    return X;
}

bool write_png_grid(const MatrixXf& batch,
                    int gridCols,
                    int gridRows,
                    const std::string& outPath)
{
    const int B = static_cast<int>(batch.rows());
    const int W = 16, H = 16;
    const int tileW = W, tileH = H;

    const int outW = gridCols * tileW;
    const int outH = gridRows * tileH;
    std::vector<unsigned char> img(outW * outH); // grayscale

    auto putTile = [&](int b, int gx, int gy) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float v = batch(b, y * W + x);
                unsigned char u8 =
                    static_cast<unsigned char>(std::round(clamp01(v) * 255.0f));
                int outX = gx * tileW + x;
                int outY = gy * tileH + y;
                img[outY * outW + outX] = u8;
            }
        }
    };

    int b = 0;
    for (int gy = 0; gy < gridRows; ++gy) {
        for (int gx = 0; gx < gridCols; ++gx) {
            if (b < B) {
                putTile(b, gx, gy);
                ++b;
            }
        }
    }

    // 1-channel PNG
    int ok = stbi_write_png(outPath.c_str(),
                            outW, outH,
                            /*comp=*/1,
                            img.data(),
                            outW /*stride*/);
    return ok != 0;
}

BatchStats compute_stats(const MatrixXf& X)
{
    BatchStats s;
    const int B = static_cast<int>(X.rows());
    const int D = static_cast<int>(X.cols());
    float sum = 0.0f;
    int ones = 0;

    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < D; ++j) {
            float v = X(i, j);
            sum += v;
            if (v >= 0.5f) ++ones; // crude "area"
        }
    }

    s.mean_pixel = sum / float(B * D);
    s.ones_total = ones;
    return s;
}

Eigen::RowVectorXf compute_dataset_mean(int N, std::mt19937& rng, ShapeType force)
{
    Eigen::RowVectorXf mean = Eigen::RowVectorXf::Zero(16*16);
    const int chunk = 500; // accumulate in chunks to avoid huge mem
    int remaining = N;
    while (remaining > 0) {
        int b = std::min(chunk, remaining);
        Eigen::MatrixXf X = make_batch_downsampled(b, rng, force);
        mean += X.colwise().mean();   // add batch mean
        remaining -= b;
    }
    mean /= float( (N + chunk - 1) / chunk ); // average of batch-means
    return mean;
}