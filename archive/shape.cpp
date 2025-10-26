#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef USE_SYSTEM_EIGEN
  #include <Eigen/Dense>
#else
  #include <Eigen/Dense> // from third_party/eigen include dir
#endif

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

// Convert a batch (B x 256 floats in [0,1]) to a single PNG grid.
// gridCols * gridRows must be >= B. Each tile is 16x16.
// Output path: e.g., "assets/preview_shapes.png"
bool write_png_grid(const MatrixXf& batch, int gridCols, int gridRows, const std::string& outPath) {
    const int B = static_cast<int>(batch.rows());
    const int W = 16, H = 16;
    const int tileW = W, tileH = H;

    const int outW = gridCols * tileW;
    const int outH = gridRows * tileH;
    std::vector<unsigned char> img(outW * outH); // grayscale

    auto putTile = [&](int b, int gx, int gy) {
        // b-th row → one 16x16 tile
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float v = batch(b, y * W + x); // row-major flatten
                unsigned char u8 = static_cast<unsigned char>(std::round(clamp01(v) * 255.0f));
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

    // Write grayscale PNG (stride = outW)
    int ok = stbi_write_png(outPath.c_str(), outW, outH, /*comp=*/1, img.data(), outW);
    return ok != 0;
}

// ------------------------------------------------------------
// Geometry helpers

// Same-side test for triangle: sign of cross product
static inline float edge_sign(float px, float py, float ax, float ay, float bx, float by) {
    // (p - b) x (a - b)
    return (px - bx) * (ay - by) - (ax - bx) * (py - by);
}

enum class ShapeType { Circle = 0, Square = 1, Triangle = 2, Any = 3 };

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

// ------------------------------------------------------------
// Parameter sampling with mild randomness; keeps shapes on-canvas.

ShapeParams sample_params(std::mt19937& rng, ShapeType force = ShapeType::Any) {
    std::uniform_real_distribution<float> U01(0.f, 1.f);

    ShapeParams p{};
    if (force == ShapeType::Any) {
        int r = static_cast<int>(std::floor(U01(rng) * 3.0f)) % 3;
        p.type = static_cast<ShapeType>(r);
    } else {
        p.type = force;
    }

    // Centers around the middle (8,8) with mild variation
    // Keep within [5,11] to avoid truncation
    std::uniform_real_distribution<float> Uc(5.f, 11.f);
    p.cx = Uc(rng);
    p.cy = Uc(rng);

    if (p.type == ShapeType::Circle) {
        std::uniform_real_distribution<float> Ur(3.0f, 5.0f);
        p.radius = Ur(rng);
    } else if (p.type == ShapeType::Square) {
        std::uniform_real_distribution<float> Us(6.0f, 10.0f);
        p.side = Us(rng);
        // optional: slightly adjust center range based on side (already safe above)
    } else if (p.type == ShapeType::Triangle) {
        std::uniform_real_distribution<float> Ubw(6.0f, 10.0f);
        std::uniform_real_distribution<float> Uh(6.0f, 10.0f);
        p.tri_bw = Ubw(rng);
        p.tri_h  = Uh(rng);
    }
    return p;
}

// ------------------------------------------------------------
// Inside tests at a pixel center (px,py). Return coverage in [0,1].
// supersample = 1 (no AA), 2 (2x2), or 4 (4x4).

float coverage_circle(const ShapeParams& sp, float px, float py, int supersample) {
    auto inside = [&](float x, float y) {
        float dx = x - sp.cx;
        float dy = y - sp.cy;
        return (dx*dx + dy*dy) <= (sp.radius * sp.radius) ? 1.0f : 0.0f;
    };

    if (supersample <= 1) return inside(px, py);

    int S = supersample;
    float step = 1.0f / static_cast<float>(S + 1); // small jitter across pixel
    float base = -0.5f + step; // center-ish spread
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

float coverage_square(const ShapeParams& sp, float px, float py, int supersample) {
    float h = sp.side * 0.5f; // half size
    auto inside = [&](float x, float y) {
        return (std::fabs(x - sp.cx) <= h && std::fabs(y - sp.cy) <= h) ? 1.0f : 0.0f;
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

float coverage_triangle(const ShapeParams& sp, float px, float py, int supersample) {
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
        return !(has_neg && has_pos) ? 1.0f : 0.0f; // all same sign => inside
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

// ---- High-res (HR) → Low-res (LR) downsample by average pooling ----
static inline RowVectorXf downsample_64_to_16(const std::vector<float>& hr) {
    // hr is 64x64 in row-major, values in [0,1]
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

// ---- Render a single shape at 64x64, then downsample to 16x16 ----
RowVectorXf rasterize_one_downsampled(const ShapeParams& sp) {
    const int HR = 64;   // high-res canvas
    const float scale = 16.0f / float(HR); // map HR pixel centers into [0,16)
    std::vector<float> hr(HR * HR);

    // Fill HR buffer using existing coverage_* functions at supersample=1.
    for (int y = 0; y < HR; ++y) {
        float py = (y + 0.5f) * scale; // map to 16x16 coordinate space
        for (int x = 0; x < HR; ++x) {
            float px = (x + 0.5f) * scale;
            float v = 0.0f;
            switch (sp.type) {
                case ShapeType::Circle:   v = coverage_circle  (sp, px, py, /*supersample=*/1); break;
                case ShapeType::Square:   v = coverage_square  (sp, px, py, /*supersample=*/1); break;
                case ShapeType::Triangle: v = coverage_triangle(sp, px, py, /*supersample=*/1); break;
                default: break;
            }
            hr[y * HR + x] = clamp01(v);
        }
    }
    return downsample_64_to_16(hr);
}

// ------------------------------------------------------------
// One image (flattened 256-vector) for given params.
// supersample = 1 (no AA), 2 or 4 for smoother edges.

RowVectorXf rasterize_one(const ShapeParams& sp, int supersample = 1) {
    const int W = 16, H = 16;
    RowVectorXf row( W * H );
    int idx = 0;
    for (int y = 0; y < H; ++y) {
        float py = y + 0.5f;
        for (int x = 0; x < W; ++x) {
            float px = x + 0.5f;

            float v = 0.0f;
            switch (sp.type) {
                case ShapeType::Circle:   v = coverage_circle(sp, px, py, supersample); break;
                case ShapeType::Square:   v = coverage_square(sp, px, py, supersample); break;
                case ShapeType::Triangle: v = coverage_triangle(sp, px, py, supersample); break;
                default: v = 0.0f; break;
            }
            row(idx++) = clamp01(v);
        }
    }
    return row;
}

// ------------------------------------------------------------
// Batch generator: returns B x 256 matrix in [0,1].
// If force != Any, generates that shape only (useful for per-shape previews).

MatrixXf make_batch(int batch_size, std::mt19937& rng, ShapeType force = ShapeType::Any,
                    int supersample = 1) {
    MatrixXf X(batch_size, 16 * 16);
    for (int i = 0; i < batch_size; ++i) {
        ShapeParams sp = sample_params(rng, force);
        X.row(i) = rasterize_one(sp, supersample);
    }
    return X;
}

// ---- Batch generator using 64→16 downsampled rasterization ----
MatrixXf make_batch_downsampled(int batch_size, std::mt19937& rng,
                                ShapeType force = ShapeType::Any) {
    MatrixXf X(batch_size, 16 * 16);
    for (int i = 0; i < batch_size; ++i) {
        ShapeParams sp = sample_params(rng, force);
        X.row(i) = rasterize_one_downsampled(sp);  // 64→16 average-pool
    }
    return X;
}

// ------------------------------------------------------------
// Simple stats: mean pixel & ones count (useful quick QA)

struct BatchStats {
    float mean_pixel = 0.0f;
    int ones_total = 0;
};

BatchStats compute_stats(const MatrixXf& X) {
    BatchStats s;
    const int B = static_cast<int>(X.rows());
    const int D = static_cast<int>(X.cols());
    float sum = 0.0f;
    int ones = 0;
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < D; ++j) {
            float v = X(i, j);
            sum += v;
            if (v >= 0.5f) ++ones; // crude area proxy for binary images
        }
    }
    s.mean_pixel = sum / float(B * D);
    s.ones_total = ones;
    return s;
}

// ------------------------------------------------------------

int main() {
    // Reproducible RNG
    const uint32_t seed = 1337u;
    std::mt19937 rng(seed);

    // Mixed preview (16 samples → 4x4 grid)
    {
        MatrixXf X = make_batch_downsampled(16, rng, ShapeType::Any);
        bool ok = write_png_grid(X, /*gridCols=*/4, /*gridRows=*/4, "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/preview_mixed.png");
        if (!ok) {
            std::cerr << "Failed to write assets/preview_mixed.png\n";
            return 1;
        }
        BatchStats s = compute_stats(X);
        std::cout << "[mixed] mean_pixel=" << s.mean_pixel
                  << " ones_total=" << s.ones_total << "\n";
    }

    // Per-shape previews (force each type), with light supersampling
    {
        std::mt19937 rng2(seed); // reset seed so per-shape is reproducible too

        MatrixXf C = make_batch_downsampled(16, rng2, ShapeType::Circle);
        MatrixXf S = make_batch_downsampled(16, rng2, ShapeType::Square);
        MatrixXf T = make_batch_downsampled(16, rng2, ShapeType::Triangle);

        write_png_grid(C, 4, 4, "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/preview_circles.png");
        write_png_grid(S, 4, 4, "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/preview_squares.png");
        write_png_grid(T, 4, 4, "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/preview_triangles.png");

        auto sc = compute_stats(C);
        auto ss = compute_stats(S);
        auto st = compute_stats(T);

        std::ofstream log("logs/gen_stats.csv", std::ios::app);
        log << "seed,mean_mixed,ones_mixed\n";
        // Log mixed stats were printed above; log per-shape here:
        log << seed << ",(see stdout)," << "(see stdout)" << "\n";
        log << "circles_mean," << sc.mean_pixel << ",circles_ones," << sc.ones_total << "\n";
        log << "squares_mean," << ss.mean_pixel << ",squares_ones," << ss.ones_total << "\n";
        log << "triangles_mean," << st.mean_pixel << ",triangles_ones," << st.ones_total << "\n";
        std::cout << "[circles]  mean=" << sc.mean_pixel << " ones=" << sc.ones_total << "\n";
        std::cout << "[squares]  mean=" << ss.mean_pixel << " ones=" << ss.ones_total << "\n";
        std::cout << "[triangles] mean=" << st.mean_pixel << " ones=" << st.ones_total << "\n";
    }

    std::cout << "Wrote assets/preview_{mixed,circles,squares,triangles}.png\n";
    std::cout << "Done.\n";
    return 0;
}