#include "shape.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/third_party/stb/stb_image_write.h"


// ------------------------------------------------------------
// Global MNIST storage (lazy-loaded)
// ------------------------------------------------------------
static Eigen::MatrixXf g_train_images;
static Eigen::MatrixXf g_test_images;
static bool g_loaded = false;


// ------------------------------------------------------------
// Helper: read big-endian 32-bit int
// ------------------------------------------------------------
static uint32_t read_uint32(std::ifstream &f)
{
    unsigned char b[4];
    f.read((char*)b, 4);
    return (uint32_t(b[0]) << 24) |
           (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) << 8 ) |
            uint32_t(b[3]);
}

// ------------------------------------------------------------
// Load MNIST IDX3 image file
// ------------------------------------------------------------
static Eigen::MatrixXf load_idx3_images(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "ERROR: cannot open MNIST file: " << path << "\n";
        exit(1);
    }

    uint32_t magic   = read_uint32(f);
    uint32_t n_images= read_uint32(f);
    uint32_t n_rows  = read_uint32(f);
    uint32_t n_cols  = read_uint32(f);

    assert(magic == 2051); // Magic number for MNIST images
    assert(n_rows == 28 && n_cols == 28);

    Eigen::MatrixXf X(n_images, 28*28);

    std::vector<unsigned char> buffer(28*28);

    for (uint32_t i = 0; i < n_images; ++i) {
        f.read((char*)buffer.data(), 28*28);
        for (int j = 0; j < 28*28; ++j) {
            X(i, j) = float(buffer[j]) / 255.0f; // normalize
        }
    }
    return X;
}


// ------------------------------------------------------------
// Load MNIST dataset once
// ------------------------------------------------------------
static void load_mnist()
{
    if (g_loaded) return;

    std::cout << "Loading MNIST...\n";

    g_train_images = load_idx3_images("/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/MNIST/train-images.idx3-ubyte");
    g_test_images  = load_idx3_images("/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/MNIST/t10k-images.idx3-ubyte");

    g_loaded = true;

    std::cout << "MNIST loaded: "
              << g_train_images.rows() << " train, "
              << g_test_images.rows() << " test images.\n";
}


// ------------------------------------------------------------
// Public: sample a batch of MNIST images
// ------------------------------------------------------------
Eigen::MatrixXf make_batch_mnist(int batch_size,
                                 std::mt19937 &rng,
                                 bool use_train)
{
    load_mnist();

    const Eigen::MatrixXf &Xsrc = use_train ? g_train_images : g_test_images;

    std::uniform_int_distribution<int> U(0, Xsrc.rows() - 1);

    Eigen::MatrixXf X(batch_size, 28*28);

    for (int i = 0; i < batch_size; ++i) {
        int idx = U(rng);
        X.row(i) = Xsrc.row(idx);
    }
    return X;
}


// ------------------------------------------------------------
// Save MNIST images in a grid (like your shapes saver)
// ------------------------------------------------------------
bool write_png_grid_mnist(const Eigen::MatrixXf &batch,
                          int gridCols,
                          int gridRows,
                          const std::string &outPath)
{
    const int B = batch.rows();
    const int W = 28, H = 28;

    const int outW = gridCols * W;
    const int outH = gridRows * H;

    std::vector<unsigned char> img(outW * outH);

    auto putTile = [&](int b, int gx, int gy) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {

                float v = batch(b, y * W + x);
                unsigned char u8 = (unsigned char)(std::round(std::clamp(v, 0.0f, 1.0f) * 255.0f));

                int outX = gx * W + x;
                int outY = gy * H + y;

                img[outY * outW + outX] = u8;
            }
        }
    };

    int b = 0;
    for (int gy = 0; gy < gridRows; ++gy) {
        for (int gx = 0; gx < gridCols; ++gx) {
            if (b < B)
                putTile(b++, gx, gy);
        }
    }

    int ok = stbi_write_png(outPath.c_str(), outW, outH, 1, img.data(), outW);
    return ok != 0;
}