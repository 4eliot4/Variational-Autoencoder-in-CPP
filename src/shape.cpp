#include "shape.h"
#include "network.h"

#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <stdexcept>

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

static Eigen::MatrixXf load_idx3_images(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);

    if (!f)
        throw std::runtime_error(
            "Cannot open image file: " + path
        );

    uint32_t magic = read_uint32(f);
    uint32_t n_images = read_uint32(f);
    uint32_t n_rows = read_uint32(f);
    uint32_t n_cols = read_uint32(f);

    if (magic != 2051)
        throw std::runtime_error("Invalid IDX image format");

    if (n_rows != IMAGE_SIZE || n_cols != IMAGE_SIZE)
        throw std::runtime_error(
            "Dataset resolution does not match IMAGE_SIZE"
        );

    Eigen::MatrixXf X(n_images, INPUT_SIZE);
    std::vector<unsigned char> buffer(INPUT_SIZE);

    for (uint32_t i = 0; i < n_images; ++i)
    {
        if (!f.read(
            reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(buffer.size())))
        {
            throw std::runtime_error(
                "Incomplete image data: " + path
            );
        }

        for (int j = 0; j < INPUT_SIZE; ++j)
        {
            X(i, j) = float(buffer[j]) / 255.0f;
        }
    }

    return X;
}



// ------------------------------------------------------------
// Load MNIST dataset once
// ------------------------------------------------------------
static void load_dataset()
{
    if (g_loaded) return;

    std::cout << "Loading Dataset...\n";

    g_train_images = load_idx3_images("/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/data/quickdraw64/train-images.idx3-ubyte");
    g_test_images = load_idx3_images("/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/data/quickdraw64/test-images.idx3-ubyte");

    g_loaded = true;

    std::cout << "Dataset loaded: "
              << g_train_images.rows() << " train, "
              << g_test_images.rows() << " test images.\n";
}


// ------------------------------------------------------------
// Public: sample a batch of MNIST images
// ------------------------------------------------------------

Eigen::MatrixXf make_batch_mnist(int batch_size,std::mt19937& rng,bool use_train,std::vector<int>* sampledIndices)
{
    load_dataset();

    const Eigen::MatrixXf& Xsrc =
        use_train ? g_train_images : g_test_images;

    std::uniform_int_distribution<int> U(0, Xsrc.rows() - 1);

    Eigen::MatrixXf X(batch_size, INPUT_SIZE);

    if (sampledIndices != nullptr)
    {
        sampledIndices->clear();
        sampledIndices->reserve(batch_size);
    }

    for (int i = 0; i < batch_size; ++i)
    {
        int idx = U(rng);

        X.row(i) = Xsrc.row(idx);

        if (sampledIndices != nullptr)
            sampledIndices->push_back(idx);
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
    const int W = IMAGE_SIZE;
    const int H = IMAGE_SIZE;

    if (batch.cols() != INPUT_SIZE)
        throw std::invalid_argument("PNG grid: incorrect image dimensions");

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

// ------------------------------------------------------------
// Make a batch for one digit overfit test
// ------------------------------------------------------------
Eigen::MatrixXf make_single_image_batch(int index, int batch_size)
{
    // Make sure MNIST is loaded
    load_dataset();

    const Eigen::MatrixXf& Xsrc = g_train_images;  // this is your "mnist_train"
    assert(index >= 0 && index < Xsrc.rows()); // only in debug version of make

    Eigen::MatrixXf X(batch_size, Xsrc.cols());
    Eigen::RowVectorXf img = Xsrc.row(index);

    for (int i = 0; i < batch_size; ++i) {
        X.row(i) = img;   // repeat the SAME image
    }
    return X;
}

// ------------------------------------------------------------
// Make a batch for two digits overfit test
// ------------------------------------------------------------
Eigen::MatrixXf make_two_images_batch(int index1, int index2,int batch_size)
{
    load_dataset();

    const Eigen::MatrixXf& Xsrc = g_train_images;
    assert(index1 >= 0 && index1 < Xsrc.rows()); // only in debug version of make
    assert(index2 >= 0 && index2 < Xsrc.rows()); // only in debug version of make

    Eigen::MatrixXf X(batch_size, Xsrc.cols());
    Eigen::RowVectorXf img1 = Xsrc.row(index1);
    Eigen::RowVectorXf img2 = Xsrc.row(index2);

    for (int i = 0; i < batch_size; ++i) 
        (i % 2 == 0) ? X.row(i) = img1 : X.row(i) = img2;
    return X;
}