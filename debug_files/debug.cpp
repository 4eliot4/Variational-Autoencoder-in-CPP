#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "shapes.h"
#include "network.h"

//
// Rename it debug when finished.
//
int main ()
{
    std::mt19937 rng(1337u); // random generator
    
    Eigen::RowVectorXf mean1xD = compute_dataset_mean(/*N=*/10000, rng, ShapeType::Any);
    Eigen::MatrixXf oneRow(1, 16*16);
    oneRow.row(0) = mean1xD;
    write_png_grid(oneRow, /*gridCols=*/1, /*gridRows=*/1,
                   "/Users/daboi/Documents/Projects/VAE/Intelligent_Data_Compression_Framework/assets/DATASET_MEAN.png");
    std::cout << "Saved DATASET_MEAN.png\n";

    return 0;
}
