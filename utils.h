#pragma once
#include <vector>
#include <cmath>
#include <omp.h>

#define L_x 1.0f
#define L_y L_x
#define L_z L_x
#define A_sq (std::pow(L_x * L_y * L_z, 2) / \
              (std::pow(L_x, 2) * std::pow(L_y, 2) + \
               std::pow(L_y, 2) * std::pow(L_z, 2) + \
               std::pow(L_x, 2) * std::pow(L_z, 2)))

namespace hpc{

class Tensor3D {
private:
    std::vector<double> values;
public:
    int dim1, dim2, dim3;
    double h_x, h_y, h_z;
    Tensor3D(int n, double hx, double hy, double hz);
    Tensor3D(int d1, int d2, int d3, double hx, double hy, double hz);

    double& operator()(int i, int j, int k);
    const double& operator()(int i, int j, int k) const;
    double* data();
    const double* data() const;
};

double laplace_7point(const Tensor3D& u, int i, int j, int k);

double laplace_7point_callable(double (*u)(double, double, double), double x, double y, double z, double h_x, double h_y, double h_z);

double exact_sol(double x, double y, double z, double t);

double initial_cond(double x, double y, double z);

}
