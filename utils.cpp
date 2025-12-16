#include "utils.h"

namespace hpc{

Tensor3D::Tensor3D(int n, double hx, double hy, double hz)
    : dim1(n), dim2(n), dim3(n), values(n * n * n, 0.0), h_x(hx), h_y(hy), h_z(hz) {}

Tensor3D::Tensor3D(int d1, int d2, int d3, double hx, double hy, double hz)
    : dim1(d1), dim2(d2), dim3(d3), h_x(hx), h_y(hy), h_z(hz), values(d1 * d2 * d3, 0.0) {}

double& Tensor3D::operator()(int i, int j, int k) {
    return values[(i * dim2 + j) * dim3 + k];
}

const double& Tensor3D::operator()(int i, int j, int k) const {
    return values[(i * dim2 + j) * dim3 + k];
}

double* Tensor3D::data() {
    return values.data();
}

const double* Tensor3D::data() const {
    return values.data();
}

double laplace_7point(const Tensor3D& u, int i, int j, int k) {
    double d2u_dx2 = u(i - 1, j, k) - 2.0f * u(i, j, k) + u(i + 1, j, k);
    double d2u_dy2 = u(i, j - 1, k) - 2.0f * u(i, j, k) + u(i, j + 1, k);
    double d2u_dz2 = u(i, j, k - 1) - 2.0f * u(i, j, k) + u(i, j, k + 1);

   
    return d2u_dx2 / std::pow(u.h_x, 2) + d2u_dy2 / std::pow(u.h_y, 2) + d2u_dz2 / std::pow(u.h_z, 2);
}

double laplace_7point_callable(double (*u)(double, double, double), double x, double y, double z, double h_x, double h_y, double h_z) {
    double d2u_dx2 = u(x - h_x, y, z) - 2.0f * u(x, y, z) + u(x + h_x, y, z);
    double d2u_dy2 = u(x, y - h_y, z) - 2.0f * u(x, y, z) + u(x, y + h_y, z);
    double d2u_dz2 = u(x, y, z - h_z) - 2.0f * u(x, y, z) + u(x, y, z + h_z);

    return d2u_dx2 / std::pow(h_x, 2) + d2u_dy2 / std::pow(h_y, 2) + d2u_dz2 / std::pow(h_z, 2);
}

double exact_sol(double x, double y, double z, double t){
    return sin(M_PI * x / L_x) * sin(M_PI * y / L_y) * sin(M_PI * z / L_z) * cos(M_PI * t);
}

double initial_cond(double x, double y, double z){
    return exact_sol(x, y, z, 0);
}

}
