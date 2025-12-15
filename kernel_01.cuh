#pragma once

#ifdef __cplusplus
extern "C" {
#endif
void launch_init(
    double*& d0,
    double*& u1,
    int Ni, int Nj, int Nk,
    int istart, int jstart, int kstart,
    double hx, double hy, double hz,
    double A_sq_tau, double tau,
    double& err0, double& err1
);
#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
__global__ void init_kernel(
    double* u0, double* u1,
    int Ni, int Nj, int Nk,
    int istart, int jstart, int kstart,
    double hx, double hy, double hz,
    double A_sq_tau, double tau,
    double* block_err0, double* block_err1
);
#endif

void launch_update(
    const double* u0_h,
    const double* u1_h,
    double* u_h,
    int Ni, int Nj, int Nk,
    int istart, int jstart, int kstart,
    double hx, double hy, double hz,
    double A_sq_tau,
    double tn,
    double &local_err
);

#ifdef __CUDACC__
__global__ void update_kernel(
    const double *u0,
    const double *u1,
    double *u,
    int Ni, int Nj, int Nk,
    int istart, int jstart, int kstart,
    double hx, double hy, double hz,
    double A_sq_tau,
    double tn,
    double *block_err
);
#endif

