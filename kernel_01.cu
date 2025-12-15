#include "kernel_01.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include "utils.h"


__device__ double d_exact_sol(double x,double y,double z,double t) {                                               
    return sin(M_PI * x / L_x) * sin(M_PI * y / L_y) * sin(M_PI * z / L_z) * cos(M_PI * t); 
} 

__device__ double d_initial_cond(double x,double y,double z) {
   return d_exact_sol(x, y, z, 0);
}

__device__ double d_laplace(double (*u)(double, double, double), double x, double y, double z,
                            double h_x, double h_y, double h_z) {
    double d2u_dx2 = u(x - h_x, y, z) - 2.0f * u(x, y, z) + u(x + h_x, y, z);
    double d2u_dy2 = u(x, y - h_y, z) - 2.0f * u(x, y, z) + u(x, y + h_y, z);
    double d2u_dz2 = u(x, y, z - h_z) - 2.0f * u(x, y, z) + u(x, y, z + h_z);                                         
    return d2u_dx2 / std::pow(h_x, 2) + d2u_dy2 / std::pow(h_y, 2) + d2u_dz2 / std::pow(h_z, 2);
}

__device__ double d_laplace_7pt(const double *u1, int i, int j, int k,
    int Ni, int Nj, int Nk, double hx, double hy, double hz
){
    int idx = (i*Nj + j)*Nk + k;

    double xc = u1[idx];
    double xm = u1[((i-1)*Nj + j)*Nk + k];
    double xp = u1[((i+1)*Nj + j)*Nk + k];
    double ym = u1[(i*Nj + (j-1))*Nk + k];
    double yp = u1[(i*Nj + (j+1))*Nk + k];
    double zm = u1[(i*Nj + j)*Nk + (k-1)];
    double zp = u1[(i*Nj + j)*Nk + (k+1)];

    return (xp - 2.0*xc + xm)/(hx*hx)
         + (yp - 2.0*xc + ym)/(hy*hy)
         + (zp - 2.0*xc + zm)/(hz*hz);
}


__global__ void init_kernel(
    double* u0, double* u1,
    int Ni, int Nj, int Nk,
    int istart, int jstart, int kstart,
    double hx, double hy, double hz,
    double A_sq_tau, double tau,
    double* err0, double* err1
)
{
    
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;

    if (ii >= Ni-2 || jj >= Nj-2 || kk >= Nk-2) return; 

    int i = ii + 1;
    int j = jj + 1;
    int k = kk + 1;

    int ig = istart + ii;
    int jg = jstart + jj;
    int kg = kstart + kk;

    double x = ig * hx;
    double y = jg * hy;
    double z = kg * hz;

    double u0v = d_initial_cond(x, y, z);
    double lap = d_laplace(d_initial_cond, x, y, z, hx, hy, hz);
    double u1v = u0v + 0.5 * A_sq_tau * lap;

    int idx = (i * Nj + j) * Nk + k;
    u0[idx] = u0v;
    u1[idx] = u1v;

    int tid = ii + (Ni-2)*(jj + (Nj-2)*kk);

    err0[tid] = fabs(d_exact_sol(x, y, z, 0.0) - u0v);
    err1[tid] = fabs(d_exact_sol(x, y, z, tau) - u1v);
}


void launch_init(
    double*& d_u0, double*& d_u1,
    int Ni, int Nj, int Nk,
    int istart, int jstart, int kstart,
    double hx, double hy, double hz,
    double A_sq_tau, double tau,
    double& local_err0, double& local_err1
)
{   
    size_t nBytes = Ni*Nj*Nk*sizeof(double);
    if (d_u0 == nullptr) cudaMalloc(&d_u0, nBytes);
    if (d_u1 == nullptr) cudaMalloc(&d_u1, nBytes);
    dim3 block(8,8,8);
    dim3 grid( (Ni+block.x-3)/block.x,
               (Nj+block.y-3)/block.y,
               (Nk+block.z-3)/block.z );

    int nThreads = (Ni-2)*(Nj-2)*(Nk-2);

    thrust::device_vector<double> d_err0(nThreads, 0.0);
    thrust::device_vector<double> d_err1(nThreads, 0.0);

    init_kernel<<<grid, block>>>(
        d_u0, d_u1,
        Ni, Nj, Nk,
        istart, jstart, kstart,
        hx, hy, hz,
        A_sq_tau, tau,
        thrust::raw_pointer_cast(d_err0.data()),
        thrust::raw_pointer_cast(d_err1.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << "\n";
        exit(1);
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA post-synchronize error: " << cudaGetErrorString(err) << "\n";
        exit(1);
    }
    
    local_err0 = *thrust::max_element(d_err0.begin(), d_err0.end());
    local_err1 = *thrust::max_element(d_err1.begin(), d_err1.end());
}


__global__ void update_kernel(
    const double *u0,
    const double *u1,
    double *u,
    int Ni, int Nj, int Nk,
    int istart, int jstart, int kstart,
    double hx, double hy, double hz,
    double A_sq_tau,
    double tn,
    double *err
){
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;

    if (ii >= Ni - 2 || jj >= Nj - 2 || kk >= Nk - 2) return;

    int i = ii + 1;
    int j = jj + 1;
    int k = kk + 1;

    int ig = istart + ii;
    int jg = jstart + jj;
    int kg = kstart + kk;

    double x = ig * hx;
    double y = jg * hy;
    double z = kg * hz;

    int idx = (i * Nj + j) * Nk + k;

    double lap = d_laplace_7pt(u1, i, j, k, Ni, Nj, Nk, hx, hy, hz);
    double val = A_sq_tau * lap + 2.0 * u1[idx] - u0[idx];

    u[idx] = val;

    int tid = ii + (Ni-2) * (jj + (Nj-2) * kk);
    err[tid] = fabs(d_exact_sol(x, y, z, tn) - val);
    
}

void launch_update(
    const double* d_u0,
    const double* d_u1,
    double* d_u,
    int Ni, int Nj, int Nk,
    int istart, int jstart, int kstart,
    double hx, double hy, double hz,
    double A_sq_tau,
    double tn,
    double &local_err)
{
    dim3 block(8,8,8);
    dim3 grid(
        (Ni + block.x - 3) / block.x,
        (Nj + block.y - 3) / block.y,
        (Nk + block.z - 3) / block.z
    );
    int nThreads = (Ni-2)*(Nj-2)*(Nk-2);
    thrust::device_vector<double> d_err(nThreads, 0.0);

    update_kernel<<<grid, block>>>(
        d_u0, d_u1, d_u,
        Ni, Nj, Nk,
        istart, jstart, kstart,
        hx, hy, hz,
        A_sq_tau, tn,
        thrust::raw_pointer_cast(d_err.data())
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
        std::abort();
    }

    local_err = *thrust::max_element(d_err.begin(), d_err.begin() + nThreads);
}
