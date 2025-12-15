#include <mpi.h>
#include <iostream>
#include <vector>
#include "utils.h"    
#include "kernel_01.cuh"
#include <cuda_runtime.h>
#include <cuda.h>  
using namespace hpc;

static void compute_blocks(int M, int P, int r, int& start, int& loc)
{
    int base = M / P;
    int rem = M % P;
    if (r < rem) {
        loc = base + 1;
        start = r * loc + 1;
    } else {
        loc = base;
        start = rem * (base + 1) + (r - rem) * base + 1;
    }
}

static void exchange_ghosts(Tensor3D& U, int nx, int ny, int nz,
                            MPI_Comm comm, const int nbrs[6])
{
    MPI_Status st;

    int count_x = ny * nz;
    int count_y = nx * nz;
    int count_z = nx * ny;

    std::vector<double> sendbuf, recvbuf;

    sendbuf.resize(count_x);
    recvbuf.resize(count_x);

    if (nbrs[0] != MPI_PROC_NULL) {
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j) {
                int p = (k-1)*ny + (j-1);
                sendbuf[p] = U(1, j, k);
            }

        MPI_Sendrecv(sendbuf.data(), count_x, MPI_DOUBLE, nbrs[0], 10,
                     recvbuf.data(), count_x, MPI_DOUBLE, nbrs[0], 11,
                     comm, &st);

        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j) {
                int p = (k-1)*ny + (j-1);
                U(0, j, k) = recvbuf[p];
            }
    }

    if (nbrs[1] != MPI_PROC_NULL) {
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j) {
                int p = (k-1)*ny + (j-1);
                sendbuf[p] = U(nx, j, k);
            }

        MPI_Sendrecv(sendbuf.data(), count_x, MPI_DOUBLE, nbrs[1], 11,
                     recvbuf.data(), count_x, MPI_DOUBLE, nbrs[1], 10,
                     comm, &st);

        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j) {
                int p = (k-1)*ny + (j-1);
                U(nx+1, j, k) = recvbuf[p];
            }
    }

    sendbuf.resize(count_y);
    recvbuf.resize(count_y);

    if (nbrs[2] != MPI_PROC_NULL) {
        for (int k = 1; k <= nz; ++k)
            for (int i = 1; i <= nx; ++i) {
                int p = (k-1)*nx + (i-1);
                sendbuf[p] = U(i, 1, k);
            }

        MPI_Sendrecv(sendbuf.data(), count_y, MPI_DOUBLE, nbrs[2], 20,
                     recvbuf.data(), count_y, MPI_DOUBLE, nbrs[2], 21,
                     comm, &st);

        for (int k = 1; k <= nz; ++k)
            for (int i = 1; i <= nx; ++i) {
                int p = (k-1)*nx + (i-1);
                U(i, 0, k) = recvbuf[p];
            }
    }

    if (nbrs[3] != MPI_PROC_NULL) {
        for (int k = 1; k <= nz; ++k)
            for (int i = 1; i <= nx; ++i) {
                int p = (k-1)*nx + (i-1);
                sendbuf[p] = U(i, ny, k);
            }

        MPI_Sendrecv(sendbuf.data(), count_y, MPI_DOUBLE, nbrs[3], 21,
                     recvbuf.data(), count_y, MPI_DOUBLE, nbrs[3], 20,
                     comm, &st);

        for (int k = 1; k <= nz; ++k)
            for (int i = 1; i <= nx; ++i) {
                int p = (k-1)*nx + (i-1);
                U(i, ny+1, k) = recvbuf[p];
            }
    }

    sendbuf.resize(count_z);
    recvbuf.resize(count_z);

    if (nbrs[4] != MPI_PROC_NULL) {
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                int p = (j-1)*nx + (i-1);
                sendbuf[p] = U(i, j, 1);
            }

        MPI_Sendrecv(sendbuf.data(), count_z, MPI_DOUBLE, nbrs[4], 30,
                     recvbuf.data(), count_z, MPI_DOUBLE, nbrs[4], 31,
                     comm, &st);

        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                int p = (j-1)*nx + (i-1);
                U(i, j, 0) = recvbuf[p];
            }
    }

    if (nbrs[5] != MPI_PROC_NULL) {
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                int p = (j-1)*nx + (i-1);
                sendbuf[p] = U(i, j, nz);
            }

        MPI_Sendrecv(sendbuf.data(), count_z, MPI_DOUBLE, nbrs[5], 31,
                     recvbuf.data(), count_z, MPI_DOUBLE, nbrs[5], 30,
                     comm, &st);

        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                int p = (j-1)*nx + (i-1);
                U(i, j, nz+1) = recvbuf[p];
            }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);

    int device_id = rank % num_devices;
    cudaSetDevice(device_id);
    int N = std::stoi(argv[1]);
    int K = std::stoi(argv[2]);

    double hx = L_x / N;
    double hy = L_y / N;
    double hz = L_z / N;

    double tau;
    if (argc < 4) {
        tau = 1.0 / std::sqrt(A_sq * (1.0/(hx*hx) + 1.0/(hy*hy) + 1.0/(hz*hz)));
        if (rank == 0) std::cout << "tau = " << tau << "\n";
    } else tau = std::stod(argv[3]);

    double A_sq_tau = A_sq * tau * tau;

    int dims[3] = {0,0,0};
    MPI_Dims_create(size, 3, dims);
    int periods[3] = {0,0,0};
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &comm);

    int coords[3];
    MPI_Cart_coords(comm, rank, 3, coords);

    int nbrs[6];
    MPI_Cart_shift(comm, 0, 1, &nbrs[0], &nbrs[1]); 
    MPI_Cart_shift(comm, 1, 1, &nbrs[2], &nbrs[3]);
    MPI_Cart_shift(comm, 2, 1, &nbrs[4], &nbrs[5]); 

    int istart, i_n;
    int jstart, j_n;
    int kstart, k_n;

    compute_blocks(N-1, dims[0], coords[0], istart, i_n);
    compute_blocks(N-1, dims[1], coords[1], jstart, j_n);
    compute_blocks(N-1, dims[2], coords[2], kstart, k_n);

    Tensor3D u0(i_n+2, j_n+2, k_n+2, hx, hy, hz);
    Tensor3D u1(i_n+2, j_n+2, k_n+2, hx, hy, hz);
    Tensor3D u(i_n+2, j_n+2, k_n+2, hx, hy, hz);

    double local_err0=0, local_err1=0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
    double* d_u0 = nullptr;
    double* d_u1 = nullptr;
    launch_init(d_u0, d_u1, i_n+2, j_n+2, k_n+2,
                istart, jstart, kstart,
                hx, hy, hz,
                A_sq_tau, tau,
                local_err0, local_err1);
    
    double err0, err1;
    MPI_Reduce(&local_err0, &err0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_err1, &err1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout<<"Step 0, error = "<<err0<<"\n";
        std::cout<<"Step 1, error = "<<err1<<"\n";
    }
    
    double *d_u = nullptr;
    size_t arr_size = (i_n+2)*(j_n+2)*(k_n+2);
    cudaMalloc(&d_u, arr_size*sizeof(double));
    for (int n = 2; n <= K; n++)
    {
        double tn = n * tau;
	cudaMemcpy(u0.data(), d_u0, (i_n+2)*(j_n+2)*(k_n+2)*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(u1.data(), d_u1, (i_n+2)*(j_n+2)*(k_n+2)*sizeof(double), cudaMemcpyDeviceToHost);
        exchange_ghosts(u1, i_n, j_n, k_n, comm, nbrs);
        exchange_ghosts(u0, i_n, j_n, k_n, comm, nbrs);
        cudaMemcpy(d_u0, u0.data(), (i_n+2)*(j_n+2)*(k_n+2)*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_u1, u1.data(), (i_n+2)*(j_n+2)*(k_n+2)*sizeof(double), cudaMemcpyHostToDevice);
        double local_err = 0.0;

        launch_update(
            d_u0,
            d_u1,
            d_u,
            i_n+2, j_n+2, k_n+2,
            istart, jstart, kstart,
            hx, hy, hz,
            A_sq_tau,
            tn,
            local_err
        );
        

        double err;
        MPI_Reduce(&local_err, &err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) std::cout << "Step " << n << ", error = " << err << "\n";

        std::swap(d_u0, d_u1);
        std::swap(d_u1, d_u);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Elapsed time: " << max_elapsed << " seconds\n";
    }
    cudaFree(d_u0);
    cudaFree(d_u1);
    cudaFree(d_u);
    MPI_Finalize();
    return 0;
}
