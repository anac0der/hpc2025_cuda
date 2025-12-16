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

void exchange_ghosts(
    double* send_x0, double* send_x1,
    double* send_y0, double* send_y1,
    double* send_z0, double* send_z1,
    double* recv_x0, double* recv_x1,
    double* recv_y0, double* recv_y1,
    double* recv_z0, double* recv_z1,
    int nx, int ny, int nz,
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
        for (int p = 0; p < count_x; ++p) {
            sendbuf[p] = send_x0[p];
        }

        MPI_Sendrecv(sendbuf.data(), count_x, MPI_DOUBLE, nbrs[0], 10,
                     recvbuf.data(), count_x, MPI_DOUBLE, nbrs[0], 11,
                     comm, &st);

        for (int p = 0; p < count_x; ++p) {
            recv_x0[p] = recvbuf[p];
        }
    }

    if (nbrs[1] != MPI_PROC_NULL) {
        for (int p = 0; p < count_x; ++p) {
            sendbuf[p] = send_x1[p];
        }

        MPI_Sendrecv(sendbuf.data(), count_x, MPI_DOUBLE, nbrs[1], 11,
                     recvbuf.data(), count_x, MPI_DOUBLE, nbrs[1], 10,
                     comm, &st);

        for (int p = 0; p < count_x; ++p) {
            recv_x1[p] = recvbuf[p];
        }
    }

    sendbuf.resize(count_y);
    recvbuf.resize(count_y);

    if (nbrs[2] != MPI_PROC_NULL) {
        for (int p = 0; p < count_y; ++p) {
            sendbuf[p] = send_y0[p];
        }

        MPI_Sendrecv(sendbuf.data(), count_y, MPI_DOUBLE, nbrs[2], 20,
                     recvbuf.data(), count_y, MPI_DOUBLE, nbrs[2], 21,
                     comm, &st);

        for (int p = 0; p < count_y; ++p) {
            recv_y0[p] = recvbuf[p];
        }
    }

    if (nbrs[3] != MPI_PROC_NULL) {
        for (int p = 0; p < count_y; ++p) {
            sendbuf[p] = send_y1[p];
        }

        MPI_Sendrecv(sendbuf.data(), count_y, MPI_DOUBLE, nbrs[3], 21,
                     recvbuf.data(), count_y, MPI_DOUBLE, nbrs[3], 20,
                     comm, &st);

        for (int p = 0; p < count_y; ++p) {
            recv_y1[p] = recvbuf[p];
        }
    }

    sendbuf.resize(count_z);
    recvbuf.resize(count_z);

    if (nbrs[4] != MPI_PROC_NULL) {
        for (int p = 0; p < count_z; ++p) {
            sendbuf[p] = send_z0[p];
        }

        MPI_Sendrecv(sendbuf.data(), count_z, MPI_DOUBLE, nbrs[4], 30,
                     recvbuf.data(), count_z, MPI_DOUBLE, nbrs[4], 31,
                     comm, &st);

        for (int p = 0; p < count_z; ++p) {
            recv_z0[p] = recvbuf[p];
        }
    }

    if (nbrs[5] != MPI_PROC_NULL) {
        for (int p = 0; p < count_z; ++p) {
            sendbuf[p] = send_z1[p];
        }

        MPI_Sendrecv(sendbuf.data(), count_z, MPI_DOUBLE, nbrs[5], 31,
                     recvbuf.data(), count_z, MPI_DOUBLE, nbrs[5], 30,
                     comm, &st);

        for (int p = 0; p < count_z; ++p) {
            recv_z1[p] = recvbuf[p];
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    double t_start_init = MPI_Wtime();
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
    std::vector<double> send_x0(j_n * k_n);  
    std::vector<double> send_x1(j_n * k_n);  
    std::vector<double> send_y0(i_n * k_n);  
    std::vector<double> send_y1(i_n * k_n);  
    std::vector<double> send_z0(i_n * j_n);  
    std::vector<double> send_z1(i_n * j_n);  
  
    std::vector<double> recv_x0(j_n * k_n);  
    std::vector<double> recv_x1(j_n * k_n);  
    std::vector<double> recv_y0(i_n * k_n);  
    std::vector<double> recv_y1(i_n * k_n);  
    std::vector<double> recv_z0(i_n * j_n);  
    std::vector<double> recv_z1(i_n * j_n);  
    
    double* d_boundary_x0;
    double* d_boundary_x1;
    double* d_boundary_y0;
    double* d_boundary_y1;
    double* d_boundary_z0;
    double* d_boundary_z1;

    int size_x = j_n * k_n;  
    int size_y = i_n * k_n;    
    int size_z = i_n * j_n;  
 
    cudaMalloc(&d_boundary_x0, size_x * sizeof(double));
    cudaMalloc(&d_boundary_x1, size_x * sizeof(double));
    cudaMalloc(&d_boundary_y0, size_y * sizeof(double));
    cudaMalloc(&d_boundary_y1, size_y * sizeof(double));
    cudaMalloc(&d_boundary_z0, size_z * sizeof(double));
    cudaMalloc(&d_boundary_z1, size_z * sizeof(double));
    double local_err0=0, local_err1=0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start_01 = MPI_Wtime();
    double* d_u0 = nullptr;
    double* d_u1 = nullptr;

    launch_init(d_u0, d_u1, 
                d_boundary_x0, d_boundary_x1, 
                d_boundary_y0, d_boundary_y1,
                d_boundary_z0, d_boundary_z1,
                i_n+2, j_n+2, k_n+2,
                istart, jstart, kstart,
                hx, hy, hz,
                A_sq_tau, tau,
                local_err0, local_err1);
    double t_end_01 = MPI_Wtime();
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
    double t_cpy_start, t_cpy_end, t_cpy_2_start, t_cpy_2_end, t_main_start, t_main_end;
    for (int n = 2; n <= K; n++)
    {
        double tn = n * tau;
   	t_cpy_start = MPI_Wtime();
         
        cudaMemcpy(send_x0.data(), d_boundary_x0, size_x*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(send_x1.data(), d_boundary_x1, size_x*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(send_y0.data(), d_boundary_y0, size_y*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(send_y1.data(), d_boundary_y1, size_y*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(send_z0.data(), d_boundary_z0, size_z*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(send_z1.data(), d_boundary_z1, size_z*sizeof(double), cudaMemcpyDeviceToHost);
        t_cpy_end = MPI_Wtime();
        
        exchange_ghosts(send_x0.data(), send_x1.data(),
                        send_y0.data(), send_y1.data(),
                        send_z0.data(), send_z1.data(),
                        recv_x0.data(), recv_x1.data(),
                        recv_y0.data(), recv_y1.data(),
                        recv_z0.data(), recv_z1.data(),
                        i_n, j_n, k_n, comm, nbrs);
        
        t_cpy_2_start = MPI_Wtime();
        cudaMemcpy(d_boundary_x0, recv_x0.data(), size_x*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_boundary_x1, recv_x1.data(), size_x*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_boundary_y0, recv_y0.data(), size_y*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_boundary_y1, recv_y1.data(), size_y*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_boundary_z0, recv_z0.data(), size_z*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_boundary_z1, recv_z1.data(), size_z*sizeof(double), cudaMemcpyHostToDevice);
        
        t_cpy_2_end = MPI_Wtime();
        double local_err = 0.0;
        t_main_start = MPI_Wtime();
        launch_update(
            d_u0,
            d_u1,
            d_u,
            d_boundary_x0, d_boundary_x1,
            d_boundary_y0, d_boundary_y1,
            d_boundary_z0, d_boundary_z1,
            i_n+2, j_n+2, k_n+2,
            istart, jstart, kstart,
            hx, hy, hz,
            A_sq_tau,
            tn,
            local_err
        );
        
        t_main_end = MPI_Wtime();
        double err;
        MPI_Reduce(&local_err, &err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) std::cout << "Step " << n << ", error = " << err << "\n";

        std::swap(d_u0, d_u1);
        std::swap(d_u1, d_u);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start_init;
    double init = t_start_01 - t_start_init;
    double first_cycle = t_end_01 - t_start_01;
    double device2host = (t_cpy_end - t_cpy_start) * (K - 2);
    double host2device = (t_cpy_2_end - t_cpy_2_start) * (K - 2);
    double exchange = (t_cpy_2_start - t_cpy_end) * (K - 2);
    double main_c = (t_main_end - t_main_start) * (K - 2);
    

    double max_elapsed, max_init, max_first_cycle;
    double total_device2host, total_host2device, total_exchange, total_main_c;
    double max_device2host, max_host2device, max_exchange, max_main_c;

// Reduce elapsed time (max across all processes)
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//
// // Reduce initialization time (max across all processes)
    MPI_Reduce(&init, &max_init, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//
// // Reduce first cycle time (max across all processes)
    MPI_Reduce(&first_cycle, &max_first_cycle, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//
// // Reduce device-to-host copy time (sum across all processes)
   MPI_Reduce(&device2host, &total_device2host, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//
// // Reduce host-to-device copy time (sum across all processes)
   MPI_Reduce(&host2device, &total_host2device, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//
// // Reduce exchange time (sum across all processes)
   MPI_Reduce(&exchange, &total_exchange, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//
// // Reduce main computation time (sum across all processes)
   MPI_Reduce(&main_c, &total_main_c, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//
// // Also get max values for the timing components
   MPI_Reduce(&device2host, &max_device2host, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&host2device, &max_host2device, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&exchange, &max_exchange, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&main_c, &max_main_c, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
           std::cout << "Elapsed time: " << max_elapsed << " seconds\n";
    std::cout << "Initialization time: " << max_init << " seconds\n";
    std::cout << "First cycle time: " << max_first_cycle << " seconds\n";
    std::cout << "Device to Host copy time: " << max_device2host << " seconds\n";
    std::cout << "Host to Device copy time: " << max_host2device << " seconds\n";
    std::cout << "Exchange time: " << max_exchange << " seconds\n";
    std::cout << "Main computation time: " << max_main_c << " seconds\n";
    }
    cudaFree(d_u0);
    cudaFree(d_u1);
    cudaFree(d_u);
    cudaFree(d_boundary_x0);
    cudaFree(d_boundary_x1);
    cudaFree(d_boundary_y0);
    cudaFree(d_boundary_y1);
    cudaFree(d_boundary_z0);
    cudaFree(d_boundary_z1);
    MPI_Finalize();
    return 0;
}
