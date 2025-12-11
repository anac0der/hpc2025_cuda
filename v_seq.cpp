#include <iostream>
#include "utils.h"
#include <chrono>

using namespace hpc;

int main(int argc, char** argv) {
    auto t0 = std::chrono::high_resolution_clock::now();;
    int N = std::stoi(argv[1]); // grid resolution
    int K = std::stoi(argv[2]); // number of time steps
    double h_x = L_x / N;
    double h_y = L_y / N;
    double h_z = L_z / N;
    double tau; //temporal step
    if (argc < 4){
        tau = 1.0 / std::sqrt(A_sq * (1.0/(h_x * h_x) + 1.0/(h_y * h_y) + 1.0/(h_z * h_z))); // automatic choice of time step value
        std::cout << "tau value will be chosen automatically by CFL condition" << "\n";
    }
    else{
        tau = std::stof(argv[3]);
    }
    std::cout << "tau: "<< tau << "\n";
    Tensor3D u_prev_prev = Tensor3D(N + 1, h_x, h_y, h_z);
    Tensor3D u_prev = Tensor3D(N + 1, h_x, h_y, h_z);
    double A_sq_tau = A_sq * std::pow(tau, 2);

    // u^0, u^1
    double max_error_0 = -1;
    double max_error_1 = -1;

    for (int i = 1; i < N; i++){
        for (int j = 1; j < N; j++){
            for (int k = 1; k < N; k++){
                double x = i * h_x;
                double y = j * h_y;
                double z = k * h_z;
                u_prev_prev(i, j, k) = initial_cond(x, y, z);
                u_prev(i, j, k) = u_prev_prev(i, j, k) + (A_sq_tau / 2) * laplace_7point_callable(initial_cond, x, y, z, h_x, h_y, h_z);
                double diff_0 = exact_sol(x, y, z, 0) - u_prev_prev(i, j, k);
                max_error_0 = std::fmax(std::abs(diff_0), max_error_0);
                double diff_1 = exact_sol(x, y, z, tau) - u_prev(i, j, k);
                max_error_1 = std::fmax(std::abs(diff_1), max_error_1);
            }
        }
    }
    
    std::cout << "Step 0, error: " << max_error_0 << "\n";
    std::cout << "Step 1, error: " << max_error_1 << "\n";

    Tensor3D u = Tensor3D(N + 1, h_x, h_y, h_z);
    for (int n = 2; n <= K; n++){
        double max_error = -1;
        double t_n = n * tau;
        for (int i = 1; i < N; i++){
            for (int j = 1; j < N; j++){
                for (int k = 1; k < N; k++){
                    u(i, j, k) = A_sq_tau * laplace_7point(u_prev, i, j, k) + 2 * u_prev(i, j, k) - u_prev_prev(i, j, k);
                    double x = i * h_x;
                    double y = j * h_y;
                    double z = k * h_z;
                    double diff = exact_sol(x, y, z, t_n) - u(i, j, k);
                    max_error = std::fmax(std::abs(diff), max_error);
                }
            }
        }
        std::cout << "Step " << n << ", error: " << max_error << "\n";
        std::swap(u_prev_prev, u_prev);
        std::swap(u, u_prev);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    std::cout << "Execution time: " << duration.count() / 1000000.0f << " seconds " << "\n";
    return 0;
}
