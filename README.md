# CUDA task at HPC course (sequential code branch)

Overall report -- in report_cuda.pdf
### MPI
MPI code build:
```
make -f make_mpi
```
Run:
```
mpirun -n N ./run_mpi 256 50 0.002
```
Here `N` is the number of processes, `128` is the number of grid nodes, `50` is the number of iterations and `0.002` is the time step.


### MPI+CUDA
MPI+CUDA code build (on Polus):
```
make -f make_mpi_cuda ARCH=sm_60 HOST_COMP=mpicxx
```
Run:
```
mpirun -n N ./run_mpi_cuda 256 50 0.002
```
Here `N` is the number of processes, `128` is the number of grid nodes, `50` is the number of iterations and `0.002` is the time step.

