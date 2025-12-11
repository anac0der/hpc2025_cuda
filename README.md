# CUDA task at HPC course (sequential code branch)

### MPI
MPI code build:
```
make -f make_mpi
```
Run:
```
mpirun -n N ./run_mpi 128 50 0.005
```
Here `N` is the number of processes, `128` is the number of grid nodes, `50` is the number of iterations and `0.005` is the time step.
