# PDS_Exercise_2
Second exercise for Parallel and Distributed Systems. Searches kNN with MPI

# Install openBLAS for Debian/Ubuntu
sudo apt-get install libopenblas-dev

# Install OpenMPI for Debian/Ubuntu:
sudo apt-get install openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev

# Compile an MPI program(mpi-hello.c):
mpicc mpi-hello.c -o out

# Run an MPI program, where -np argument is the number of proccesses:
mpirun --oversubscribe -np 4 out
