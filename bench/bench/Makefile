# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc
MPICC = mpicc

# define the C compiler flags and libs
CFLAGS = -Wall 
CLIBS = -lm -L$(NETLIB_LAPACK_ROOT)/lib64 -lcblas
INCLUDES = -I$(NETLIB_LAPACK_ROOT)/include

test_mpi:
	cd ..; make lib;
	cp ../lib/*.a ../inc/knnring.h ./
	$(MPICC) $(INCLUDES) $(CFLAGS) tester_mpi.c knnring_synchronous.a $(CLIBS) -o $@

test_mpi_async:
	cd ..; make lib;
	cp ../lib/*.a ../inc/knnring.h ./
	$(MPICC) $(INCLUDES) $(CFLAGS) tester_mpi.c knnring_asynchronous.a $(CLIBS) -o $@
clean:
	rm *.a knnring.h test_mpi test_mpi_async
