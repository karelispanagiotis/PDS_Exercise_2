# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc-7
MPICC = mpicc
MPIRUN = mpirun -np 2

# define the C compiler flags
CFLAGS = -lblas -lm -Wall

test_sequential:
	cd ..; make lib;
	cp ../lib/*.a ../inc/knnring.h ./
	$(CC) $(CFLAGS) tester.c knnring_sequential.a -o $@ 
	./test_sequential
	rm *.a test_sequential knnring.h

test_mpi:
	cd ..; make lib;
	cp ../lib/*.a ../inc/knnring.h ./
	$(MPICC) $(CFLAGS) tester_mpi.c knnring_synchronous.a -o $@
	$(MPIRUN) $@
	rm *.a test_mpi knnring.h

