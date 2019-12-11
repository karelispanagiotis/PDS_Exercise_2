# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc-7
#CC = gcc-6
MPICC = mpicc
MPIRUN = mpirun --oversubscribe -np 4

# define the C compiler flags
CFLAGS = -Wall 
CLIBS = -lm -lopenblas

test_sequential:
	cd ..; make lib;
	cp ../lib/*.a ../inc/knnring.h ./
	$(CC) $(CFLAGS) tester.c knnring_sequential.a $(CLIBS) -o $@ 
	./test_sequential
	rm *.a test_sequential knnring.h

test_mpi:
	cd ..; make lib;
	cp ../lib/*.a ../inc/knnring.h ./
	$(MPICC) $(CFLAGS) tester_mpi.c knnring_synchronous.a $(CLIBS) -o $@
	$(MPIRUN) $@
	rm *.a $@ knnring.h

test_mpi_async:
	cd ..; make lib;
	cp ../lib/*.a ../inc/knnring.h ./
	$(MPICC) $(CFLAGS) tester_mpi.c knnring_asynchronous.a $(CLIBS) -o $@
	$(MPIRUN) $@
	rm *.a $@ knnring.h
