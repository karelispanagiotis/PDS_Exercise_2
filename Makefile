#define the shell to bash
SHELL := /bin/bash

#define the C compiler to use
CC = gcc-7

#define the MPI compiler to use
MPIC = mpicc

#define compile-time flags
CFLAGS = -Wall -O3 -lblas

#define directories containing header files
INCLUDES = -I ./inc

lib: knnring_sequential.o knnring_synchronous.o
	ar rcs lib/knnring_sequential.a lib/knnring_sequential.o
	ar rcs lib/knnring_synchronous.a lib/knnring_synchronous.o lib/knnring_sequential.o
	rm ./lib/*.o

knnring_sequential.o: src/knnring_sequential.c
	$(CC) $(CFLAGS) $(INCLUDES) -c src/knnring_sequential.c -o lib/$@

knnring_synchronous.o: src/knnring_synchronous.c knnring_sequential.o
	$(MPIC) $(CFlAGS) $(INCLUDES) -c src/knnring_synchronous.c -o lib/$@
