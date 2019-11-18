# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc-7

# define the C compiler flags
CFLAGS = -lblas -lm -Wall

test_sequential:
	cd ..; make lib;
	cp ../lib/*.a ../inc/knnring.h ./
	$(CC) $(CFLAGS) tester.c knnring_sequential.a -o $@ 
	./test_sequential
	rm *.a test_sequential

