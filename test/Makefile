# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc-6

test_sequential:
	cd ..; make lib;
	cp ../lib/*.a ../inc/knnring.h ./
	$(CC) tester.c knnring_sequential.a -o $@ -lm -lblas
	./test_sequential

