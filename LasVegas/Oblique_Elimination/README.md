# ECDLP-LasVegas
## Oblique Elimination

### Introduction
This folder contains code for the paper titled **Oblique Elimination to solve the elliptic cirve discrete logarithm problem** from 2022.

Algorithm 2 from this paper is implemented in this code base.
The code can be compiled using the make file. 
An executable named lasVeagas is geneted after successful compilation.

### Compiling/Running Code.
	1. Compiling : make 
	2. Running : mpirun -n np* lasVegas
	   (*np - number of processors to be used.)

The following dependencies are required for the code to work.
### Dependencies:
	1. C/C++ compiler
	2. OpemMPI
	3. Number Theory Library (NTL) and GNU-GMP

The folder input contains the input files for binary as well as prime field.
Input to the program can be changes in main.cpp file.
