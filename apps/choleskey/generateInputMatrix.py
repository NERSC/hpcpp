#!/usr/bin/env python
# coding: utf-8

# This is an Jupyter Script for generating the Input Positive Definition(PD) Matrix for Cholesky decomposition.
# The generated .txt file stores the entire PD matrix with the matrix elems row by row. 
# This input.txt file can be used directly In benchmark part in each cpp file.


import numpy as np

#enter matrix size
matrixSize = 4

#1. Generates a random matrix A of size matrixSize x matrixSize.
A = np.random.rand(matrixSize, matrixSize)
print(A)

#2. Computes the product of matrix A and its transpose, i.e., A * A^T, storing the result in matrix B.
B = np.dot(A, A.transpose())
print("B")
print(B)

#3. Adds the transpose of matrix B to itself, generating a symmetric matrix C.
C = B + B.T  # Ensure it's symmetric

print("C")
print(C)

try:
    D = np.linalg.cholesky(C)
    print("Matrix C is positive definite.")
    
    # Write the Cholesky decomposition result to a text file
    with open(f"cholesky_result_{matrixSize}.txt", "w") as file:
        for row in C:
            for value in row:
                file.write(f"{value}\n")
except np.linalg.LinAlgError:
    print("Matrix C is not positive definite.")

