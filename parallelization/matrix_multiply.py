# Matrix multiplication is a computationally intensive task that can benefit from parallelization. 
# We'll use Numba's numba.prange function to parallelize the nested loops for matrix multiplication, 
# taking advantage of multiple CPU cores.

import numpy as np
import time
import numba


def matrix_multiply_without_numba(A, B):
    m, n, p = A.shape[0], A.shape[1], B.shape[1]
    result = np.zeros((m, p), dtype=np.float64)

    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i, j] += A[i, k] * B[k, j]

    return result


@numba.njit(parallel=True)
def matrix_multiply_with_numba(A, B):
    m, n, p = A.shape[0], A.shape[1], B.shape[1]
    result = np.zeros((m, p), dtype=np.float64)

    for i in numba.prange(m):
        for j in range(p):
            for k in range(n):
                result[i, j] += A[i, k] * B[k, j]

    return result

def main():
    # Generate large random matrices
    size = 200
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    # Without Numba
    start_time = time.time()
    result_without_numba = matrix_multiply_without_numba(A, B)
    end_time = time.time()
    execution_time_without_numba = end_time - start_time

    print("Matrix Multiplication without Numba:")
    print("Execution time:", execution_time_without_numba, "seconds")

    # With Numba Parallelization
    start_time = time.time()
    result_with_numba = matrix_multiply_with_numba(A, B)
    end_time = time.time()
    execution_time_with_numba = end_time - start_time

    print("Matrix Multiplication with Numba Parallelization:")
    print("Execution time:", execution_time_with_numba, "seconds")

if __name__ == "__main__":
    main()
