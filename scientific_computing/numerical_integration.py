import time
import numba

def f(x):
    # The function to be integrated
    return x**2

def numerical_integration_without_numba(f, a, b, n):
    h = (b - a) / n
    integral = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        x = a + i * h
        integral += f(x)
    integral *= h
    return integral


@numba.jit
def g(x):
    # The function to be integrated
    return x**2

@numba.jit
def numerical_integration_with_numba(f, a, b, n):
    h = (b - a) / n
    integral = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        x = a + i * h
        integral += f(x)
    integral *= h
    return integral

def main():
    a = 0.0  # Lower limit of integration
    b = 1.0  # Upper limit of integration
    n = 10000000  # Number of trapezoids
    
    # Without Numba
    start_time = time.time()
    result_without_numba = numerical_integration_without_numba(f, a, b, n)
    end_time = time.time()
    execution_time_without_numba = end_time - start_time

    print("Numerical Integration without Numba:")
    print("Result:", result_without_numba)
    print("Execution time:", execution_time_without_numba, "seconds")

    # With Numba
    start_time = time.time()
    result_with_numba = numerical_integration_with_numba(g, a, b, n)
    end_time = time.time()
    execution_time_with_numba = end_time - start_time

    print("Numerical Integration with Numba:")
    print("Result:", result_with_numba)
    print("Execution time:", execution_time_with_numba, "seconds")

if __name__ == "__main__":
    main()
