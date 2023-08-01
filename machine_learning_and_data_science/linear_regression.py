import numpy as np
import time
import numba


def linear_regression_without_numba(X, y):
    n = len(X)
    X_mean = np.mean(X)
    y_mean = np.mean(y)

    numerator = 0.0
    denominator = 0.0

    for i in range(n):
        numerator += (X[i] - X_mean) * (y[i] - y_mean)
        denominator += (X[i] - X_mean) ** 2

    slope = numerator / denominator
    intercept = y_mean - slope * X_mean
    return slope, intercept


@numba.jit
def linear_regression_with_numba(X, y):
    n = len(X)
    X_mean = np.mean(X)
    y_mean = np.mean(y)

    numerator = 0.0
    denominator = 0.0

    for i in range(n):
        numerator += (X[i] - X_mean) * (y[i] - y_mean)
        denominator += (X[i] - X_mean) ** 2

    slope = numerator / denominator
    intercept = y_mean - slope * X_mean
    return slope, intercept

def main():
    # Generate a large dataset
    np.random.seed(0)
    X = np.random.rand(10000000)  # Predictor variable
    y = 2 * X + 3 + np.random.randn(10000000)  # Target variable (with some noise)

    # Without Numba
    start_time = time.time()
    slope, intercept = linear_regression_without_numba(X, y)
    end_time = time.time()
    execution_time_without_numba = end_time - start_time

    print("Linear Regression without Numba:")
    print("Slope:", slope)
    print("Intercept:", intercept)
    print("Execution time:", execution_time_without_numba, "seconds")

    # With Numba
    start_time = time.time()
    slope, intercept = linear_regression_with_numba(X, y)
    end_time = time.time()
    execution_time_with_numba = end_time - start_time

    print("Linear Regression with Numba:")
    print("Slope:", slope)
    print("Intercept:", intercept)
    print("Execution time:", execution_time_with_numba, "seconds")

if __name__ == "__main__":
    main()
