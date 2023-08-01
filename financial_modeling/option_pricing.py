# Monte Carlo simulation is a widely used technique for option pricing in finance. 
# It involves simulating the future stock price using random walks and then calculating 
# the option payoff based on the simulated stock prices.

import numpy as np
import time
import numba


def option_pricing_without_numba(S0, K, r, sigma, T, num_simulations, num_steps):
    dt = T / num_steps
    total_payoff = 0.0

    for _ in range(num_simulations):
        S = S0
        for _ in range(num_steps):
            epsilon = np.random.normal(0.0, 1.0)
            S *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * epsilon)

        total_payoff += max(S - K, 0)

    option_price = total_payoff / num_simulations
    return option_price


@numba.jit
def option_pricing_with_numba(S0, K, r, sigma, T, num_simulations, num_steps):
    dt = T / num_steps
    total_payoff = 0.0

    for _ in range(num_simulations):
        S = S0
        for _ in range(num_steps):
            epsilon = np.random.normal(0.0, 1.0)
            S *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * epsilon)

        total_payoff += max(S - K, 0)

    option_price = total_payoff / num_simulations
    return option_price


def main():
    # Option parameters
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    r = 0.05    # Risk-free interest rate
    sigma = 0.2 # Volatility (standard deviation of returns)
    T = 1.0     # Time to expiration (in years)

    # Monte Carlo simulation parameters
    num_simulations = 100000  # Number of simulations
    num_steps = 252           # Number of steps (days) for each simulation
    
    # Without Numba
    start_time = time.time()
    option_price_without_numba = option_pricing_without_numba(S0, K, r, sigma, T, num_simulations, num_steps)
    end_time = time.time()
    execution_time_without_numba = end_time - start_time

    print("Option Pricing without Numba:")
    print("Option Price:", option_price_without_numba)
    print("Execution time:", execution_time_without_numba, "seconds")

    # With Numba
    start_time = time.time()
    option_price_with_numba = option_pricing_with_numba(S0, K, r, sigma, T, num_simulations, num_steps)
    end_time = time.time()
    execution_time_with_numba = end_time - start_time

    print("Option Pricing with Numba:")
    print("Option Price:", option_price_with_numba)
    print("Execution time:", execution_time_with_numba, "seconds")

if __name__ == "__main__":
    main()
