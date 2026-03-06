"""
AI Stats Lab
Random Variables and Distributions
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():
    """
    STEP 1
    Compute analytically

        P(X > 5)
        P(X < 5)
        P(3 < X < 7)

    STEP 2
    Simulate 100000 samples from Exp(1)

    STEP 3
    Estimate P(X > 5) using simulation

    RETURN

        analytic_gt5
        analytic_lt5
        analytic_interval
        simulated_gt5
    """

    lambda_ = 1
    x = 5

    analytic_gt5 = math.exp(-lambda_ * x)         # P(X > 5)
    analytic_lt5 = 1 - analytic_gt5               # P(X < 5)
    analytic_interval = math.exp(-lambda_ * 3) - math.exp(-lambda_ * 7)  # P(3 < X < 7)

    # Simulation
    samples = np.random.exponential(scale=1/lambda_, size=100000)
    simulated_gt5 = np.mean(samples > 5)         # Proportion > 5

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5

cdf_probabilities()


# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():
    """
    Candidate PDF

        f(x) = 2x e^{-x^2} for x >= 0

    STEP 1
    Verify non-negativity

    STEP 2
    Compute

        integral_0^∞ f(x) dx

    STEP 3
    Determine if valid PDF

    STEP 4
    Plot f(x) on [0,3]

    RETURN

        integral_value
        is_valid_pdf
    """

    f = lambda x: 2*x*np.exp(-x**2)
    
    # STEP 1: Non-negativity check
    is_non_negative = np.all(f(np.linspace(0, 3, 1000)) >= 0)
    
    # STEP 2: Integral from 0 to infinity
    integral_value, _ = quad(f, 0, np.inf)
    
    # STEP 3: Valid PDF?
    is_valid_pdf = is_non_negative and np.isclose(integral_value, 1)
    
    # STEP 4: Plot
    x_vals = np.linspace(0, 3, 1000)
    y_vals = f(x_vals)
    plt.plot(x_vals, y_vals)
    plt.title("PDF f(x) = 2x e^(-x^2)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
    
    return integral_value, is_valid_pdf

pdf_validation_plot()

# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():
    """
    X ~ Exp(1)

    STEP 1
    Compute analytically

        P(X > 5)
        P(1 < X < 3)

    STEP 2
    Simulate 100000 samples

    STEP 3
    Estimate probabilities using simulation

    RETURN

        analytic_gt5
        analytic_interval
        simulated_gt5
        simulated_interval
    """

    lambda_ = 1
    
    # Analytical
    analytic_gt5 = math.exp(-lambda_ * 5)
    analytic_interval = math.exp(-lambda_ * 1) - math.exp(-lambda_ * 3)  # P(1 < X < 3)
    
    # Simulation
    samples = np.random.exponential(scale=1/lambda_, size=100000)
    simulated_gt5 = np.mean(samples > 5)
    simulated_interval = np.mean((samples > 1) & (samples < 3))
    
    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval

exponential_probabilities()


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
    """
    X ~ N(10,2^2)

    STEP 1
    Standardize variable

        Z = (X - 10)/2

    STEP 2
    Compute analytically

        P(X ≤ 12)
        P(8 < X < 12)

    STEP 3
    Simulate 100000 samples

    STEP 4
    Estimate probabilities

    RETURN

        analytic_le12
        analytic_interval
        simulated_le12
        simulated_interval
    """

    mu = 10
    sigma = 2
    
    # Analytical
    z_12 = (12 - mu) / sigma
    z_8 = (8 - mu) / sigma
    
    analytic_le12 = norm.cdf(z_12)            # P(X ≤ 12)
    analytic_interval = norm.cdf(z_12) - norm.cdf(z_8)  # P(8 < X < 12)
    
    # Simulation
    samples = np.random.normal(loc=mu, scale=sigma, size=100000)
    simulated_le12 = np.mean(samples <= 12)
    simulated_interval = np.mean((samples > 8) & (samples < 12))
    
    return analytic_le12, analytic_interval, simulated_le12, simulated_interval

gaussian_probabilities()
