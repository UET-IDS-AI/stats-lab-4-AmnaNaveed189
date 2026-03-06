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
    """

    # Analytical values using CDF F(x)=1-e^{-x}

    analytic_gt5 = math.exp(-5)

    analytic_lt5 = 1 - math.exp(-5)

    analytic_interval = (1 - math.exp(-7)) - (1 - math.exp(-3))

    # Monte Carlo simulation
    samples = np.random.exponential(scale=1, size=100000)

    simulated_gt5 = np.mean(samples > 5)

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5


# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():
    """
    Candidate PDF

        f(x) = 2x e^{-x^2} for x >= 0
    """

    # define function
    def f(x):
        return 2 * x * np.exp(-x**2)

    # check integral from 0 to infinity
    integral_value, _ = quad(lambda x: 2*x*np.exp(-x**2), 0, np.inf)

    # non negativity check (sample points)
    x_vals = np.linspace(0, 3, 100)
    non_negative = np.all(f(x_vals) >= 0)

    # valid pdf condition
    is_valid_pdf = non_negative and abs(integral_value - 1) < 1e-3

    # plot
    plt.plot(x_vals, f(x_vals))
    plt.title("PDF: f(x) = 2x e^{-x^2}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    return integral_value, is_valid_pdf


# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():
    """
    X ~ Exp(1)
    """

    # Analytical probabilities
    analytic_gt5 = math.exp(-5)

    analytic_interval = math.exp(-1) - math.exp(-3)

    # Simulation
    samples = np.random.exponential(scale=1, size=100000)

    simulated_gt5 = np.mean(samples > 5)

    simulated_interval = np.mean((samples > 1) & (samples < 3))

    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
    """
    X ~ N(10,2^2)
    """

    # Analytical probabilities using scipy
    analytic_le12 = norm.cdf(12, loc=10, scale=2)

    analytic_interval = norm.cdf(12, loc=10, scale=2) - norm.cdf(8, loc=10, scale=2)

    # Simulation
    samples = np.random.normal(loc=10, scale=2, size=100000)

    simulated_le12 = np.mean(samples <= 12)

    simulated_interval = np.mean((samples > 8) & (samples < 12))

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval

