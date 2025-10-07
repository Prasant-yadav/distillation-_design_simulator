import numpy as np
from scipy.optimize import fsolve


def fenske_equation(xD, xB, alpha):
    """Calculates minimum number of stages using Fenske equation."""
    if xD <= 0 or xB <= 0 or xD >= 1 or xB >= 1:
        return 0
    N_min = np.log((xD / (1 - xD)) * ((1 - xB) / xB)) / np.log(alpha)
    return N_min


def underwood_equation(alpha, xF, q, xD):
    """Calculates R_min using the Gilliland correlation approximation."""
    try:
        # Calculate equilibrium vapor composition at feed composition
        y_eq_xF = alpha * xF / (1 + (alpha - 1) * xF)

        # Use the formula: R_min = (xD - y_eq(xF)) / (y_eq(xF) - xF)
        if (y_eq_xF - xF) == 0:
            return 1.0  # Avoid division by zero, return a fallback

        R_min = (xD - y_eq_xF) / (y_eq_xF - xF)

        return max(0.1, R_min)
    except:
        # Fallback for any other errors
        return 1.0
