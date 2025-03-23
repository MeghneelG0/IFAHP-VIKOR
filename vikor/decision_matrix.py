import numpy as np
import pandas as pd
from ifahp_weights.weights import convert_to_crisp_weights

# Assuming you need to convert some fuzzy weights to crisp weights
# fuzzy_weights = [...]  # Define your fuzzy weights here
# crisp_weights = convert_to_crisp_weights(fuzzy_weights)


def determine_best_and_worst(decision_matrix, criteria_types):
    """
    Determines the best and worst values for each criterion.
    Args:
    - decision_matrix: A numpy array of shape (m, n), where m is the number of alternatives and n is the number of criteria.
    - criteria_types: A list indicating the type of each criterion ('max' for beneficial, 'min' for non-beneficial).
    Returns:
    - f_star: The best values for each criterion.
    - f_minus: The worst values for each criterion.
    """
    f_star = []
    f_minus = []

    for j, criterion_type in enumerate(criteria_types):
        if criterion_type == 'max':  # Beneficial criterion
            f_star.append(np.max(decision_matrix[:, j]))
            f_minus.append(np.min(decision_matrix[:, j]))
        elif criterion_type == 'min':  # Non-beneficial criterion
            f_star.append(np.min(decision_matrix[:, j]))
            f_minus.append(np.max(decision_matrix[:, j]))

    return np.array(f_star), np.array(f_minus)


def substitute_values(decision_matrix, weights, f_star, f_minus):
    """
    Substitutes each value in the decision matrix with the given formula.
    Args:
    - decision_matrix: The decision matrix.
    - weights: The weights for each criterion.
    - f_star: The best values for each criterion.
    - f_minus: The worst values for each criterion.
    Returns:
    - substituted_matrix: The substituted decision matrix.
    """
    m, n = decision_matrix.shape
    substituted_matrix = np.zeros_like(decision_matrix, dtype=float)

    for i in range(m):
        for j in range(n):
            numerator = f_star[j] - decision_matrix[i, j]
            denominator = f_star[j] - f_minus[j] + 1e-10
            substituted_matrix[i, j] = crisp_weights[j] * (numerator / denominator)

    return substituted_matrix
