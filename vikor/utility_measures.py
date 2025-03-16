import numpy as np

def calculate_s(substituted_matrix):
    """
    Calculates the utility measure (S_i) for each alternative.
    Args:
    - substituted_matrix: The substituted decision matrix.
    Returns:
    - S: The utility measure for each alternative.
    """
    S = np.sum(substituted_matrix, axis=1)
    return S

def calculate_r(substituted_matrix):
    """
    Calculates the regret measure (R_i) for each alternative.
    Args:
    - substituted_matrix: The substituted decision matrix.
    Returns:
    - R: The regret measure for each alternative.
    """
    R = np.max(substituted_matrix, axis=1)
    return R

def calculate_q(S, R, S_star, S_minus, R_star, R_minus, v=0.5):
    """
    Calculates the compromise solution (Q_i) for each alternative.
    Args:
    - S: The utility measure for each alternative.
    - R: The regret measure for each alternative.
    - S_star, S_minus, R_star, R_minus: Best and worst values for S_i and R_i.
    - v: Strategy weight (default is 0.5).
    Returns:
    - Q: The compromise solution for each alternative.
    """
    m = len(S)
    Q = np.zeros(m)

    for i in range(m):
        s_term = (S[i] - S_star) / (S_minus - S_star + 1e-10)
        r_term = (R[i] - R_star) / (R_minus - R_star + 1e-10)
        Q[i] = v * s_term + (1 - v) * r_term

    return Q


