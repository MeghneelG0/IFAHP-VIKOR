from .utility_measures import calculate_s, calculate_r, calculate_q
def check_vikor_conditions(Q, S, R, rankings):
    """
    Checks the VIKOR conditions for acceptable advantage and stability.
    Args:
    - Q: The compromise solution (Q_i) for each alternative.
    - S: The utility measure (S_i) for each alternative.
    - R: The regret measure (R_i) for each alternative.
    - rankings: A list of tuples containing the alternative index and its Q, S, and R values, sorted by Q.
    Returns:
    - compromise_solution: The final compromise solution (list of alternatives).
    - conditions_satisfied: Boolean indicating whether both conditions are satisfied.
    """
    j = len(Q)  # Total number of alternatives
    DQ = 1 / (j - 1)  # Threshold for acceptable advantage

    # Extract the top-ranked alternative (A1) and the second-ranked alternative (A2)
    A1_index, A1_values = rankings[0]
    A2_index, A2_values = rankings[1]

    # Condition 1: Acceptable Advantage
    Q_A1 = A1_values[0]  # Q value of A1
    Q_A2 = A2_values[0]  # Q value of A2
    acceptable_advantage = (Q_A2 - Q_A1) >= DQ

    # Condition 2: Acceptable Stability
    # Check if A1 ranks first in either S or R
    S_rankings = sorted(enumerate(S), key=lambda x: x[1])  # Sort by S_i
    R_rankings = sorted(enumerate(R), key=lambda x: x[1])  # Sort by R_i
    A1_S_rank = next(i for i, (index, _) in enumerate(S_rankings) if index == A1_index)
    A1_R_rank = next(i for i, (index, _) in enumerate(R_rankings) if index == A1_index)
    acceptable_stability = (A1_S_rank == 0) or (A1_R_rank == 0)

    # Determine the compromise solution
    compromise_solution = []
    if acceptable_advantage and acceptable_stability:
        # Both conditions are satisfied: A1 is the best alternative
        compromise_solution.append(A1_index)
        conditions_satisfied = True
    else:
        # One or both conditions are violated: Extend the compromise solution
        conditions_satisfied = False
        for rank, (index, (q, _, _)) in enumerate(rankings):
            if q - Q_A1 < DQ:
                compromise_solution.append(index)

    return compromise_solution, conditions_satisfied