import numpy as np
import pandas as pd
from ifahp_weights.pmc_matrix import construct_perfect_multiplicative_consistent_ifpr
from ifahp_weights.repair_matrix import repair_ifahp_algorithm_2    
from ifahp_weights.consistency_check import check_consistency 
from ifahp_weights.weights import compute_criterion_weights, convert_to_crisp_weights
from vikor.decision_matrix import determine_best_and_worst
from vikor.decision_matrix import substitute_values
from vikor.utility_measures import calculate_s, calculate_r, calculate_q
from vikor.ranking import rank_alternatives
from vikor.vikor_conditions import check_vikor_conditions
def main():
    # Edit this IFPR matrix
    R = np.array([
        [[0.5, 0.3], [0.2, 0.6], [0.1, 0.8]],
        [[0.6, 0.2], [0.5, 0.3], [0.3, 0.6]],
        [[0.8, 0.1], [0.6, 0.3], [0.5, 0.3]]
    ])
    # Define the decision matrix 
    # Edit this decision matrix 
    decision_matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    # Define criteria types
    # Edit this criteria types
    criteria_types = ['min', 'max', 'max']  # Cost: min, Value: max, Importance: max    

    R_bar = construct_perfect_multiplicative_consistent_ifpr(R)

    consistent = check_consistency(R, R_bar, tau=0.1)
    if not consistent:
        R_repaired = repair_ifahp_algorithm_2(R_bar, sigma=0.8, tau=0.1, max_iter=100)

    weights = compute_criterion_weights(R_repaired)

    crisp_weights = convert_to_crisp_weights(weights)
    print("Final crisp weights:", crisp_weights)

    f_star, f_minus = determine_best_and_worst(decision_matrix, criteria_types)

    substituted_matrix = substitute_values(decision_matrix, crisp_weights, f_star, f_minus)
    S = calculate_s(substituted_matrix)
    R = calculate_r(substituted_matrix)
    
    S_star = np.min(S)
    S_minus = np.max(S)
    R_star = np.min(R)
    R_minus = np.max(R)
    Q = calculate_q(S, R, S_star, S_minus, R_star, R_minus)

    rankings = rank_alternatives(Q, S, R)
    print("\nRanked Alternatives:")
    for rank, (index, (q, s, r)) in enumerate(rankings, start=1):
        print(f"Rank {rank}: Alternative {index + 1}, Q = {q:.4f}, S = {s:.4f}, R = {r:.4f}")
    
    compromise_solution, conditions_satisfied = check_vikor_conditions(Q, S, R, rankings)
    if conditions_satisfied:
        print("Both conditions are satisfied.")
        print(f"Best Alternative: {compromise_solution[0] + 1}")
    else:
        print("One or both conditions are violated.")
        print("Compromise Solution (Extended):", [idx + 1 for idx in compromise_solution])




if __name__ == "__main__":
    main()