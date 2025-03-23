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

def compute_ifahp_vikor(verbose=False):
    """
    Performs IFAHP and VIKOR calculations
    
    Args:
        verbose: If True, prints detailed steps of the calculation
        
    Returns:
        Tuple of (rankings, Q, S, R, compromise_solution)
    """
    # Initial IFPR matrix for criteria
    R = np.array([
        [[0.5, 0.3], [0.2, 0.6], [0.1, 0.8]],
        [[0.6, 0.2], [0.5, 0.3], [0.3, 0.6]],
        [[0.8, 0.1], [0.6, 0.3], [0.5, 0.3]]
    ])
    
    # Define the decision matrix 
    try:
        # Try to load from CSV
        data = pd.read_csv("Requirements.csv")
        data = data.drop(columns=['Pre-requisite', 'Test Cases'])
        # Map Importance values ('H', 'M', 'L') to numerical values (9, 5, 3)
        importance_mapping = {'H': 9, 'M': 5, 'L': 3}
        data['Importance'] = data['Importance'].map(importance_mapping)
        
        # Extract the decision matrix
        decision_matrix = data[['Cost', 'Value', 'Importance']].values
        if verbose:
            print(f"Loaded decision matrix with {len(decision_matrix)} alternatives.")
    except Exception as e:
        if verbose:
            print(f"Could not load from CSV: {e}")
            print("Using example decision matrix instead.")
        # Example decision matrix as fallback
        decision_matrix = np.array([
            [8, 9, 9],
            [3, 2, 9],
            [7, 8, 9],
            [4, 3, 5],
            [6, 7, 9]
        ])
    
    # Define criteria types (Cost: min, Value: max, Importance: max)
    criteria_types = ['min', 'max', 'max']

    # Step 1: Construct perfect multiplicative consistent IFPR
    if verbose:
        print("\nStep 1: Computing Perfect Multiplicative Consistent IFPR...")
    R_bar = construct_perfect_multiplicative_consistent_ifpr(R)
    
    # Step 2: Check consistency
    if verbose:
        print("\nStep 2: Checking consistency...")
    consistent = check_consistency(R, R_bar, tau=0.1, verbose=verbose)
    
    # Step 3: Repair the matrix if needed
    if not consistent:
        if verbose:
            print("\nStep 3: Repairing the matrix...")
        R_repaired = repair_ifahp_algorithm_2(R_bar, sigma=0.8, tau=0.1, max_iter=100, verbose=verbose)
    else:
        R_repaired = R
        if verbose:
            print("Matrix is already consistent, no repair needed.")
    
    # Step 4: Compute weights
    if verbose:
        print("\nStep 4: Computing weights...")
    weights = compute_criterion_weights(R_repaired)
    
    # Step 5: Convert to crisp weights
    crisp_weights = convert_to_crisp_weights(weights)
    if verbose:
        print("\nFinal crisp weights:", crisp_weights)
    
    # Step 6: Determine best and worst values
    if verbose:
        print("\nStep 6: Determining best and worst values...")
    f_star, f_minus = determine_best_and_worst(decision_matrix, criteria_types)
    if verbose:
        print(f"Best values (f_star): {f_star}")
        print(f"Worst values (f_minus): {f_minus}")
    
    # Step 7: Substitute values in decision matrix
    if verbose:
        print("\nStep 7: Substituting values in decision matrix...")
    substituted_matrix = substitute_values(decision_matrix, crisp_weights, f_star, f_minus)
    
    # Step 8: Calculate utility measures
    if verbose:
        print("\nStep 8: Calculating utility measures...")
    S = calculate_s(substituted_matrix)
    R = calculate_r(substituted_matrix)
    
    # Step 9: Calculate best and worst values for S and R
    S_star = np.min(S)
    S_minus = np.max(S)
    R_star = np.min(R)
    R_minus = np.max(R)
    if verbose:
        print(f"S_star: {S_star:.4f}, S_minus: {S_minus:.4f}")
        print(f"R_star: {R_star:.4f}, R_minus: {R_minus:.4f}")
    
    # Step 10: Calculate compromise solution Q
    if verbose:
        print("\nStep 10: Calculating compromise solution Q...")
    Q = calculate_q(S, R, S_star, S_minus, R_star, R_minus)
    
    # Step 11: Rank alternatives
    if verbose:
        print("\nStep 11: Ranking alternatives...")
    rankings = rank_alternatives(Q, S, R)
    
    # Step 12: Check VIKOR conditions
    if verbose:
        print("\nStep 12: Checking VIKOR conditions...")
    compromise_solution, conditions_satisfied = check_vikor_conditions(Q, S, R, rankings)
    
    return rankings, Q, S, R, compromise_solution, conditions_satisfied, data if 'data' in locals() else None

def main():
    # Run the calculations
    rankings, Q, S, R, compromise_solution, conditions_satisfied, data = compute_ifahp_vikor(verbose=False)
    
    print("\n===== IFAHP-VIKOR Requirements Prioritization Results =====")
    
    # Show top 10 ranked alternatives
    print("\nTop 10 Ranked Requirements:")
    for rank, (index, (q, s, r)) in enumerate(rankings[:10], start=1):
        if data is not None:
            req_id = data.iloc[index]['Req ID']
            req_name = data.iloc[index]['Req Name']
            print(f"Rank {rank}: Req ID {req_id} - {req_name[:50]}")
        else:
            print(f"Rank {rank}: Alternative {index + 1}")
    
    # Show compromise solution
    print("\nCompromise Solution Status:")
    if conditions_satisfied:
        print("Both VIKOR conditions are satisfied.")
        if data is not None:
            best_index = compromise_solution[0]
            best_req_id = data.iloc[best_index]['Req ID']
            best_req_name = data.iloc[best_index]['Req Name']
            print(f"Best Requirement: ID {best_req_id} - {best_req_name}")
        else:
            print(f"Best Alternative: {compromise_solution[0] + 1}")
    else:
        print("One or both VIKOR conditions are violated.")
        print("Extended Compromise Solution:")
        if data is not None:
            for idx in compromise_solution:
                req_id = data.iloc[idx]['Req ID']
                req_name = data.iloc[idx]['Req Name']
                print(f"- Req ID {req_id} - {req_name[:50]}")
        else:
            print(f"Alternatives: {[idx + 1 for idx in compromise_solution]}")
    
    # Save results to CSV if data is available
    if data is not None:
        try:
            data['Q'] = Q
            data['S'] = S
            data['R'] = R
            data['Ranking'] = 0
            for rank, (index, _) in enumerate(rankings, start=1):
                data.at[index, 'Ranking'] = rank
                
            # Reorder columns
            desired_columns = ['Req ID', 'Req Name', 'Description', 'Cost', 'Value', 'Importance', 'Q', 'S', 'R', 'Ranking']
            data = data[desired_columns]
            data = data.sort_values(by='Ranking', ascending=True)
            
            # Save to CSV
            output_file = "Ranked_Requirements.csv"
            data.to_csv(output_file, index=False)
            print(f"\nComplete ranked requirements saved to '{output_file}'")
        except Exception as e:
            print(f"Could not save results: {e}")

if __name__ == "__main__":
    main()