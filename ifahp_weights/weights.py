import numpy as np
def compute_criterion_weights(R):
    """
    Computes the intuitionistic fuzzy weights ω_i = (μ_i, ν_i) for each row i
    using Formula (26).
    """
    n = R.shape[0]
    weights = np.zeros((n, 2), dtype=float)
    
    # Calculate the denominators first (they are the same for all weights)
    sum_all_mu = 0.0
    sum_all_one_minus_nu = 0.0
    
    for i in range(n):
        for k in range(n):
            sum_all_mu += R[i, k, 0]  # Sum of all μ(i,k)
            sum_all_one_minus_nu += (1.0 - R[i, k, 1])  # Sum of all (1-ν(i,k))
    
    # Now calculate weights for each criterion
    for i in range(n):
        sum_mu_i = sum(R[i, k, 0] for k in range(n))  # Sum of μ(i,k) for row i
        sum_one_minus_nu_i = sum(1.0 - R[i, k, 1] for k in range(n))  # Sum of (1-ν(i,k)) for row i
        
        # Calculate weights according to formula 26
        if sum_all_one_minus_nu <= 1e-15 or sum_all_mu <= 1e-15:
            w_mu = 0.5
            w_nu = 0.5
        else:
            w_mu = sum_mu_i / sum_all_one_minus_nu
            w_nu = 1.0 - (sum_one_minus_nu_i / sum_all_mu)
        
        weights[i, 0] = w_mu
        weights[i, 1] = w_nu
    
    return weights

def convert_to_crisp_weights(weights):
    """
    Converts intuitionistic fuzzy weights to crisp weights.
    
    Args:
    - weights: np.ndarray of shape (n, 2) where weights[i, 0] is μ_i and weights[i, 1] is ν_i
    
    Returns:
    - crisp_weights: Normalized crisp weights
    """
    n = weights.shape[0]
    crisp_values = np.zeros(n)
    
    for i in range(n):
        mu = weights[i, 0]
        nu = weights[i, 1]
        
        # Score function: S(ω_i) = μ_i - ν_i
        score = mu - nu
        
        # Accuracy function: H(ω_i) = μ_i + ν_i
        accuracy = mu + nu
        
        # Combined score (this is one approach)
        crisp_values[i] = 0.5 * (score + 1) * (1 + accuracy - score)
    
    # Normalize to ensure sum is 1
    sum_crisp = np.sum(crisp_values)
    if sum_crisp > 0:
        crisp_weights = crisp_values / sum_crisp
    else:
        # Equal weights if all values are 0
        crisp_weights = np.ones(n) / n
        
    return crisp_weights