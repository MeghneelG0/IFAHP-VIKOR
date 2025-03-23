import numpy as np
from .consistency_check import check_consistency
from .pmc_matrix import construct_perfect_multiplicative_consistent_ifpr

def repair_ifahp_algorithm_2(R, sigma=0.5, tau=0.1, max_iter=100, verbose=False):
    """
    Repairs the IFPR R using the fused intuitionistic preference algorithm (Algorithm 2)
    until the consistency measure d(R, R_bar) < tau.
    
    The update rules for off-diagonal entries are:
    
      μ^(p+1)(i,k) = [ (μ^(p)(i,k))^(1-σ) * (μ̄(i,k))^σ ] /
                      { (μ^(p)(i,k))^(1-σ) * (μ̄(i,k))^σ + (1-μ^(p)(i,k))^(1-σ) * (1-μ̄(i,k))^σ }
    
      ν^(p+1)(i,k) = [ (ν^(p)(i,k))^(1-σ) * (ν̄(i,k))^σ ] /
                      { (ν^(p)(i,k))^(1-σ) * (ν̄(i,k))^σ + (1-ν^(p)(i,k))^(1-σ) * (1-ν̄(i,k))^σ }
    
    Diagonal entries are fixed to (0.5, 0.5) and the lower-triangular part is
    updated via reciprocity.
    """
    # First, compute the perfect IFPR matrix.
    R_bar = construct_perfect_multiplicative_consistent_ifpr(R)
    if verbose:
        print("Initial consistency check:")
    if check_consistency(R, R_bar, tau, verbose):
        if verbose:
            print("R is already consistent with R_bar; no repair needed.")
        return R

    # Start the iterative repair process.
    R_current = R.copy()
    n = R_current.shape[0]
    
    for p in range(max_iter):
        R_next = np.copy(R_current)
        
        # Update all off-diagonal entries using the ratio-based formulas.
        for i in range(n):
            for k in range(n):
                if i != k:
                    mu_p = R_current[i, k, 0]
                    nu_p = R_current[i, k, 1]
                    mu_bar = R_bar[i, k, 0]
                    nu_bar = R_bar[i, k, 1]
                    
                    # Membership update:
                    num_mu = (mu_p ** (1 - sigma)) * (mu_bar ** sigma)
                    den_mu = num_mu + ((1 - mu_p) ** (1 - sigma)) * ((1 - mu_bar) ** sigma)
                    mu_next = num_mu / den_mu if den_mu != 0 else 0.5
                    
                    # Non-membership update:
                    num_nu = (nu_p ** (1 - sigma)) * (nu_bar ** sigma)
                    den_nu = num_nu + ((1 - nu_p) ** (1 - sigma)) * ((1 - nu_bar) ** sigma)
                    nu_next = num_nu / den_nu if den_nu != 0 else 0.5
                    
                    R_next[i, k, 0] = mu_next
                    R_next[i, k, 1] = nu_next
        
        # Enforce the diagonal to remain (0.5, 0.5)
        for i in range(n):
            R_next[i, i] = [0.5, 0.5]
        
        # Enforce reciprocity for the lower-triangular part:
        for i in range(n):
            for k in range(i+1, n):
                mu_val, nu_val = R_next[i, k]
                R_next[k, i] = [nu_val, mu_val]
        
        if verbose:
            print(f"Iteration {p+1} consistency check:")
        if check_consistency(R_next, R_bar, tau, verbose):
            if verbose:
                print(f"Repair successful after {p+1} iterations.")
            return R_next
        
        R_current = R_next
    
    if verbose:
        print("Max iterations reached; returning final repaired matrix.")
    return R_current