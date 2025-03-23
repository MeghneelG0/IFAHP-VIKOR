def check_consistency(R, R_bar, tau=0.1, verbose=False):
    """
    Computes the distance d between the original IFPR matrix R and 
    the perfectly multiplicative consistent IFPR matrix R_bar using:
    
      d(R, R_bar) = (1 / ((n-1) * (n-2))) * sum_{i=1}^{n} sum_{k=i+1}^{n} 
                     [|mu_bar(i,k) - mu(i,k)| + |nu_bar(i,k) - nu(i,k)| + |pi_bar(i,k) - pi(i,k)|],
                     
    where for each entry,
      pi(i,k) = 1 - mu(i,k) - nu(i,k) and
      pi_bar(i,k) = 1 - mu_bar(i,k) - nu_bar(i,k).
    
    Parameters:
      R      : Original IFPR matrix (n x n x 2 numpy array)
      R_bar  : Consistent IFPR matrix (n x n x 2 numpy array)
      tau    : Threshold for acceptable consistency (default: 0.1)
      verbose: Whether to print the distance (default: False)
    
    Returns:
      (bool) True if d < tau, else False.
    """
    n = R.shape[0]
    total_diff = 0.0
    
    # Loop over the upper triangular indices (i < k)
    for i in range(n):
        for k in range(i+1, n):
            mu = R[i, k, 0]
            nu = R[i, k, 1]
            pi = 1 - mu - nu
            
            mu_bar = R_bar[i, k, 0]
            nu_bar = R_bar[i, k, 1]
            pi_bar = 1 - mu_bar - nu_bar
            
            diff = abs(mu_bar - mu) + abs(nu_bar - nu) + abs(pi_bar - pi)
            total_diff += diff
    
    # Use normalization factor as in the paper: (n-1)*(n-2) or 2*(n-1)(n-2) idk man
    denominator = (n - 1) * (n - 2)
    d = total_diff / denominator
    
    if verbose:
        print("Distance d(R, R_bar) =", d)
    return d < tau