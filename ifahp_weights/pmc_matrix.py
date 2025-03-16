import numpy as np
def construct_perfect_multiplicative_consistent_ifpr(R):
    """
    Constructs the Perfect Multiplicative Consistent IFPR (R_bar) from the original IFPR R.
    
    For i < k, if we define the chain over t = i+1, ..., k-1, then:
    
       μ̃(i,k) = { [∏ₜ ( μ(i,t) * μ(t,k) )]^(exponent) } /
                  { [∏ₜ ( μ(i,t) * μ(t,k) )]^(exponent) + [∏ₜ ( (1-μ(i,t)) * (1-μ(t,k)) )]^(exponent) }
    
       ν̃(i,k) is defined similarly using ν.
    
    The exponent is chosen as follows:
      - If chain length = (k - i - 1) equals 1, exponent = 1.
      - If chain length > 1, exponent = 0.5 (i.e. take square root).
    
    For adjacent indices (k = i+1), the original values are kept.
    The lower-triangular part is filled by reciprocity:
      R_bar[k,i] = (ν̃(i,k), μ̃(i,k))
    """
    n = R.shape[0]
    R_bar = np.zeros((n, n, 2), dtype=float)

    # 1. Set diagonal entries to (0.5, 0.5)
    for i in range(n):
        R_bar[i, i] = [0.5, 0.5]

    # 2. Compute upper-triangular entries
    for i in range(n):
        for k in range(i+1, n):
            if k == i + 1:
                # For adjacent indices, simply use the original value.
                R_bar[i, k] = R[i, k]
            else:
                exponent = 1.0 / (k - i - 1)
                # Compute chain-products for membership
                prod_mu = 1.0
                prod_1mu = 1.0
                for t in range(i+1, k):
                    prod_mu   *= (R[i, t, 0] * R[t, k, 0])
                    prod_1mu  *= ((1 - R[i, t, 0]) * (1 - R[t, k, 0]))
                left_mu  = prod_mu ** exponent
                right_mu = prod_1mu ** exponent
                denom_mu = left_mu + right_mu
                new_mu = left_mu / denom_mu if denom_mu != 0 else 0.5

                # Compute chain-products for non-membership
                prod_nu = 1.0
                prod_1nu = 1.0
                for t in range(i+1, k):
                    prod_nu   *= (R[i, t, 1] * R[t, k, 1])
                    prod_1nu  *= ((1 - R[i, t, 1]) * (1 - R[t, k, 1]))
                left_nu  = prod_nu ** exponent
                right_nu = prod_1nu ** exponent
                denom_nu = left_nu + right_nu
                new_nu = left_nu / denom_nu if denom_nu != 0 else 0.5

                R_bar[i, k] = [new_mu, new_nu]

    # 3. Fill the lower-triangular part using reciprocity:
    # If R_bar[i,k] = (μ, ν), then set R_bar[k,i] = (ν, μ).
    for i in range(n):
        for k in range(i+1, n):
            mu_val, nu_val = R_bar[i, k]
            R_bar[k, i] = [nu_val, mu_val]

    return R_bar