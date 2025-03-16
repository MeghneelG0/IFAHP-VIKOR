from .utility_measures import calculate_s, calculate_r, calculate_q
def rank_alternatives(Q, S, R):
    """
    Ranks alternatives based on Q, S, and R.
    Args:
    - Q: The compromise solution for each alternative.
    - S: The utility measure for each alternative.
    - R: The regret measure for each alternative.
    Returns:
    - rankings: A list of tuples containing the alternative index and its Q, S, and R values, sorted by Q.
    """
    rankings = sorted(enumerate(zip(Q, S, R)), key=lambda x: (x[1][0], x[1][1], x[1][2]))
    return rankings