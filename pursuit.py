import random

def lri_g(P):
    """In pursuit, we choose based on our approximations"""
    rng = random.uniform(0,1)
    P_normal = [i/sum(P) for i in P]

    for i, action in enumerate(P_normal):
        if action > rng:
            return i
        rng -= action
    return len(P) - 1

def lri_f(penalty, P, A, r):
    """Update P with vector addition based on notes for pursuit"""
    P = P[:]
    if penalty:
        return P

    # First we have to find the best approximation thus far

    best_ratio = 0
    best_action = 1

    for i, action in enumerate(A):
        if action['total'] == 0:
            continue
        ratio = float(action['reward']) / action['total']
        if ratio > best_ratio:
            best_ratio = ratio
            best_action = i

    # This creates an identity matrix for the action'd element
    # Multiply each element (the one) by 1-Kr (or, r)
    identity = [ int(i == best_action) for i, _ in enumerate(P)]
    identity = [i * (r) for i in identity]

    # Multiply each element in the P vector by Kr (or, 1-r)
    P = [i * (1-r) for i in P]

    # Add the two vectors together
    P = [sum(a) for a in zip(identity, P)]
    return P