import random
import time

def create_environment(C):
    return lambda a: 1 if random.uniform(0, 1) < C[a] else 0

def lri_g(P):
    """How actions are selected"""
    rng = random.uniform(0,1)
    P_normal = [i/sum(P) for i in P]

    for i, action in enumerate(P_normal):
        if action > rng:
            return i
        rng -= action
    return len(P) - 1

def lri_f(action, penalty, P, r):
    """Update P via vector addition. Note: No numpy to reduce package requirements"""
    P = P[:]
    if penalty:
        return P
    # This creates an identity matrix for the action'd element
    # Multiply each element (the one) by 1-Kr (or, r)
    identity = [ int(i == action) for i, _ in enumerate(P)]
    identity = [i * (r) for i in identity]

    # Multiply each element in the P vector by Kr (or, 1-r)
    P = [i * (1-r) for i in P]

    # Add the two vectors together
    P = [sum(a) for a in zip(identity, P)]
    return P

def lrp_f(action, penalty, P, r, p):
    """How P is updated"""
    if not penalty:
        P[action] = P[action] + r*(1-P[action])
        for j in range(1, len(P)):
            if j != action:
                P[j] = (1-r) * P[j]
    else:
        P[action] = (1-p) * P[action]
        for j in range(1, len(P)):
            if j != action:
                P[j] = (p / (len(P) - 1)) + (1 - p) * P[j]
    return P

def absorbing_simulation(environment, f, g, reps):
    count = [0, 0, 0]
    total = 0
    iterations = 0
    elapsed = 0
    for _ in range(reps):
        P = [0, 0.5, 0.5]
        # Get the automata into a converged state
        iterations = 0
        start = time.time()
        while P[1] < 0.98 and P[2] < 0.98:
            action = g(P)
            penalty = environment(action)
            P = f(action, penalty, P)
            iterations += 1
        elapsed += time.time() - start
        count[0] += P[0]
        count[1] += P[1]
        count[2] += P[2]
        total += iterations
    return (count[1] / reps, count[2] / reps, total / reps, elapsed / reps)