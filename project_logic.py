# An elevator stops at K floors
# It performs the following actions:
#   Request occurs at floor i
#   Elevator travels from i to j, where i in [1...k] and i != j
#   Algorithm selects a floor to idle, m
# The goal of the algorithm is to minimize the overall time, T
#   Let T_1: Time people wait when requesting the elevator, |i-m|
#   Let T_2: Time taken to go from floor j to the new idling floor n, |j-n|
#   T = T_1 + (T_2)/2
# The inputs for the problem consist of:
#   E: Probability vector of a person entering at floor e_i, i=1..k
#   L: Probability vector of a person exiting at floor l_i, i=1..k
# We can always assume that the elevator arrives at m before the next
# passenger requests the elevator at floor i
# Attempt to devise an expedient scheme
# Find two LA-based solutions (FSSA &|| VSSA &|| Discretized)
import random
import vssa
import fssa
import pursuit

def rand():
    return random.uniform(0,1)

def select_floor(probabilities, excluding=None):
    """Select a floor. If we must exclude one, it's an optional parameter"""
    prob = probabilities[:]
    floor_prob = rand()

    # If we are excluding a floor, remove it from the list
    if excluding is not None:
        prob.pop(excluding)
        prob = [i/sum(prob) for i in prob]

    # Perform a basic proportional search for a floor
    for i, floor in enumerate(prob):
        if floor > floor_prob:
            # If we're excluding a floor here, we need to return the index including the excluded
            if excluding is None or i < excluding:
                return i
            return i + 1
        floor_prob -= floor

def create_elevator_requestor(E, L, i=None):
    """Little recursive lambda generator which returns a tuple of selected floors according to E, L"""
    if i != None:
        return (i, select_floor(L, i))
    return lambda: create_elevator_requestor(E, L, select_floor(E))

def overall_time(start, idle, request):
    empty_to_idle = abs(start - idle)
    idle_to_passenger = abs(idle - request)
    return (float(empty_to_idle) / 2.0) + idle_to_passenger

def environment(start, idle, request, num_floors):
    """Returns penalty or not"""
    time = overall_time(start, idle, request)
    worst = (num_floors - 1) + (num_floors - 2) / 2.0
    penalty = rand() < (time / worst)
    return penalty

def dumb_simulator(E, L, P, num_floors):
    total_time = 0
    current_floor = 1
    requestor = create_elevator_requestor(E, L)
    for _ in range(10000):
        request = requestor()
        time = overall_time(current_floor, current_floor, request[0])
        total_time += time
        current_floor = request[1]
    return (total_time/10000, 0)

def dumb_bottom_simulator(E, L, P, num_floors):
    total_time = 0
    current_floor = 1
    requestor = create_elevator_requestor(E, L)
    for _ in range(10000):
        request = requestor()
        time = overall_time(current_floor, 1, request[0])
        total_time += time
        current_floor = request[1]
    return (total_time/10000, 0)

def rand_simulator(E, L, P, num_floors):
    total_time = 0
    current_floor = 1
    requestor = create_elevator_requestor(E, L)
    for _ in range(10000):
        request = requestor()
        idle = random.randint(1, num_floors)
        time = overall_time(current_floor, idle, request[0])
        total_time += time
        current_floor = request[1]
    return (total_time/10000, 0)

def solution_one(E, L, P, K):
    """What if there's one machine"""
    requestor = create_elevator_requestor(E, L)
    reward_const = 0.05
    P = P[:]
    g = vssa.lri_g
    f = lambda action, penalty, P: vssa.lri_f(action, penalty, P, reward_const)
    check_penalty = lambda start, idle, request: environment(start, idle, request, K)
    current_floor = 1
    total_time = 0
    worst = (K-1) + float(K-2) / 2.0
    count = 0
    while (max(P) < 0.98):
        idle = g(P)
        request = requestor()
        T = overall_time(current_floor, idle, request[0])
        penalty = check_penalty(current_floor, idle, request[0])
        P = f(idle, penalty, P)
        current_floor = request[1]
        count += 1
    for i in range(10000):
        idle = g(P)
        request = requestor()
        time = overall_time(current_floor, idle, request[0])
        total_time += time
        current_floor = request[1]
    # Let's find some good penalty and reward values
    return (float(total_time) / 10000, count)

def solution_two(E, L, P, K):
    """What if we use K machines"""
    reward_const = 0.05
    requestor = create_elevator_requestor(E, L)
    g = vssa.lri_g
    f = lambda action, penalty, P: vssa.lri_f(action, penalty, P, reward_const)
    check_penalty = lambda start, idle, request: environment(start, idle, request, K)
    floors = [[1]]
    # Populate each floor with a probability distribution
    for _ in range(K):
        floors.append(P[:])
    current_floor = 1

    # Complex way of saying that every floor must have converged
    worst = (K-1) + (K-2) / 2.0

    count = 0

    while min( [max(floor) for floor in floors] ) < 0.98:
        # Pick a floor to idle at
        idle = g(floors[current_floor])
        # Request comes in
        request = requestor()
        penalty = check_penalty(current_floor, idle, request[0])
        floors[current_floor] = f(idle, penalty, floors[current_floor])
        current_floor = request[1]
        count += 1
    current_floor = 1
    total_time = 0
    for _ in range(10000):
        idle = floors[current_floor].index(max(floors[current_floor]))
        request = requestor()
        time = overall_time(current_floor, idle, request[0])
        current_floor = request[1]
        total_time += time
    return (float(total_time) / 10000, count)

def solution_three(E, L, P, K):
    """What if we use K machines, with an L_RI pursuit algorithm?"""
    reward_const = 0.01
    requestor = create_elevator_requestor(E, L)
    g = pursuit.lri_g
    f = lambda penalty, P, A: pursuit.lri_f(penalty, P, A, reward_const)
    check_penalty = lambda start, idle, request: environment(start, idle, request, K)
    floors_P = [[1]]

    # Now, each floor must store an approximation vector. Each vector contains a dict for each floor
    floors_A = []
    floor = []

    count = 0

    for _ in range(K):
        floor.append({'reward':0, 'total':0})
    floors_A.append(floor)
    # Populate each floor with a probability distribution
    for _ in range(K):
        floors_P.append(P[:])
        floor = []
        for _ in range(K+1):
            floor.append({'reward':0, 'total':0})
        floors_A.append(floor)

    # Now we need to populate our approximation. ~5 each should suffice, for each floor
    for current_floor in range(1, K):
        for idle in range(1, K):
            for _ in range(10):
                request = requestor()
                penalty = check_penalty(current_floor, idle, request[0])
                floors_A[current_floor][idle]['reward'] += not penalty
                floors_A[current_floor][idle]['total'] += 1
                count += 1
    current_floor = 1

    # Complex way of saying that every floor must have converged
    worst = (K-1) + (K-2) / 2.0
    while min( [max(floor) for floor in floors_P] ) < 0.98:
        # Pick a floor to idle at
        idle = g(floors_P[current_floor])
        # Request comes in
        request = requestor()
        penalty = check_penalty(current_floor, idle, request[0])
        if current_floor > len(floors_A):
            print("CURRENT")
            print(floors_A)
        if idle > len(floors_A[current_floor]):
            print("IDLE")
            print(current_floor)
            print(floors_A[current_floor])
        
        floors_A[current_floor][idle]['reward'] += not penalty
        floors_A[current_floor][idle]['total'] += 1
        floors_P[current_floor] = f(penalty, floors_P[current_floor], floors_A[current_floor])
        current_floor = request[1]
        count += 1
    current_floor = 1
    total_time = 0
    
    for _ in range(10000):
        idle = floors_P[current_floor].index(max(floors_P[current_floor]))
        request = requestor()
        time = overall_time(current_floor, idle, request[0])
        current_floor = request[1]
        total_time += time
    return (float(total_time) / 10000, count)

def get_low_heavy_vectors(K):
    """Given K floors, generate vectors E,L,P which have bottom heavy exits (L)"""
    E = [rand() for i in range(K)]
    E = [i/sum(E) for i in E]
    E.insert(0, 0)

    L = [rand() for i in range(K-1)]
    L.insert(0, sum(L)*10)
    L = [i/sum(L) for i in L]
    L.insert(0, 0)

    # Initial probability
    P = [1 for i in range(K)]
    P = [float(i)/sum(P) for i in P]
    P.insert(0, 0)

    return (E, L, P)

def get_random_vectors(K):
    """Given K floors, generate vectors E,L,P which have bottom heavy exits (L)"""
    E = [rand() for i in range(K)]
    E = [i/sum(E) for i in E]
    E.insert(0, 0)

    L = [rand() for i in range(K)]
    L = [i/sum(L) for i in L]
    L.insert(0, 0)

    # Initial probability
    P = [1 for i in range(K)]
    P = [float(i)/sum(P) for i in P]
    P.insert(0, 0)

    return (E, L, P)

def get_set_vectors():
    """Given K floors, generate vectors E,L,P which have bottom heavy exits (L)"""
    E = [0, 0.06, 0.12, 0.31, 0.38, 0.13]

    L = [0, 0.31, 0.12, 0.06, 0.12, 0.38]

    # Initial probability
    P = [1 for i in range(len(E)-1)]
    P = [float(i)/sum(P) for i in P]
    P.insert(0, 0)

    return (E, L, P)

def get_set10_vectors():
    """Given K floors, generate vectors E,L,P which have bottom heavy exits (L)"""
    E = [0, 0.03, 0.03, 0.1, 0.13, 0.17, 0.23, 0.15, 0.1, 0.03, 0.03]

    L = [0, 0.13, 0.15, 0.23, 0.03, 0.1, 0.03, 0.1, 0.17, 0.03, 0.03]

    # Initial probability
    P = [1 for i in range(len(E)-1)]
    P = [float(i)/sum(P) for i in P]
    P.insert(0, 0)

    return (E, L, P)