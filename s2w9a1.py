import random
import matplotlib.pyplot as plt
import math

def random_pairs(R):
    pairs = []
    choices = range(1, len(R)+1)
    
    for i in range(3):
        partner1 = random.choice(choices)
        choices.remove(partner1)
        partner2 = random.choice(choices)
        choices.remove(partner2)
        
        pairs.append([partner1, partner2])
    
    return pairs

def choose_ij(R, pairs):
    choices = range(1, len(R)+1)
    
    i = random.choice(choices)
    choices.remove(i)
    j = random.choice(choices)
    choices.remove(j)
    
    return i, j

def swap(R, pairs):
    i, j = choose_ij(R, pairs)
    newPairs = list(pairs)
    
    for pair in newPairs:
        if i in pair:
            in_i = pair.index(i)
            in_pair1 = newPairs.index(pair)
        if j in pair:
            in_j = pair.index(j)
            in_pair2 = newPairs.index(pair)
    
    newPairs[in_pair1][in_i], newPairs[in_pair2][in_j] = newPairs[in_pair2][in_j], newPairs[in_pair1][in_i]
    return newPairs

def calc_D(R, pairs):
    total = 0
    totals = []
    
    for pair in pairs:
        i = pair[0] - 1
        j = pair[1] - 1
        
        total += R[i][j]
        totals.append(R[i][j])
        total += R[j][i]
        totals.append(R[j][i])
    
    D = total / float(len(R))
    print [D, max(totals), min(totals)]
    return [D, max(totals), min(totals)]

def hill_climbing(R, pairs):
    D = calc_D(R, pairs)
    newPairs = swap(R, pairs)
    
    newD = calc_D(R, newPairs)
    
    while D < newD:
        newPairs = swap(R, pairs)
        newD = calc_D(R, newPairs)
    
    return D

def calc_I(R, pairs):
    total = 0
    
    for pair in pairs:
        i = pair[0] - 1
        j = pair[1] - 1
        
        total += R[i][j] ** 2
        total += R[j][i] ** 2
    
    I = total / float(len(R)) - ((calc_D(R, pairs)[0]) ** 2)
    
    return I

R = [[6, 3, 1, 2, 5, 4],
    [4, 6, 5, 3, 2, 1],
    [4, 1, 6, 2, 3, 5],
    [5, 2, 3, 6, 1, 4],
    [2, 3, 1, 4, 6, 5],
    [2, 5, 3, 4, 1, 6]]

#pairs = random_pairs(R)
pairs = [[[1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 5], [4, 6]],
        [[1, 2], [3, 6], [4, 5]],
        [[1, 3], [2, 4], [5, 6]],
        [[1, 3], [2, 5], [4, 6]],
        [[1, 3], [2, 6], [4, 5]],
        [[1, 4], [2, 3], [5, 6]],
        [[1, 4], [2, 5], [3, 6]],
        [[1, 4], [2, 6], [3, 5]],
        [[1, 5], [2, 3], [4, 6]],
        [[1, 5], [2, 4], [3, 6]],
        [[1, 5], [2, 6], [3, 4]],
        [[1, 6], [2, 3], [4, 5]],
        [[1, 6], [2, 4], [3, 5]],
        [[1, 6], [2, 5], [3, 4]]]

Is = []
maxDs = []
minDs = []
Ds = []

for pair in pairs:
    Is.append(calc_I(R, pair))
    Ds.append(calc_D(R, pair)[0])
    maxDs.append(calc_D(R, pair)[1])
    minDs.append(calc_D(R, pair)[2])

plt.figure()
plt.scatter(Ds, Is, label="$I(F)$")
#plt.scatter(Ds, maxDs, label="$max(D)$")
#plt.plot(Ds, minDs, label="$min(D)$")
plt.legend()

plt.show()