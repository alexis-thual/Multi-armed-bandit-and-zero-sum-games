import numpy as np
import matplotlib.pyplot as plt

verbose = True
mixed = True

# False:
# beta = 0.001
# gamma = 0.1

if mixed:
    nActions = [3,3]
    matrix = np.array([
        [[0,0], [1,-1], [-1,1]],
        [[-1,1], [0,0], [1,-1]],
        [[1,-1], [-1,1], [0,0]]
    ])

else:
    nActions = [2,2]
    matrix = np.array([
        [[-4,4], [1,-1]],
        [[1,-1], [3,-3]]
    ])

def choose(probabilities):
    choice = random.uniform(0, 1)
    choiceIndex = 0

    for proba in probabilities:
        choice -= proba
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1

# p1 = np.ones(nActions[0]) / nActions[0]
p1 = np.random.random(nActions[0])
p1 /= np.sum(p1)
regrets1 = []

# p2 = np.ones(nActions[1]) / nActions[1]
p2 = np.random.random(nActions[1])
p2 /= np.sum(p2)
regrets2 = []

if verbose:
    print(p1)
    print(p2)
    print()

w1 = np.ones(nActions[0])
w2 = np.ones(nActions[1])

M1 = np.max(matrix[:,:,0])
M2 = np.max(matrix[:,:,1])
m1 = np.min(matrix[:,:,0])
m2 = np.min(matrix[:,:,1])

beta = 0.01
gamma = 0.01

steps = 500

for step in range(steps):
    a1 = choose(p1)
    a2 = choose(p2)

    r1 = (matrix[a1, a2, 0] - m1) / (M1 - m1)
    regrets1.append((np.max(matrix[:,a2,0]) - matrix[a1, a2, 0]) / (M1 - m1))
    r2 = (matrix[a1, a2, 1] - m2) / (M2 - m2)
    regrets2.append((np.max(matrix[a1,:,1]) - matrix[a1, a2, 1]) / (M2 - m2))

    g1 = beta / p1
    g1[a1] += r1 / p1[a1]
    w1 = w1 * np.exp(gamma / nActions[0] * g1)
    # w1[a1] = w1[a1] * np.exp(gamma / nActions[0] * g1[a1])
    p1 = (1 - gamma) * w1 / np.sum(w1) + gamma / nActions[0]

    g2 = beta / p2
    g2[a2] += r2 / p2[a2]
    w2 = w2 * np.exp(gamma / nActions[1] * g2)
    # w2[a2] = w2[a2] * np.exp(gamma / nActions[1] * g2[a2])
    p2 = (1 - gamma) * w2 / np.sum(w2) + gamma / nActions[1]

print(p1)
print(p2)
plt.plot(np.cumsum(regrets1), label='Player 1')
plt.plot(np.cumsum(regrets2), label='Player 2')
plt.legend()
plt.show()
