from utils import (make_tableau, shift_tableau, non_basic_variables,
                    pivot_tableau)

import warnings

import numpy as np
from itertools import cycle

def tableau_to_strategy(tableau, basic_labels, strategy_labels):
    vertex = []
    for column in strategy_labels:
        if column in basic_labels:
            for i, row in enumerate(tableau[:, column]):
                if row != 0:
                    vertex.append(tableau[i, -1] / row)
        else:
            vertex.append(0)

    strategy = np.array(vertex)
    return strategy / sum(strategy)

def lemke_howson(A, B, initial_dropped_label=0):
    if np.min(A) <= 0:
        A = A + abs(np.min(A)) + 1
    if np.min(B) <= 0:
        B = B + abs(np.min(B)) + 1

    # build tableaux
    col_tableau = make_tableau(A)
    col_tableau = shift_tableau(col_tableau, A.shape)
    row_tableau = make_tableau(B.transpose())
    full_labels = set(range(sum(A.shape)))

    if initial_dropped_label in non_basic_variables(row_tableau):
        tableux = cycle((row_tableau, col_tableau))
    else:
        tableux = cycle((col_tableau, row_tableau))

    # First pivot (to drop a label)
    entering_label = pivot_tableau(next(tableux), initial_dropped_label)
    while non_basic_variables(row_tableau).union(non_basic_variables(col_tableau)) != full_labels:
        entering_label = pivot_tableau(next(tableux), next(iter(entering_label)))

    row_strategy = tableau_to_strategy(row_tableau, non_basic_variables(col_tableau),
                                       range(A.shape[0]))
    col_strategy = tableau_to_strategy(col_tableau, non_basic_variables(row_tableau),
                                       range(A.shape[0], sum(A.shape)))

    return row_strategy, col_strategy

# %% Test cell
lemke_howson(A,B)


# %% Test cell
matrix = np.array([
    [[0,0], [1,-1], [-1,1]],
    [[-1,1], [0,0], [1,-1]],
    [[1,-1], [-1,1], [0,0]]
])

# matrix = np.array([
#     [[-4,4], [1,-1]],
#     [[1,-1], [3,-3]]
# ])

A = matrix[:,:,0]
A
B = matrix[:,:,1]


# %% Test cell
if np.min(A) <= 0:
    A = A + abs(np.min(A)) + 1
if np.min(B) <= 0:
    B = B + abs(np.min(B)) + 1

A

# build tableaux
col_tableau = make_tableau(A)
col_tableau = shift_tableau(col_tableau, A.shape)
row_tableau = make_tableau(B.transpose())
full_labels = set(range(sum(A.shape)))

col_tableau
row_tableau
full_labels
non_basic_variables(row_tableau)

columns = row_tableau[:,:-1].transpose()
columns
[np.count_nonzero(col) != 1 for col in columns]

initial_dropped_label=0
tableux = cycle((row_tableau, col_tableau))
entering_label = pivot_tableau(next(tableux), initial_dropped_label)
next(iter(entering_label))
