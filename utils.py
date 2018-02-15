import numpy as np

def make_tableau(M):
    return np.append(np.append(M, np.eye(M.shape[0]), axis=1),
                     np.ones((M.shape[0], 1)), axis=1)

def shift_tableau(tableau, shape):
    return np.append(np.roll(tableau[:,:-1], shape[0], axis=1),
                     np.ones((shape[0], 1)), axis=1)

def find_pivot_row(tableau, column_index):
    return np.argmax(tableau[:, column_index] / tableau[:, -1])

def non_basic_variables(tableau):
    columns = tableau[:,:-1].transpose()
    return set(np.where([np.count_nonzero(col) != 1 for col in columns])[0])

def pivot_tableau(tableau, column_index):
    original_labels = non_basic_variables(tableau)
    pivot_row_index = find_pivot_row(tableau, column_index)
    pivot_element = tableau[pivot_row_index, column_index]

    for i, _ in enumerate(tableau):
        if i != pivot_row_index:
            tableau[i, :] = (
                tableau[i, :] * pivot_element -
                tableau[pivot_row_index, :] * tableau[i, column_index])

    return non_basic_variables(tableau) - original_labels
