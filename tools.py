import numpy as np

def Nash_equilibrium(rewards):
    existence_Nash_equilibrium = False
    Nash_equilibriums = []
    nb_actions = len(rewards)
    for i in range(nb_actions):
        for j in range(nb_actions):
            if rewards[i,j,0] == max(rewards[:,j,0]) and rewards[i,j,1] == max(rewards[i,:,1]):
                existence_Nash_equilibrium = True
                Nash_equilibriums.append((i,j))

    return existence_Nash_equilibrium, Nash_equilibriums

def has_converged(p1,p2):
    n_iter_conv = 20
    if np.unique(p1.actions[-n_iter_conv:]).size == 1:
        p1_converged_to = np.unique(p1.actions[-n_iter_conv:])[0]
        if np.unique(p2.actions[-n_iter_conv:]).size == 1:
            p2_converged_to = np.unique(p2.actions[-n_iter_conv:])[0]
            return True, (p1_converged_to,p2_converged_to)
    return False, (-1,-1)
