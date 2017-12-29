import numpy as np


# %% NFG class
class NormalFormGame:
    def __init__(self, verbose=True):
        self.matrix = np.array([
            [[4,3], [-1,-1]],
            [[0,0], [3,4]]
        ])

    def rewards(self, actions):
        '''
        Input:
            actions as a k-uplet (a1, ..., ak)
        Output:
            rewards for each players
        '''
        return self.matrix[actions]
