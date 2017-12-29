import numpy as np


# %% NFG class
class NormalFormGame:
    def __init__(self, verbose=False):
        self.verbose = verbose

        self.nActions = [2,2]
        self.matrix = np.array([
            [[-4,4], [1,-1]],
            [[1,-1], [3,-3]]
        ])

    def rewards(self, actions):
        '''
        Input: actions as a 2-uplet (a1, a2)
        Output: rewards for each player
        '''
        return self.matrix[actions]

    def bestReward(self, playerIndex, opponentAction):
        # If column-player:
        if playerIndex == 0:
            return np.max(self.matrix[opponentAction, :, playerIndex])
        else:
            return np.max(self.matrix[:, opponentAction, playerIndex])

# %% Test cell
matrix = np.array([
    [[4,3], [-1,-1]],
    [[0,0], [3,4]]
])

matrix[0,:,0]
