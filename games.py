import numpy as np
import random


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


class ExtensiveFormGame:
    ## The markets for lemon
    def __init__(self, verbose=False):
        self.verbose = verbose

        self.nActions = [2,2]
        self.matrix = np.array([[ #Bad Car
            [[1,1], [0,0]],        #Low price, Buy or not
            [[2,-2], [0,0]]        #High price, Buy or not
        ],[                       #Good Car
            [[-2,2], [0,0]],        #Low price, Buy or not
            [[1,1], [0,0]]         #High price, Buy or not
        ]])
        self.goodCar = random.randint(0, 1)

    def rewards(self, actions):
        '''
        Input: actions as a 2-uplet (actionSeller, actionBuyer)
        Output: rewards for each player
        '''
        return self.matrix[self.goodCar, actions[0], actions[1]]

    def newDeal(self):
        self.goodCar = random.randint(0, 1)
        return self.goodCar

    def bestReward(self, playerIndex, opponentAction):
        if playerIndex == 0:
            return np.max(self.matrix[self.goodCar, :, opponentAction, playerIndex])
        else:
            return np.max(self.matrix[self.goodCar, opponentAction, :, playerIndex])


if __name__ == "__main__":
    a = ExtensiveFormGame()
    print(a.goodCar)
    print(a.bestReward(1, 0))
