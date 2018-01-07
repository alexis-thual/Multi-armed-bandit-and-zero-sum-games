import numpy as np
import random


# %% NFG class
class NormalFormGame:
    def __init__(self, verbose=False, DeterministicReward = True):
        self.verbose = verbose

        self.nActions = [2,2]
        self.matrix = np.array([
            [[-4,4], [1,-1]],
            [[1,-1], [3,-3]]
        ])
        self.DeterministicReward = DeterministicReward

    def rewards(self, actions):
        '''
        Input: actions as a 2-uplet (a1, a2)
        Output: rewards for each player
        '''
        if self.DeterministicReward:
            rewards = self.matrix[actions]
        else:
            rewards = np.random.normal(loc=self.matrix[actions], scale = [0.5,0.5])
        return rewards

    def bestReward(self, playerIndex, opponentAction):
        # If column-player:
        if playerIndex == 0:
            return np.max(self.matrix[opponentAction, :, playerIndex])
        else:
            return np.max(self.matrix[:, opponentAction, playerIndex])


class ExtensiveFormGame:
    ## The markets for lemon
    def __init__(self, verbose=False, DeterministicReward = True):
        self.verbose = verbose

        self.nActions = [2,2]
        self.matrix = np.array([[ #Bad Car
            [[-1,1], [0,0]],        #Low price, Buy or not
            [[2,-2], [1,-1]]        #High price, Buy or not
        ],[                       #Good Car
            [[-2,2], [-1,1]],        #Low price, Buy or not
            [[1,-1], [0,0]]         #High price, Buy or not
        ]])
        self.goodCar = random.randint(0, 1)
        self.DeterministicReward = DeterministicReward

    def rewards(self, actions):
        '''
        Input: actions as a 2-uplet (actionSeller, actionBuyer)
        Output: rewards for each player
        '''
        if self.DeterministicReward:
            rewards = self.matrix[self.goodCar, actions[0], actions[1]]
        else:
            rewards = np.random.normal(loc=self.matrix[self.goodCar, actions[0], actions[1]], scale = [0.5,0.5])
        return rewards

    def newDeal(self):
        self.goodCar = random.randint(0, 1)
        return self.goodCar

    def bestReward(self, playerIndex, opponentAction):
        if playerIndex == 0:
            return np.max(self.matrix[self.goodCar, :, opponentAction, playerIndex])
        else:
            return np.max(self.matrix[self.goodCar, opponentAction, :, playerIndex])


if __name__ == "__main__":
    a = ExtensiveFormGame(DeterministicReward = False)
    print(a.rewards((0,1)))
