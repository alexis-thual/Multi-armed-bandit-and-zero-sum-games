import numpy as np
import matplotlib.pyplot as plt

from normalFormGame import NormalFormGame


# %% Player cell
class Player:
    def __init__(self, model, playerIndex, verbose=False, strategy='UCB'):
        self.verbose = verbose
        self.strategy = strategy
        self.model = model
        self.nActions = model.nActions[playerIndex]
        self.playerIndex = playerIndex
        self.actions = []
        self.regrets = []

        # UCB attributes
        self.explorationTime = self.nActions
        self.S = np.zeros(self.nActions)
        self.N = np.zeros(self.nActions)
        self.rho = 0.2

    def chooseAction(self):
        action = np.random.randint(self.nActions)

        if self.strategy == 'UCB':
            t = len(self.actions)
            action = t if (t < self.explorationTime) else np.argmax(self.S / self.N + self.rho * np.sqrt(np.log(t) / (2*self.N)))
            self.N[action] += 1

        return action

    def step(self):
        action = self.chooseAction()
        self.actions.append(action)

        return action

    def analyzeStep(self, playerAction, opponentAction, reward):
        regret = self.model.bestReward(self.playerIndex, opponentAction) - reward
        self.regrets.append(regret)

        if self.strategy == 'UCB':
            self.S[playerAction] += reward

    def plotAnalysis(self):
        # plt.figure(1, figsize=(13,8))
        plt.figure(1)
        plt.plot(self.regrets)
        plt.show()
