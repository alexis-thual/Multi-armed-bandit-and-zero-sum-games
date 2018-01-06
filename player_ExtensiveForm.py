import numpy as np
import matplotlib.pyplot as plt
import random

from games import ExtensiveFormGame


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
        self.nbInformation = 2
        self.actionsByInfo = [[],[]]

        # UCB attributes
        self.explorationTime = self.nActions
        self.S = np.zeros((self.nbInformation, self.nActions))
        self.N = np.zeros((self.nbInformation, self.nActions))
        self.rho = 0.2

    def chooseAction(self, information):
        action = np.random.randint(self.nActions)

        if self.strategy == 'UCB':
            t = len(self.actions)
            action = random.randint(0,self.nActions-1) if (t < 4*self.explorationTime) else np.argmax(self.S[information] / self.N[information] + self.rho * np.sqrt(np.log(t) / (2*self.N[information])))
            self.N[information,action] += 1

        return action

    def step(self, information):
        action = self.chooseAction(information)
        self.actions.append(action)
        self.actionsByInfo[information].append(action)

        return action

    def analyzeStep(self, playerAction, opponentAction, rewards, information):
        reward = rewards[self.playerIndex]
        regret = self.model.bestReward(self.playerIndex, opponentAction) - reward
        self.regrets.append(regret)
        if self.strategy == 'UCB':
            self.S[information,playerAction] += reward

    def plotAnalysis(self, player):
        # plt.figure(1, figsize=(13,8))
        plt.figure(1)
        plt.plot(self.regrets)
        plt.title("Regrets for %s"%(player))
        plt.show()
        plt.figure(1)
        plt.plot(self.actionsByInfo[0])
        plt.title("Action for info = 0 for %s"%(player))
        plt.show()
        plt.figure(1)
        plt.plot(self.actionsByInfo[1])
        plt.title("Action for info = 1 for %s"%(player))
        plt.show()
