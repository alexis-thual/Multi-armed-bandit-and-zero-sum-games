import numpy as np
import matplotlib.pyplot as plt

from games import NormalFormGame
from tools import *
import math


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
        # self.probabilityDistibution = [1/self.nActions]*self.nActions
        self.probabilityDistibution = [0.01,0.01,0.98]

        # UCB attributes
        self.explorationTime = self.nActions
        self.S = np.zeros(self.nActions)
        self.N = np.zeros(self.nActions)
        self.rho = 0.2

        #Exp3 attributes
        self.gamma = 0.2
        self.eta = self.gamma/self.nActions
        self.beta = 1
        self.weights = np.ones(self.nActions)
        self.estimatedRewards = np.zeros(self.nActions)

    def chooseAction(self):
        action = np.random.randint(self.nActions)
        if self.strategy == 'UCB':
            t = len(self.actions)
            action = t if (t < self.nActions) else np.argmax(self.S / self.N + self.rho * np.sqrt(np.log(t) / (2*self.N)))
        elif self.strategy == 'Exp3':
            action = choose(self.probabilityDistibution)
        elif self.strategy == 'Thompson sampling':
            t = len(self.actions)
            action = t if (t < self.nActions) else np.argmax(np.random.beta((self.S+4*self.N)/8+1, self.N - (self.S+4)/8 + 1))
        elif self.strategy == 'Naive':
            t = len(self.actions)
            action = t if (t < self.nActions) else np.argmax(self.S/self.N)
        else :
            print("strategy not implemented : random policy applied -- choose 'UCB', 'Exp3', 'Thompson sampling' or 'Naive'")
        return action

    def step(self):
        action = self.chooseAction()
        self.actions.append(action)
        self.N[action] += 1
        return action

    def analyzeStep(self, playerAction, opponentAction, rewards):
        reward = rewards[self.playerIndex]
        regret = self.model.bestReward(self.playerIndex, opponentAction) - reward
        self.regrets.append(regret)
        self.S[playerAction] += reward

        if self.strategy == 'Exp3':
            maxreward = 1
            minreward = -1
            # estimatedReward = reward//self.probabilityDistibution[playerAction]
            # self.weights[playerAction] *= np.exp(self.gamma/self.nActions*estimatedReward)
            for a in range(self.nActions):
                self.estimatedRewards[a] = self.beta/self.probabilityDistibution[a]
            self.estimatedRewards[playerAction] += (maxreward - reward)/(maxreward-minreward)/self.probabilityDistibution[playerAction]
            self.weights = self.weights * np.exp(self.eta * self.estimatedRewards)
            self.weights /= np.sum(self.weights)
            self.probabilityDistibution = (1 - self.gamma) * self.weights + self.gamma/self.nActions


    def plotAnalysis(self, player):
        plt.figure(1)
        plt.plot(np.cumsum(self.regrets))
        plt.title("Regrets for %s"%(player))
        plt.show()
        plt.figure(1)
        plt.plot(self.actions)
        plt.title("Actions for %s"%(player))
        plt.show()
