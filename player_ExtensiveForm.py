import numpy as np
import matplotlib.pyplot as plt
import random

from games import ExtensiveFormGame
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
        if playerIndex == 0:
            self.nbInformation = model.nInfo
        else:
            self.nbInformation = model.nActions[0]

        self.actionsByInfo = []
        for i in range(self.nbInformation):
            self.actionsByInfo.append([])
        # UCB attributes
        self.explorationTime = self.nActions
        self.S = np.zeros((self.nbInformation, self.nActions))
        self.N = np.zeros((self.nbInformation, self.nActions))
        self.rho = 0.2

        #Exp3 attributes
        self.gamma = 0.01
        self.eta = self.gamma/self.nActions
        self.beta = 0.01
        self.weights = np.ones(self.nActions)

        self.probabilityDistibution = np.random.random((self.nbInformation,self.nActions))
        for i in range(self.nbInformation):
            self.probabilityDistibution[i] /= np.sum(self.probabilityDistibution[i])

    def chooseAction(self, information):
        action = np.random.randint(self.nActions)

        if self.strategy == 'UCB':
            t = len(self.actionsByInfo[information])
            action = t if (t < self.nActions) else np.argmax(self.S[information] / self.N[information] + self.rho * np.sqrt(np.log(t) / (2*self.N[information])))
        elif self.strategy == 'Exp3':
            action = choose(self.probabilityDistibution[information])
        elif self.strategy == 'Thompson sampling':
            t = len(self.actionsByInfo[information])
            action = t if (t < self.nActions) else np.argmax(np.random.beta((self.S[information]+4*self.N[information])/8+1, self.N[information] - (self.S[information]+4)/8 + 1))
        elif self.strategy == 'Naive':
            t = len(self.actionsByInfo[information])
            action = t if (t < self.nActions) else np.argmax(self.S[information]/self.N[information])
        else :
            print("strategy not implemented : random policy applied -- choose 'UCB', 'Thompson sampling' or 'Naive'")

        return action

    def step(self, information):
        action = self.chooseAction(information)
        self.actions.append(action)
        self.actionsByInfo[information].append(action)
        self.N[information,action] += 1
        return action

    def analyzeStep(self, playerAction, opponentAction, rewards, information):
        reward = rewards[self.playerIndex]
        regret = self.model.bestReward(self.playerIndex, opponentAction) - reward
        self.regrets.append(regret)
        self.S[information,playerAction] += reward

        if self.strategy == 'Exp3':
            maxreward = np.max(self.model.matrix[information,:,:,self.playerIndex])
            minreward = np.min(self.model.matrix[information,:,:,self.playerIndex])

            estimatedRewards = self.beta/self.probabilityDistibution[information]
            estimatedRewards[playerAction] += (reward - minreward)/(maxreward-minreward)/self.probabilityDistibution[information,playerAction]
            self.weights = self.weights * np.exp(self.eta * estimatedRewards)
            self.probabilityDistibution[information] = (1 - self.gamma) * self.weights/ np.sum(self.weights) + self.gamma/self.nActions

    def plotAnalysis(self, player):
        plt.figure(1)
        plt.plot(np.cumsum(self.regrets))
        plt.title("Regrets for %s"%(player))
        plt.show()
        for i in range(self.nbInformation):
            plt.figure(1)
            plt.plot(self.actionsByInfo[i])
            plt.title("Action for info = %i for %s"%(i,player))
            plt.show()
