import numpy as np
import matplotlib.pyplot as plt

from games import NormalFormGame


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
            action = t if (t < self.nActions) else np.argmax(self.S / self.N + self.rho * np.sqrt(np.log(t) / (2*self.N)))

        elif self.strategy == 'Thompson sampling':
            t = len(self.actions)
            action = t if (t < self.nActions) else np.argmax(np.random.beta((self.S+4)/8+1, self.N - (self.S+4)/8 + 1))

        elif self.strategy == 'Naive':
            t = len(self.actions)
            action = t if (t < self.nActions) else np.argmax(self.S/self.N)
        else :
            print("strategy not implemented : random policy applied -- choose 'UCB', 'Thompson sampling' or 'Naive'")

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


    def plotAnalysis(self, player):
        plt.figure(1)
        plt.plot(np.cumsum(self.regrets))
        plt.title("Regrets for %s"%(player))
        plt.show()
        plt.figure(1)
        plt.plot(self.actions)
        plt.title("Actions for %s"%(player))
        plt.show()
