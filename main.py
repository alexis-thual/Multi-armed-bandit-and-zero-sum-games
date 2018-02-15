import numpy as np

from player_ExtensiveForm import Player as ExtensivePlayer
from player_normalForm import Player as NormalPlayer
from games import NormalFormGame, ExtensiveFormGame
from tools import *

#gameType = "Normal"
gameType = "Extensive"

#RandomInitialization = True
RandomInitialization = False

#strategy = 'UCB'
strategy = 'Exp3'
#strategy = 'Thompson sampling'
#strategy = 'Naive'

#DeterministicReward = False
DeterministicReward = True

mixed = False



if __name__ == "__main__":
    if gameType == "Normal":
        verbose = True
        game = NormalFormGame(verbose=verbose, DeterministicReward = DeterministicReward, random = RandomInitialization, nb_actions=2, mixed = mixed)
        p1 = NormalPlayer(game, 0, verbose=verbose, strategy = strategy)
        p2 = NormalPlayer(game, 1, verbose=verbose, strategy = strategy)

        nStep = 5000
        for _ in range(nStep):
            a1 = p1.step()
            a2 = p2.step()

            rewards = game.rewards((a1, a2))

            p1.analyzeStep(a1, a2, rewards)
            p2.analyzeStep(a2, a1, rewards)
        NE, NEs = Nash_equilibrium(game.matrix)
        # if NE:
        #     print("Nash equilibrium(s) : %s"%(str(NEs)))
        # else:
        #     print("No Nash equilibrium")
        # algo_convergence, converge_points = has_converged(p1,p2)
        # if algo_convergence:
        #     print("Algo converged to : %s"%(str(converge_points)))
        # else:
        #     print("No convergence")
        #
        # if converge_points in NEs:
        #     print("Convergence to Nash equilibrium !!!")

        print(p1.probabilityDistibution)
        print(p2.probabilityDistibution)

        # p1.plotAnalysis("First player")
        # p2.plotAnalysis("Second player")




    if gameType == "Extensive":
        verbose = True
        game = ExtensiveFormGame(verbose=verbose, DeterministicReward = DeterministicReward, random = RandomInitialization, nb_info = 4, nb_actions=4)
        seller = ExtensivePlayer(game, 0, verbose=verbose, strategy = strategy)
        buyer = ExtensivePlayer(game, 1, verbose=verbose, strategy = strategy)

        nStep = 5000
        for _ in range(nStep):
            info = game.newDeal()
            actionSeller = seller.step(info)
            actionBuyer = buyer.step(actionSeller)

            rewards = game.rewards((actionSeller, actionBuyer))

            seller.analyzeStep(actionSeller, actionBuyer, rewards, info)
            buyer.analyzeStep(actionBuyer, actionSeller, rewards, actionSeller)

        print(seller.probabilityDistibution)
        print(buyer.probabilityDistibution)

        seller.plotAnalysis("seller")
        buyer.plotAnalysis("buyer")
