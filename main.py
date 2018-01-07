import numpy as np

from player_ExtensiveForm import Player as ExtensivePlayer
from player_normalForm import Player as NormalPlayer
from games import NormalFormGame, ExtensiveFormGame

gameType = "Normal"
#gameType = "Extensive"

DeterministicReward = False
#DeterministicReward = True

if __name__ == "__main__":
    if gameType == "Normal":
        verbose = True
        game = NormalFormGame(verbose=verbose, DeterministicReward = DeterministicReward)
        p1 = NormalPlayer(game, 0, verbose=verbose)
        p2 = NormalPlayer(game, 1, verbose=verbose)

        nStep = 100
        for _ in range(nStep):
            a1 = p1.step()
            a2 = p2.step()

            rewards = game.rewards((a1, a2))

            p1.analyzeStep(a1, a2, rewards)
            p2.analyzeStep(a2, a1, rewards)

        p1.plotAnalysis("First player")
        p2.plotAnalysis("Second player")

    if gameType == "Extensive":
        verbose = True
        game = ExtensiveFormGame(verbose=verbose, DeterministicReward = DeterministicReward)
        seller = ExtensivePlayer(game, 0, verbose=verbose)
        buyer = ExtensivePlayer(game, 1, verbose=verbose)

        nStep = 100
        for _ in range(nStep):
            car = game.newDeal()
            actionSeller = seller.step(car)
            actionBuyer = buyer.step(actionSeller)

            rewards = game.rewards((actionSeller, actionBuyer))
            print(rewards)

            seller.analyzeStep(actionSeller, actionBuyer, rewards, car)
            buyer.analyzeStep(actionBuyer, actionSeller, rewards, actionSeller)
            print(actionSeller, actionBuyer)

        seller.plotAnalysis("seller")
        buyer.plotAnalysis("buyer")
