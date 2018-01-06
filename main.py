import numpy as np

from player_ExtensiveForm import Player as ExtensivePlayer
from games import NormalFormGame, ExtensiveFormGame


if __name__ == "__main__":
    verbose = True
    game = ExtensiveFormGame(verbose=verbose)
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
