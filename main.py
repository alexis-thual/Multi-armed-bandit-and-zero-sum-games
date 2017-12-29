import numpy as np

from player import Player
from normalFormGame import NormalFormGame


if __name__ == "__main__":
    verbose = True
    game = NormalFormGame(verbose=verbose)
    player0 = Player(game, 0, verbose=verbose)
    player1 = Player(game, 1, verbose=verbose)

    nStep = 20
    for _ in range(nStep):
        a0 = player0.step()
        a1 = player1.step()

        rewards = game.rewards((a0, a1))

        player0.analyzeStep(a0, a1, rewards[0])
        player1.analyzeStep(a1, a0, rewards[1])
        print(a0, a1)

    player0.plotAnalysis()
