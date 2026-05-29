"""Genome = flat float32 weights for the MLP brain the C host runs.

Layout (must match brain.c):  W1[H*P], b1[H], W2[A*H], b2[A]
Forward pass:  h = tanh(W1 x + b1) ;  y = tanh(W2 h + b2)

The orchestrator owns genomes and performs mutation/crossover, then ships them
to a host via ASSIGN.  The C side is a pure evaluator.
"""
import numpy as np
from config import P, A, H

GENOME_LEN = H * P + H + A * H + A


def _slices():
    i = 0
    W1 = slice(i, i + H * P); i += H * P
    b1 = slice(i, i + H);     i += H
    W2 = slice(i, i + A * H); i += A * H
    b2 = slice(i, i + A)
    return W1, b1, W2, b2


def seed_forager(rng):
    """A hand-biased seed so the very first population visibly forages in 3D.

    Inputs 2,3,4 are the unit vector toward nearest food; outputs 0,1,2 are the
    thrust vector.  We route food-direction straight through to thrust, plus
    small noise for individuality, plus a positive reproduce bias (b2[3])."""
    g = np.zeros(GENOME_LEN, np.float32)
    W1s, b1s, W2s, b2s = _slices()
    W1 = np.zeros((H, P), np.float32)
    W2 = np.zeros((A, H), np.float32)
    for c in range(3):                 # hidden c carries food-dir component c
        W1[c, 2 + c] = 1.0
        W2[c, c] = 1.6                 # thrust c follows hidden c (toward food)
    # mild neighbour avoidance via spare hidden units 3,4,5
    for c in range(3):
        W1[3 + c, 6 + c] = 1.0
        W2[c, 3 + c] = -0.5
    b2 = np.zeros(A, np.float32)
    b2[3] = 0.6                        # reproduce intent > 0 by default
    g[W1s] = W1.ravel()
    g[W2s] = W2.ravel()
    g[b2s] = b2
    g += rng.normal(0, 0.05, GENOME_LEN).astype(np.float32)
    return g


def mutate(genome, rate, rng):
    child = genome.copy()
    child += rng.normal(0, rate, genome.shape).astype(np.float32)
    return child
