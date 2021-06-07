import numpy as np


# ----------------------------------------------------------------
# Simple genetic algorithm
# ----------------------------------------------------------------

def simple_selection(scores, chromosomes, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if not np.isclose(np.sum(scores), 1):
        probabilities = scores / np.sum(scores)
    else:
        probabilities = scores
    parent1 = rng.choice(np.arange(len(scores)), p=probabilities)
    parent2 = rng.choice(np.arange(len(scores)), p=probabilities)
    while parent1 == parent2:
        parent2 = rng.choice(np.arange(len(scores)), p=probabilities)

    return chromosomes[parent1], chromosomes[parent2]


def simple_crossover(chrom1, chrom2, rng=None, crossover_rate=None):
    if rng is None:
        rng = np.random.default_rng()
    if crossover_rate is None:
        crossover_rate = 1 / chrom1.shape[0]  # adaptive
    crossover_points = rng.choice([0, 1],
                                  p=[1 - crossover_rate, crossover_rate],
                                  size=chrom1.shape[0])
    crossover_point = np.where(crossover_points == 1)[0]
    if len(crossover_point) < 1:
        new_chrom = chrom1.copy()
    else:
        crossover_point = crossover_point[0]
        new_chrom = chrom1.copy()
        new_chrom[crossover_point:] = chrom2[crossover_point:]
    return new_chrom


# mutation operator
def simple_mutation(new_chrom, mutation_rate=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if mutation_rate is None:
        mutation_rate = 1 / new_chrom.shape[0] ** 2  # adaptive
    for i in range(new_chrom.shape[0]):
        # check for a mutation
        if rng.random() < mutation_rate:
            # flip the bit
            new_chrom[i] = 1 - new_chrom[i]
    return new_chrom


# ----------------------------------------------------------------
# RGA (The Rival Genetic Algorithm)
# https://link.springer.com/content/pdf/10.1007/s11227-020-03378-9.pdf
# ----------------------------------------------------------------

def rga_selection(scores, chromosomes, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if np.sum(scores) != 1:
        probabilities = scores / np.sum(scores)
    else:
        probabilities = scores
