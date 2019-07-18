import math
import numpy as np
import random
import matplotlib.pyplot as plt


def lerp(a, b, t):
    return (b - a) * t + a


def normal_distribution_density_function(mu, sigma):
    def probability(x):
        return 1 / (math.sqrt(2 * math.pi) * sigma) * math.exp(- (x - mu) ** 2 / (2 * sigma))
    return probability


def metropolis_hastings(probability_density, min_val, max_val, k):
    p = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                p[i, j] = 1 / k * min(1, probability_density(lerp(min_val, max_val, j / k)) / probability_density(lerp(min_val, max_val, i / k)))
    for i in range(k):
        summed_probability = 0
        for j in range(k):
            if i != j:
                summed_probability += p[i, j]
        p[i, i] = 1 - summed_probability
    return p


def sample_from_vector(vec):
    x = random.random()
    for i in range(len(vec)):
        if x < vec[i]:
            return i
        x -= vec[i]


def sample(probability_density, min_val, max_val, k, n):
    p = metropolis_hastings(probability_density, min_val, max_val, k)
    samples = []
    for i in range(n):
        x = random.randint(0, k - 1)
        for j in range(2000):
            x = sample_from_vector(p[x])
        samples.append(x)
    return samples


def plot_histogram(samples, k):
    plt.hist(samples, bins=k)
    plt.show()


k = 60

plot_histogram(sample(normal_distribution_density_function(k/2, k/6), 0, k, k, 1000), k)
