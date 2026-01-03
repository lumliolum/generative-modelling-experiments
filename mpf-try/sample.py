# In this file, we will write code that can sample from a ising model.
import numpy as np


def sample_using_gibbs(J, nsamples, burin, istps):
    # the idea is that after burnin you will get one sample
    # then after every istps you will get a sample
    sampling_steps = burin + (nsamples - 1) * istps
    d = J.shape[0]

    # the below array tells which dimension to update at each step
    update_dims = np.random.randint(0, d, size=sampling_steps)

    # array to store samples
    X = np.zeros((nsamples, d))

    # seperate the bias or the diagonal term from J
    bias = np.diag(J)
    J = J - np.diag(bias)

    # sample a binary vector which acts as an initialization
    x = np.random.binomial(1, 0.5, size=d)

    # store the samples
    store_step = burin - 1
    store_idx = 0

    for step in range(sampling_steps):
        chosen_dim = update_dims[step]
        act = 2 * np.sum(J[chosen_dim, :] * x) + bias[chosen_dim]
        p = 1 / (1 + np.exp(act))
        # with probability p the chosen dimension is 1 or 0
        x[chosen_dim] = np.random.binomial(1, p)

        if step == store_step:
            X[store_idx, :] = x
            store_idx += 1
            store_step += istps

    return X


if __name__ == "__main__":
    # dimensions
    d = 10
    nsamples = 1000
    # number of independent steps
    istps = 10 * d
    # burn in for gibbs sampling
    burin = 100 * d

    # choose a random coupling matrix to generate
    J = 3 * np.random.normal(size=(d, d)) / np.sqrt(d)
    J = (J + J.T) / 2
    J = J - np.diag(np.diag(J))
    # replacing diagonal elements with sum of each column (but only non diagonal elements)
    J = J - np.diag(np.sum(J, axis=0))

    X = sample_using_gibbs(J, nsamples, burin, istps)
    print(X.shape)
