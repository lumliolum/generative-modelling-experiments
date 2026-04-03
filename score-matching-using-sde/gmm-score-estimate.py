import time

import torch
import numpy as np
import scienceplots
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models import MLP
from utils import GaussianMixture

# use style
plt.style.use(["science", "no-latex"])


if __name__ == "__main__":
    # seed and device
    seed = 42
    device = torch.device("cuda:0")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # define the initial parameters of the gaussian mixture model
    initial_alphas = np.array([0.5, 0.5])
    initial_means = np.array([[-2, -2], [2, 2]])
    initial_covs = np.array([
        [[0.5, 0],[0, 0.5]],
        [[0.5, 0], [0, 0.5]]
    ])

    # don't know how much I will use but let's see
    initial_gmm = GaussianMixture(initial_alphas, initial_means, initial_covs)

    # samples
    training_samples = initial_gmm.sample(5000)

    # we run the SDE for 10 seconds in total and we discretize it into 100 steps. So each step is 0.1 seconds.
    total_time = 10
    timesteps = 100

    # define the model
    model = MLP(input_dim=2, hidden_dim=128, output_dim=2)
    model.to(device)

    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    optimizer.zero_grad()

    iterations = 25000
    batch_size = 128

    # set to train mode
    model.train()

    train_losses = []
    start_time = time.time()

    # the training methodology I'm using is not standard one.
    for step in range(iterations):
        # zero grad
        optimizer.zero_grad()

        # sample a batch of data
        indices = np.random.choice(len(training_samples), batch_size, replace=False)
        batch = training_samples[indices]
        # convert to torch tensor
        batch = torch.tensor(batch, dtype=torch.float32, device=device)

        # sample timesteps
        ts = np.random.randint(0, timesteps, size=batch_size)
        ts = torch.tensor(ts, dtype=torch.long, device=device)

        # if timestep is 0, it denotes the transition from 0 to total_time /timesteps
        # if timestep is 1, it denotes the transition from 0 to 2 * total_time / timesteps
        # if timestep is 2, it denotes the transition from 0 to 3 * total_time / timesteps
        # if timestep is (timesteps - 1), it denotes the transition from 0 to total_time.

        # using timesteps, get t
        t = (ts + 1) * (total_time / timesteps)

        # sample noise
        # randn_like puts the output in same device as input.
        noise = torch.randn_like(batch)

        # compute the noisy batch
        noisy_batch = (torch.exp(-t)[:, None]) * batch + (torch.sqrt(1 - torch.exp(-2 * t))[:, None]) * noise

        # predict the score
        # remember that we should give t.
        pred = model(noisy_batch, t)

        # compute the loss function
        # weights = 1 / (1 - torch.exp(-2 * t))
        weights = (1 - torch.exp(-2 * t))
        # weights = 1.0
        loss = pred + (noise / (torch.sqrt(1 - torch.exp(-2 * t))[:, None]))
        loss = torch.mean(loss ** 2, axis=1)
        loss = torch.mean(weights * loss)

        # backward
        loss.backward()
        optimizer.step()

        # calculate the normal mse, so that we can track.
        # its not in usual interest to track weighted mse. It will not give you
        # much signal - according to me.
        with torch.no_grad():
            track_loss = F.mse_loss(pred, -noise / (torch.sqrt(1 - torch.exp(-2 * t))[:, None]))
            train_losses.append(track_loss.item())

        if (step + 1) % 1000 == 0:
            end_time = time.time()
            timetaken = round(end_time - start_time, 2)
            print(f"step = {step + 1} / {iterations}, time taken = {timetaken} seconds, weighted loss = {round(loss.item(), 4)}, unweighted loss = {round(track_loss.item(), 4)}")

    # save the model
    torch.save(model.state_dict(), "gmm_score_estimation.bin")

    # save the plot
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses)
    plt.xlabel("iterations")
    plt.ylabel("mse loss")
    plt.savefig("loss_plot.png")
    plt.close()

    # we will plot the loss.
    print("Training Completed")
