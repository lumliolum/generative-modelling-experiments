import os
import math
import random

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_normalize_transform(p, channels):
    if channels == 1:
        mean, std = [0.5], [0.5]
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # get normalize transform, which takes pil image
    # and gives the result in [-1, 1] range
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.uint8, scale=True),
        transforms.RandomHorizontalFlip(p=p),
        transforms.ToDtype(torch.float32, scale=True), # in [0, 1] range
        transforms.Normalize(mean=mean, std=std)
    ])


class JustImageDataset(Dataset):
    def __init__(self, dataset, transform, indices):
        self.dataset = dataset
        self.transform = transform
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[self.indices[index]]
        image = self.transform(image)
        return image


def split_train_val_indices(dataset_length, val_fraction):
    indices = np.arange(dataset_length)
    np.random.shuffle(indices)
    val_size = int(dataset_length * val_fraction)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def train_one_step(model, x_batch, noise_var, optimizer, do_grad_clip, grad_clip, device, objective, ema):
    # do zero grad
    optimizer.zero_grad()

    # transfer to device
    x_batch = x_batch.to(device)
    # sample noise
    noise = torch.randn_like(x_batch)
    # get the noisy input
    x_batch_noisy = x_batch + math.sqrt(noise_var) * noise

    if objective == "score":
        # model predict the score
        score_batch_hat = model(x_batch_noisy)
        # compute the loss between predicted score and true score
        # true score will be "- noise / sqrt(noise_var)"
        loss = F.mse_loss(score_batch_hat, -noise / math.sqrt(noise_var))
    elif objective == "denoiser":
        # model predicts the denoisy version of the image
        x_batch_hat = model(x_batch_noisy)
        # compute the loss between predicted image and true image
        loss = F.mse_loss(x_batch_hat, x_batch)

    # loss backward
    loss.backward()

    # gradient clipping
    if do_grad_clip:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)

    # optimizer step
    optimizer.step()

    # ema update
    if ema is not None:
        ema.update()

    return loss.item()


def eval_one_batch(model, x_batch, noise_var, device, objective):
    # transfer to device
    x_batch = x_batch.to(device)
    # sample noise
    noise =  torch.randn_like(x_batch)
    # get the noisy input
    x_batch_noisy = x_batch + math.sqrt(noise_var) * noise

    if objective == "score":
        # model predict the score
        score_batch_hat = model(x_batch_noisy)
        # compute the loss between predicted score and true score
        # true score will be "- noise / sqrt(noise_var)"
        loss = F.mse_loss(score_batch_hat, -noise / math.sqrt(noise_var))
    elif objective == "denoiser":
        # model predicts the denoisy version of the image
        x_batch_hat = model(x_batch_noisy)
        # compute the loss between predicted image and true image
        loss = F.mse_loss(x_batch_hat, x_batch)

    return loss.item()


def get_in_range(image):
    """
        image will be of shape (1, channels, height, width) or (channels, height, width)
        and it will be in range -1 to 1
    """
    if len(image.shape) == 4:
        image = image.squeeze(0)

    im = torch.permute(image, (1, 2, 0))
    im = torch.squeeze(im)
    im = 255 * 0.5 * (im .cpu().numpy() + 1)
    im = im.astype(np.uint8)
    return im


def save_sampled_image(y, denoised, save_path, step, color=True):
    xn = get_in_range(y)
    x = get_in_range(denoised)

    path = os.path.join(save_path, f"sample_step_{step}.png")

    plt.subplot(1, 2, 1)
    plt.title(f"Step - {step} - Sample")
    if color:
        plt.imshow(xn)
    else:
        plt.imshow(xn, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title(f"Step {step} - Denoised")
    if color:
        plt.imshow(x)
    else:
        plt.imshow(x, cmap="gray")

    plt.savefig(path)
    plt.close()


def langevin_sampling(model, noise_var, langevin_steps, langevin_save_every, device, save_path, input_shape, objective, color):
    model.eval()
    # the epsilon that we will use for langevin update is O(sigma ^ 2)
    epsilon = noise_var

    # initialize with gaussian noise
    # shape will be (1, channels, height, width)
    y = torch.randn(input_shape, device=device, dtype=torch.float32).unsqueeze(0)

    # get the model prediction at current sample
    denoised_tensors = []
    with torch.no_grad():
        for step in range(langevin_steps + 1):
            if objective == "score":
                score = model(y)
                # tweedie forumale
                denoised = y + noise_var * score
            elif objective == "denoiser":
                denoised = model(y)
                # tweedie forumale
                score = (denoised - y) / noise_var

            if step % langevin_save_every == 0:
                denoised_tensors.append(denoised)
                # save the sampled image
                save_sampled_image(y, denoised, save_path, step, color)

            # langevin update
            y = y + epsilon * score + math.sqrt(2 * epsilon) * torch.randn_like(y)

    return denoised_tensors


def plot_grid(data_tensors, save_path, color=True):
    """
        data_tensors is a list of tensors of shape (1, channels, height, width)
        and length 100
    """
    grid_size = 10
    plt.figure(figsize=(grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            im = get_in_range(data_tensors[index])
            plt.subplot(grid_size, grid_size, index + 1)
            plt.axis('off')
            if color:
                plt.imshow(im)
            else:
                plt.imshow(im, cmap="gray")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path)
    plt.close()
