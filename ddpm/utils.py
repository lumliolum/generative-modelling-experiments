import os
import random

import torch
import torch.nn as nn
import numpy as np
from loguru import logger
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_betas(beta_start, beta_end, timesteps, device):
    # as per the original paper, they have used linear schedule
    # for betas
    return torch.linspace(beta_start, beta_end, steps=timesteps, device=device)


def compute_alphas(betas):
    return 1 - betas


def compute_alphabars(alphas):
    return torch.cumprod(alphas, dim=0)


def get_normalize_transform(p=0):
    # get normalize transform, which takes pil image
    # and gives the result in [-1, 1] range
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.uint8, scale=True),
        transforms.RandomHorizontalFlip(p=p),
        transforms.ToDtype(torch.float32, scale=True), # in [0, 1] range
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_renormalize_transform():
    # get re normalize transform, which takes [-1, 1] tensor
    # and returns PIL Image
    return transforms.Compose([
        transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
        transforms.ToDtype(torch.uint8, scale=True),
        transforms.ToPILImage()
    ])


def extract_at_t(quantity, ts):
    """
        quantity : torch.tensor of size timesteps (T). It can be alphas, betas, or alphabars
        ts : torch.tensor of size (B). We have extract quantity at these timesteps
    """
    # repeat the quantity batch size times
    B = ts.shape[0]
    # repeat_quantity -> (B, T)
    repeat_quantity = torch.tile(quantity, dims=(B, 1))
    row_indices = torch.arange(B)

    # now slice at the required indices
    # sliced_quantity -> (B)
    sliced_quantity = repeat_quantity[row_indices, ts]
    # unsqueeze B -> (B, 1, 1, 1)
    sliced_quantity = sliced_quantity.reshape((B, 1, 1, 1))
    return sliced_quantity


def forward_sample(x_batch, t_batch, alphabars, noise=None):
    """
        x_batch -> (B, C, H, H)
        t_batch -> (B)
    """
    if noise is None:
        noise = torch.randn_like(x_batch)

    sqrt_alphabar_ts = extract_at_t(torch.sqrt(alphabars), t_batch)
    sqrt_one_minus_alphabar_ts = extract_at_t(torch.sqrt(1 - alphabars), t_batch)
    return sqrt_alphabar_ts * x_batch + sqrt_one_minus_alphabar_ts * noise


def show_forward_for_one_image(pil_image, alphabars, device, output_dir):
    normalize_transform = get_normalize_transform()
    renormalize_transform = get_renormalize_transform()

    x = torch.unsqueeze(normalize_transform(pil_image), 0).to(device)
    ts = torch.tensor([0, 19, 79, 299, 799, 999])

    noisy_samples = forward_sample(x, ts, alphabars)

    fig, ax = plt.subplots(1, len(ts), figsize=(12, 3))
    for x in range(len(ts)):
        ax[x].imshow(renormalize_transform(noisy_samples[x]))
        ax[x].set_title(f"t = {ts[x].item() + 1}")

    plt.savefig(os.path.join(output_dir, "forward-process.png"))
    plt.close()


def variational_bound(noise, noise_hat, t_batch, **kwargs):
    alphas = kwargs["alphas"]
    alphabars = kwargs["alphabars"]
    variance = kwargs["variance"]

    # extract
    alpha_ts = extract_at_t(alphas, t_batch)
    alphabar_ts = extract_at_t(alphabars, t_batch)
    variance_ts = extract_at_t(variance, t_batch)

    # squared difference
    loss = F.mse_loss(noise, noise_hat, reduction='none')
    loss = torch.sum(loss, dim=(1, 2, 3))

    loss = 0.5 * torch.pow(1 - alpha_ts, 2) * loss
    loss = loss / variance_ts
    loss = loss / (alphabar_ts * (1 - alphabar_ts))
    return torch.mean(loss)


def mse(noise, noise_hat, t_batch, **kwargs):
    loss = F.mse_loss(noise, noise_hat, reduction='none')
    loss = torch.mean(loss, dim=(1, 2, 3))
    loss = torch.mean(loss)
    return loss


def train_one_step(model, x_batch, timesteps, alphabars, optimizer, grad_clip, device):
    # do zero grad
    optimizer.zero_grad()

    B = x_batch.shape[0]

    # sample t
    t_batch = torch.randint(0, timesteps, (B, ))

    # device
    x_batch = x_batch.to(device)
    t_batch = t_batch.long().to(device)

    # sample random noise
    noise = torch.randn_like(x_batch)

    # sample forward
    x_noisy_batch = forward_sample(x_batch, t_batch, alphabars, noise)

    # pass the noisy batch to model to get the predicted noise
    noise_hat = model(x_noisy_batch, t_batch)

    # calculate the loss
    loss = mse(noise, noise_hat, t_batch)

    # backward
    loss.backward()

    # clip gradient norms
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)

    # optimizer
    optimizer.step()

    return loss.item()


def sample_at_single_step(model, x_batch, t_batch, curr_step, alphas, alphabars, variance):
    """
        This function goes from step t to t-1
        x -> (B, C, H, H)
        t -> (B, )
        curr_step -> integer between 0 to TIMESTEPS-1 denoting the current step
    """
    # get the model prediction
    noise_hat = model(x_batch, t_batch)
    sqrt_alpha_ts = extract_at_t(torch.sqrt(alphas), t_batch)
    one_minus_alpha_ts = extract_at_t(1 - alphas, t_batch)
    sqrt_one_minus_alphabar_ts = extract_at_t(torch.sqrt(1 - alphabars), t_batch)

    # calculate the mean of p(x_{t-1} | x_{t})
    mean = x_batch - ((one_minus_alpha_ts) / (sqrt_one_minus_alphabar_ts)) * noise_hat
    mean = mean / sqrt_alpha_ts

    if curr_step == 0:
        return mean
    else:
        variance_ts = extract_at_t(variance, t_batch)
        noise = torch.randn_like(x_batch)
        return mean + torch.sqrt(variance_ts) * noise


def sample_batch(model, batch_size, timesteps, shape, device, alphas, alphabars, variance):
    # get input shape
    input_shape = tuple([batch_size] + list(shape))
    # start from pure noise
    images = torch.randn(input_shape, device=device)

    with torch.no_grad():
        for timeindex in reversed(range(timesteps)):
            # for this time index, create a batch of times with this index
            t_batch = torch.tensor([timeindex] * batch_size, dtype=torch.long, device=device)
            # get the sample. update the images
            images = sample_at_single_step(model, images, t_batch, timeindex, alphas, alphabars, variance)

    # images will be in range of [-1, 1] of shape (B, C, H, W)
    images = torch.clamp(images, -1, 1)
    # convert them into 0 to 1 range.
    images = 0.5 * (images + 1)
    return images


def generate_and_save(model, timesteps, device, alphas, alphabars, variance, save_path):
    images = sample_batch(
        model=model,
        batch_size=100,
        timesteps=timesteps,
        shape=(3, 32, 32),
        device=device,
        alphas=alphas,
        alphabars=alphabars,
        variance=variance
    )

    # push to cpu
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # plot
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')

    plt.savefig(save_path)
    plt.close()
