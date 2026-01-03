import os
import json
import time
from functools import partial

import torch
import torchvision
import numpy as np
from loguru import logger
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from models import UNet
from arguments import get_parser
from eval_utils import Evaluator
from utils import (
    set_seed,
    get_betas,
    compute_alphas,
    compute_alphabars,
    train_one_step,
    get_normalize_transform,
    show_forward_for_one_image,
    sample_batch,
    generate_and_save
)


class DiffussionDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pil_img, _ = self.dataset[index]
        return self.transform(pil_img)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # create the output path
    os.makedirs(args.output_dir, exist_ok=True)

    # create log path
    log_file_path = os.path.join(args.output_dir, "run.log")
    logger.add(log_file_path)

    # log details
    logger.info(f"Using seed = {args.seed}, device = {args.device} and output dir = {args.output_dir}")

    # store the configuration json
    config_json_path = os.path.join(args.output_dir, "run_config.json")
    logger.info(f"Saving the json at path = {config_json_path}")
    with open(config_json_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    # set seed
    set_seed(args.seed)

    # dataset
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root="../data", train=True, download=True)
    logger.info(f"length of the dataset = {len(cifar10_train_dataset)}")

    # get betas
    logger.info(f"Computing betas, alphas with timesteps = {args.timesteps}")
    betas = get_betas(args.beta_start, args.beta_end, args.timesteps, args.device)
    alphas = compute_alphas(betas)
    alphabars = compute_alphabars(alphas)

    # alphabars(t-1)
    prev_alphabars = torch.cat((torch.tensor([0], device=args.device), alphabars[:-1]))

    # variance of q(x_{t-1}| x_{t}, x_{0})
    # note that beta is 1 - alpha
    variance = (1.0 - alphas) * (1.0 - prev_alphabars) / (1.0 - alphabars)

    # show forward
    show_forward_for_one_image(cifar10_train_dataset[0][0], alphabars, args.device, args.output_dir)

    # get the training transform
    # random horizontal flip with probability 0.5
    train_transform = get_normalize_transform(p=0.5)

    train_dataset = DiffussionDataset(cifar10_train_dataset, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # initialize the model
    # authors mentioned that they used 0.1 dropout for cifar10
    logger.info(f"Initializing the model")
    model = UNet(
        channels=3,
        dim=128,
        dim_multiplier=(1, 2, 2, 4),
        num_blocks_per_resolution=2,
        res_groups=32,
        num_heads=1,
        dropout_prob=0.1
    )
    model.to(args.device)

    # initialize the optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    # initialize the evaluator
    sample_fn = partial(
        sample_batch,
        timesteps=args.timesteps,
        shape=(3, 32, 32),
        alphas=alphas,
        alphabars=alphabars,
        variance=variance
    )
    evaluator = Evaluator(
        cifar_dataset=cifar10_train_dataset,
        num_samples_to_gen=50000,
        batch_size=args.eval_batch_size,
        sample_fn=sample_fn,
        device=args.device
    )

    logger.info(f"Running for epochs = {args.epochs}")
    for ep in range(args.epochs):
        logger.info(f"Epoch = {ep + 1} / {args.epochs}")

        t1 = time.time()
        train_mean_loss = 0.0
        train_steps = 0
        for x_batch in train_loader:
            loss = train_one_step(model, x_batch, args.timesteps, alphabars, optimizer, args.grad_clip, args.device)
            train_mean_loss += loss
            train_steps += 1
            # print(train_steps, flush=True)

        train_mean_loss = round(train_mean_loss / train_steps, 5)
        t2 = time.time()
        training_timetaken = round(t2 - t1, 2)

        if (ep + 1) % args.eval_epochs == 0:
            logger.info("Running evaluation")
            # here we have to run the evaluation
            # as per the original implementation, each evaluation means we have to calculate
            # fid score over the whole training set.
            t1 = time.time()
            fid = evaluator.eval(model)
            t2 = time.time()
            evaluation_timetaken = round(t2 - t1, 2)

            # at the end generate samples.
            t1 = time.time()
            logger.info(f"Generating 100 samples")
            generate_and_save(
                model,
                args.timesteps,
                args.device,
                alphas,
                alphabars,
                variance,
                os.path.join(args.output_dir, f"generated-samples-epoch-{ep + 1}.png")
            )
            t2 = time.time()
            logger.info(f"Generation completed in {round(t2 - t1, 2)} seconds")
        else:
            fid = None
            evaluation_timetaken = np.nan

        logger.info(f"Training time taken = {training_timetaken} seconds, Evaluation time taken = {evaluation_timetaken} seconds, loss = {train_mean_loss}, fid = {fid}")

        # # check the gradients of the model
        # grads = []
        # with torch.no_grad():
        #     for p in model.parameters():
        #         grads.append(torch.linalg.norm(p.grad).item())

        # # check the gradients
        # grads = np.array(grads)
        # logger.info(f"min = {np.min(grads)}, median = {np.percentile(grads, 50)}, max = {np.max(grads)}")
