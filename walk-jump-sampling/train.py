import os
import time
import json

import torch
import torchvision
from loguru import logger
import torch.optim as optim
from torch.utils.data import DataLoader

from arguments import get_parser
from ema import EMAHelper
from models import UNetSingleNoiseScale
from utils import (
    set_seed,
    get_normalize_transform,
    JustImageDataset,
    train_one_step,
    eval_one_batch,
    split_train_val_indices,
    langevin_sampling,
    plot_grid
)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # checks
    if args.training_objective not in ["denoiser", "score"]:
        message = f"Unsupported training objective = {args.training_objective}. Supported objectives are 'denoiser' and 'score'"
        logger.error(message)
        raise ValueError(message)

    if args.dataset_name not in ["mnist", "cifar10"]:
        message = f"Unsupported dataset = {args.dataset_name}. Supported datasets are 'mnist' and 'cifar10'"
        logger.error(message)
        raise ValueError(message)

    # global variables
    if args.dataset_name == "mnist":
        args.image_shape = (28, 28)
        args.channels = 1
    elif args.dataset_name == "cifar10":
        args.image_shape = (32, 32)
        args.channels = 3

    args.flip_probs = 0.0
    color = args.channels == 3

    # create the output path
    os.makedirs(args.output_dir, exist_ok=True)

    # create log path
    log_file_path = os.path.join(args.output_dir, "run.log")
    logger.add(log_file_path)

    # log details
    logger.info(f"Using seed = {args.seed}, device = {args.device} and output dir = {args.output_dir}")

    # set seed
    set_seed(args.seed)

    # get dataset
    logger.info(f"Loading the {args.dataset_name} dataset for training")
    if args.dataset_name == "mnist":
        train_ds = torchvision.datasets.MNIST(root="../data/", train=True, download=True)
    elif args.dataset_name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(root="../data/", train=True, download=True)

    logger.info(f"length of the train dataset = {len(train_ds)}")

    # split train into train and val
    train_indices, val_indices = split_train_val_indices(len(train_ds), args.val_fraction)
    logger.info(f"length of the train split = {len(train_indices)}, validation split = {len(val_indices)}")

    # get transforms
    logger.info(f"Using flip_probs = {args.flip_probs}, channels = {args.channels} for transforms")
    train_transforms = get_normalize_transform(p=args.flip_probs, channels=args.channels)
    val_transforms = get_normalize_transform(p=0.0, channels=args.channels)

    # create custom dataset using mnist dataset and the transforms
    train_dataset = JustImageDataset(train_ds, train_transforms, train_indices)
    val_dataset = JustImageDataset(train_ds, val_transforms, val_indices)

    # create dataloader
    logger.info(f"Creating the dataloader with batch size = {args.batch_size}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # initialize the model
    # irritating task to be honest.
    logger.info(f"Initializing the UNET model with model dim = {args.model_dim}")
    model = UNetSingleNoiseScale(
        channels=args.channels,
        dim=args.model_dim,
        norm_groups=args.norm_groups,
        dropout_prob=args.dropout_prob
    )
    model.to(args.device)

    if args.do_ema:
        logger.info(f"Initializaing EMA with decay = {args.ema_decay}")
        ema = EMAHelper(args.ema_decay, model)
    else:
        ema = None

    # initialize the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize the model save path
    args.model_save_path = os.path.join(args.output_dir, "single-noise-scale.bin")

    # best val loss
    best_val_loss = float("inf")

    # training loop
    logger.info(f"Running for epochs = {args.epochs} with objective = {args.training_objective}")
    train_losses, val_losses = [], []
    for ep in range(args.epochs):
        logger.info(f"Epoch = {ep + 1} / {args.epochs}")

        t1 = time.time()

        # run training
        model.train()
        train_mean_loss = 0.0
        train_steps = 0
        for x_batch in train_loader:
            loss = train_one_step(model, x_batch, args.noise_var, optimizer, args.do_grad_clip, args.grad_clip, args.device, args.training_objective, ema)
            train_mean_loss += loss
            train_steps += 1

        # apply the ema model, so that we can run the evaluation on
        # ema weights.
        if ema is not None:
            ema.apply()

        # run evaluation
        model.eval()
        val_mean_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for x_batch in val_loader:
                loss = eval_one_batch(model, x_batch, args.noise_var, args.device, args.training_objective)
                val_mean_loss += loss
                val_steps += 1

        train_mean_loss = train_mean_loss / train_steps
        val_mean_loss = val_mean_loss / val_steps

        train_losses.append(train_mean_loss)
        val_losses.append(val_mean_loss)

        t2 = time.time()
        tr_time_taken = round(t2 - t1, 2)
        logger.info(f"Training Time Taken = {tr_time_taken} seconds, loss = {train_mean_loss}, val loss = {val_mean_loss}")

        # save the best model
        if val_mean_loss < best_val_loss:
            logger.info(f"Validation loss decreased from {best_val_loss} to {val_mean_loss}")
            logger.info(f"Saving the model checkpoint at path = {args.model_save_path}")
            # note that here the model that will be saved is the EMA version, if ema flag
            # is present
            torch.save(model.state_dict(), args.model_save_path)

            # update the best val loss
            best_val_loss = val_mean_loss

        # restore the model, to continue the training
        if ema is not None:
            ema.restore()

    # load the saved model
    logger.info(f"Training Completed. Loading the model from {args.model_save_path}")
    model = UNetSingleNoiseScale(
        channels=args.channels,
        dim=args.model_dim,
        norm_groups=args.norm_groups,
        dropout_prob=args.dropout_prob
    )
    model.load_state_dict(torch.load(args.model_save_path, map_location="cpu"))
    model.to(args.device)

    # now do the langevin sampling using the trained model
    logger.info(f"Running langevin sampling for {args.langevin_steps} steps, saving every {args.langevin_save_every} steps")
    # create output directory for saving images
    image_save_path = os.path.join(args.output_dir, "samples")
    os.makedirs(image_save_path, exist_ok=True)
    # run sampling
    input_shape = (args.channels, ) + args.image_shape
    denoised_tensors = langevin_sampling(model, args.noise_var, args.langevin_steps, args.langevin_save_every, args.device, image_save_path, input_shape, args.training_objective, color=color)

    # plot the sampled grid
    if len(denoised_tensors) == 101:
        grid_save_path = os.path.join(args.output_dir, "sampled_grid.png")
        logger.info(f"Saving the sampled grid at path = {grid_save_path}")
        plot_grid(denoised_tensors[1:], grid_save_path, color)

    # plot the grid of original dataset
    sample_loader = DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=True
    )
    sample_batch = next(iter(sample_loader))
    original_grid_save_path = os.path.join(args.output_dir, "original_grid.png")
    logger.info(f"Saving the original grid at path = {original_grid_save_path}")
    plot_grid(sample_batch, original_grid_save_path, color)

    # store the configuration json
    config_json_path = os.path.join(args.output_dir, "run_config.json")
    logger.info(f"Saving the json at path = {config_json_path}")
    with open(config_json_path, "w") as f:
        json.dump(vars(args), f, indent=4)
