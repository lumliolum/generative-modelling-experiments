import os
import time

import torch
import numpy as np
from loguru import logger
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from arguments import get_parser
from models import ThreeLayerMLP
from neigbhorhoods import AdjacentNeighborhood
from utils import set_seed, softmax, plot_pmf_and_empirical_pmf, plot_training_loss, plot_generated_samples_distribution


def sample_neigh_or_inv_neighs(data, neigh_or_inv_list, neigh_cls, inverse=False):
    sampled_data = []
    for x, neigh_or_inv in zip(data, neigh_or_inv_list):
        # sidx -> sampled index
        # sd -> sampled data
        sidx = np.random.randint(len(neigh_or_inv))
        sd = neigh_or_inv[sidx]
        if inverse:
            # find out at what index x is present in neighbor list of sd
            fidx = -1
            for idx, ngh in enumerate(neigh_cls.neighborhood[sd]):
                if ngh == x:
                    fidx = idx
                    break

            sampled_data.append([sd, sidx, fidx])
        else:
            sampled_data.append([sd, sidx])

    return torch.tensor(sampled_data)


def training_loop(train_loader, model, optimizer, neigh_cls, num_categories, epochs, device):
    # set in train model
    model.train()

    train_losses = []
    for ep in range(epochs):
        logger.info(f"Epoch = {ep + 1} / {args.epochs}")

        t1 = time.time()

        train_steps = 0
        train_mean_loss = 0.0
        for x_batch in train_loader:
            x_batch = x_batch[0]

            # get the neighbors for each sample in the batch
            batch_neighs = [neigh_cls.neighborhood[x.item()] for x in x_batch]
            batch_inv_neighs = [neigh_cls.inv_neighborhood[x.item()] for x in x_batch]

            # get the number of neighbors and inverse neighbors
            batch_num_neighs = torch.tensor([len(x) for x in batch_neighs], device=device)
            batch_num_inv_neighs = torch.tensor([len(x) for x in batch_inv_neighs], device=device)

            # sample a neighbor and a inverse neighbor
            # just note that for each vertex, the number of inverse neighbors can be different.
            x_batch_sampled_neighs = sample_neigh_or_inv_neighs(x_batch.tolist(), batch_neighs, neigh_cls)
            x_batch_sampled_inv_neighs = sample_neigh_or_inv_neighs(x_batch.tolist(), batch_inv_neighs, neigh_cls, inverse=True)

            # model forward
            # normalize to 0 - 1
            x_batch = x_batch.float().unsqueeze(1).to(device) / num_categories
            nout = model(x_batch)
            # pass the sampled inverse neighbors
            x_inv_batch = x_batch_sampled_inv_neighs[:, 0]
            x_inv_batch = x_inv_batch.float().unsqueeze(1).to(device) / num_categories
            niout = model(x_inv_batch)

            # estimate first term in the loss
            # sample the neighbors according to indices above.
            sampled_nout = nout[torch.arange(len(nout)), x_batch_sampled_neighs[:, 1]]
            L1 = torch.mean(batch_num_neighs * (sampled_nout ** 2 + 2 * sampled_nout))

            # estimate second term in the loss
            # sample the inverse neighbor according to indices above.
            sampled_niout = niout[torch.arange(len(niout)), x_batch_sampled_inv_neighs[:, 2]]
            L2 = 2 * torch.mean(batch_num_inv_neighs * sampled_niout)

            # calculate the loss
            loss = L1 - L2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mean_loss += loss.item()
            train_steps += 1

        train_mean_loss = train_mean_loss / train_steps
        t2 = time.time()
        tr_time_taken = round(t2 - t1, 2)
        logger.info(f"Training Time Taken = {tr_time_taken} seconds, loss = {train_mean_loss}")

        train_losses.append(train_mean_loss)

    return train_losses


def one_sample_using_mh(categories, neigh_cls, num_categories, model, device, mh_iters=100):
    # set to eval mode
    model.eval()

    # sample a starting point from categories randomly
    p = np.random.choice(categories)
    for _ in range(mh_iters):
        # get the neighbors for the current point
        neighs = neigh_cls.neighborhood[p]

        # sample a neighbor from the list of neighbors uniformly
        pnidx = np.random.randint(len(neighs))
        pn = neighs[pnidx]

        # run model forward
        x = torch.tensor([[p]], device=device).float() / num_categories
        with torch.no_grad():
            out = model(x)

        # get the concrete score for the neighbor
        cs = out[0][pnidx].item()

        # conditional probability q(pn|p) that is neighbor given the current
        qpnp = 1 / len(neighs)
        if p in neigh_cls.neighborhood[pn]:
            qppn = 1 / len(neigh_cls.neighborhood[pn])
        else:
            # conditional probability q(p|pn) that is current given neighbor
            qppn = 0

        # calculate the acceptance probability
        # go to new state with this probability
        a = min(1, (1 + cs) * qppn / qpnp)
        if np.random.rand() < a:
            p = pn

    return p


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # use torch device
    args.device = torch.device(args.device)

    # create the output path
    os.makedirs(args.output_dir, exist_ok=True)

    # create log path
    log_file_path = os.path.join(args.output_dir, "run.log")
    logger.add(log_file_path)

    # log details
    logger.info(f"Using seed = {args.seed}, device = {args.device} and output dir = {args.output_dir}")

    # set seed
    set_seed(args.seed)

    # create the data
    categories = np.arange(args.num_categories)
    # pmf of 16 categories
    probs = softmax(np.random.uniform(0, 3, size=(args.num_categories, )))
    pmf = {categories[idx] : probs[idx] for idx in range(16)}

    # use the pmf, lets draw 1000 samples. This will be our training dataset as well.
    n_samples = 5000
    samples = np.random.choice(categories, size=n_samples, replace=True, p=probs)

    # save the samples and pmf
    plot_pmf_and_empirical_pmf(
        categories,
        probs,
        samples,
        os.path.join(args.output_dir, "pmf_and_empirical_pmf.png")
    )

    # initialize the neighborhood and model
    adj_neigh_cls = AdjacentNeighborhood(categories.tolist())

    # initialize the model
    model = ThreeLayerMLP(in_features=1, out_features=adj_neigh_cls.num_neighbors, hidden_dim=args.hidden_dim)
    model.to(args.device)

    # create the dataloader
    dataset = TensorDataset(torch.tensor(samples)) 
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # call the training
    train_losses = training_loop(loader, model, optimizer, adj_neigh_cls, args.num_categories, args.epochs, args.device)

    # plot and save training loss
    plot_training_loss(train_losses, os.path.join(args.output_dir, "training_loss.png"))

    # let's write the inference which is MH algorithm
    logger.info(f"Running inference to generate {args.num_samples_to_generate} samples")
    generated_samples = []
    for gix in range(args.num_samples_to_generate):
        sample = one_sample_using_mh(categories, adj_neigh_cls, args.num_categories, model, args.device, args.mh_iters)
        generated_samples.append(sample)
        if (gix + 1) % 10 == 0:
            logger.info(f"{gix + 1} samples generated")

    # plot and save generated samples distribution
    plot_generated_samples_distribution(generated_samples, os.path.join(args.output_dir, "generated_samples.png"))
