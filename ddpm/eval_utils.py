import torch
import numpy as np
from scipy import linalg
from loguru import logger
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader

from fid.inception import InceptionV3


class FIDDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pil_img, _ = self.dataset[index]
        return self.transform(pil_img)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class Evaluator:
    def __init__(self, cifar_dataset, num_samples_to_gen, batch_size, sample_fn, device):
        self.cifar_dataset = cifar_dataset
        self.num_samples_to_gen = num_samples_to_gen
        self.batch_size = batch_size
        self.sample_fn = sample_fn
        self.device = device

        # initialize the model. By default we will use
        # pool3 which will give us 2048 dim vector
        self.model = InceptionV3(
            resize_input=True,
            normalize_input=True,
            use_fid_inception=True,
        )
        self.model.to(self.device)
        self.feature_dim = 2048

        # we will create dataloader for real dataset
        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.ToDtype(torch.float32, scale=True), # in [0, 1] range
        ])
        self.real_dataset = FIDDataset(self.cifar_dataset, self.transform)
        self.real_loader = DataLoader(
            self.real_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # store the activations on cifar10 dataset after
        # computing it for first time
        self.real_mu = None
        self.real_sigma = None

    def get_real_inception_features(self):
        features = np.zeros((0, self.feature_dim))
        with torch.no_grad():
            for x_batch in self.real_loader:
                x_batch = x_batch.to(self.device)
                # pred -> (B, 2048, 1, 1)
                pred = self.model(x_batch)[0]
                pred = pred.squeeze().cpu().numpy()
                features = np.vstack((features, pred))

        return features

    def get_fake_inception_features(self, model):
        features = np.zeros((0, self.feature_dim))
        num_batches_to_sample = int(np.ceil(self.num_samples_to_gen / self.batch_size))

        with torch.no_grad():
            for _ in range(num_batches_to_sample):
                # use sample function to generate
                x_batch = self.sample_fn(model=model, batch_size=self.batch_size, device=self.device)
                pred = self.model(x_batch)[0]
                pred = pred.squeeze().cpu().numpy()
                features = np.vstack((features, pred))

        return features

    def compute_activation_statistics(self, features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def eval(self, model):
        # check if real inception features are calculated
        if self.real_mu is None:
            logger.info("Computing activations on real dataset")
            # caluculate the features on real dataset
            real_inception_features = self.get_real_inception_features()
            logger.info(f"real activations shape = {real_inception_features.shape}. Computing statistics on real dataset")
            self.real_mu, self.real_sigma = self.compute_activation_statistics(real_inception_features)

        # now generate images and get fake features
        logger.info(f"Generating samples and computing activations")
        fake_inception_features = self.get_fake_inception_features(model)
        logger.info(f"fake activations shape = {fake_inception_features.shape}. Computing statistics on generated sample activations")
        fake_mu, fake_sigma = self.compute_activation_statistics(fake_inception_features)

        logger.info(f"real activation mean = {self.real_mu.shape}, cov = {self.real_sigma.shape}")
        logger.info(f"fake activation mean = {fake_mu.shape}, cov = {fake_sigma.shape}")

        # calculate fid
        fid = calculate_frechet_distance(self.real_mu, self.real_sigma, fake_mu, fake_sigma)
        return fid
