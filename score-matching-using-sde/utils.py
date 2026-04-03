import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal


class GaussianMixture:
    def __init__(self, alpha, mus, sigmas):
        """
            alpha : array of mixture weights. shape (n_components,)
            mus : array of means. shape (n_components, d)
            sigmas : array of standard deviations. shape (n_components, d, d)
            where d is dimension of the data.
        """
        self.alpha = alpha
        self.mus = mus
        self.sigmas = sigmas

        # calculate the square root of the covaraince matrices
        self.sqrt_sigmas = sqrtm(self.sigmas)
        self.sigmas_inv = np.linalg.inv(self.sigmas)

        self.n_components = len(alpha)
        self.d = mus.shape[1]

    def sample(self, n_samples):
        mixture_indices = np.random.choice(len(self.alpha), size=n_samples, p=self.alpha)
        samples = np.random.normal(size=(n_samples, self.d, 1))
        samples = self.sqrt_sigmas[mixture_indices] @ samples
        samples = self.mus[mixture_indices] + np.squeeze(samples, axis=-1)
        return samples

    def pdf(self, x):
        """
            x : array of shape (n_samples, d)
        """
        pdf = 0
        for idx in range(self.n_components):
            pdf += self.alpha[idx] * multivariate_normal.pdf(x, mean=self.mus[idx], cov=self.sigmas[idx])

        return pdf

    def score(self, x):
        """
            x : array of shape (n_samples, d)
        """
        # in this function, we will calculate the score, which is the gradient of the log pdf with respect to x.
        num = 0
        den = 0
        for index in range(self.n_components):
            # (n_samples,)
            comp_pdf = self.alpha[index] * multivariate_normal.pdf(x, mean=self.mus[index], cov=self.sigmas[index])

            if len(comp_pdf.shape) == 0:
                # this will happen, if n_samples is 1, then comp_pdf will be a scalar. We need to make it an array of shape (1,)
                comp_pdf = np.array([comp_pdf])

            # (1, d, d) @ (n_samples, d, 1) -> (n_samples, d, 1)
            add = -np.expand_dims(self.sigmas_inv[index], 0) @ (x[:, :, None] - self.mus[index][None, :, None])
            # (n_samples, 1) * (n_samples, d, 1) -> (n_samples, d)
            num += comp_pdf[:, None] * np.squeeze(add, axis=-1)

            den += comp_pdf

        # (n_samples, d) / (n_samples, 1) -> (n_samples, d)
        return num / den[:, None]


def mix_gauss_params_at_t(initial_alphas, initial_means, initial_covs, t):
    """
        Given the initial parameters of a Gaussian mixture model, calculate the parameters at time t.
        initial_alphas : array of shape (n_components,)
        initial_means : array of shape (n_components, d)
        initial_covs : array of shape (n_components, d, d)
        t : time - float - greater than or equal to 0
    """
    # the weights don't change
    alphas = initial_alphas.copy()

    # means
    means = np.exp(-t) * initial_means

    # covs
    covs = np.exp(-2 * t) * initial_covs + (1 - np.exp(-2 * t)) * np.eye(initial_covs.shape[1])

    return alphas, means, covs


def gmm_with_pdf_and_scores(initial_alphas, initial_means, initial_covs, t):
    params = mix_gauss_params_at_t(initial_alphas, initial_means, initial_covs, t)
    gmm = GaussianMixture(*params)

    # calculate pdf values for contour plot
    grid_x = np.linspace(-5, 5, 50)
    grid_y = np.linspace(-5, 5, 50)

    # create a meshgrid
    X, Y = np.meshgrid(grid_x, grid_y)

    x = np.stack([X.ravel(), Y.ravel()], axis=-1)
    # get the pdf
    pdf = gmm.pdf(x)

    Z = pdf.reshape(X.shape)

    # get the scores
    scores = gmm.score(x)

    # take the first component of the score
    U = scores[:, 0].reshape(X.shape)
    V = scores[:, 1].reshape(X.shape)

    return X, Y, Z, U, V, x


def gmm_score_at_t(initial_alphas, initial_means, initial_covs, t, x):
    params = mix_gauss_params_at_t(initial_alphas, initial_means, initial_covs, t)
    gmm = GaussianMixture(*params)

    return gmm.score(x)
