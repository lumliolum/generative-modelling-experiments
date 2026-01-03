# Diffusion and Score Matching Experiments

This repository contains the code for various experiments conducted while I was studying papers related to diffusion and score matching.

All implementations are in PyTorch.

## Implemented Methods

### Concrete Score Matching

Introduced concrete score and provided a way to perform score matching in discrete spaces. The paper can be found [here](https://arxiv.org/abs/2211.00802).

### DDPM

The classic Denoising Diffusion Probabilistic Models implementation.

### Walk Jump Sampling

An interesting approach to score matching introduced by Saremi and Hyvarinen in their JMLR paper, which can be found [here](https://jmlr.org/papers/volume20/19-216/19-216.pdf). With a single noise level, this implementation successfully generates quality MNIST samples.

### Minimum Probability Flow (MPF) Learning

A parameter estimation technique introduced by Jascha Sohl-Dickstein et al. for learning energy-based models. This directory contains experimental work on this method, though the core loss function from the paper was not successfully implemented.

## Additional Resources

The `notebooks/` folder contains various small-scale experiments conducted to understand key concepts.
