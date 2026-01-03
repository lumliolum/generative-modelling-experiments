import random

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def plot_pmf_and_empirical_pmf(categories, probs, samples, save_path):
    # plot
    _, axes = plt.subplots(1, 2, figsize=(14, 4))

    # --- PMF plot ---
    sns.barplot(x=categories, y=probs, ax=axes[0])
    axes[0].set_xlabel("Categories")
    axes[0].set_ylabel("Probabilities")
    axes[0].set_title("PMF of the discrete random variable")
    axes[0].tick_params(axis='x', rotation=90)

    # --- Empirical probability plot ---
    sns.countplot(x=sorted(samples), stat="probability", ax=axes[1])
    axes[0].set_xlabel("Categories")
    axes[1].set_title("Empirical Probability (MLE)")
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_loss(train_losses, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Epochs")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_generated_samples_distribution(generated_samples, save_path):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=sorted(generated_samples), stat="probability")
    plt.xlabel("Categories")
    plt.ylabel("Probability")
    plt.title("Generated Samples Empirical Probability Distribution")
    plt.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
