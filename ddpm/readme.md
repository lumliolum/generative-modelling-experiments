In this repository, I'm trying to implement the classic DDPM and run for CIFAR-10. In this document, if needed I will outline some decision I took while implementing.

Our ELBO is

$$
    L = E_{q(x_{1}|x_{0})}[\log p_{\theta}(x_{0}|x_{1})] - KL(q(x_{T}|x_{0}) || p(x_{T})) -\sum_{t=2}^{T} E_{q(x_{t}|x_{0})}\left[KL\left(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t})\right)\right]
$$

We ignore the term $KL(q(x_{T}|x_{0}) || p(x_{T}))$ and for $2 \leq t \leq T$ we know that $q(x_{t-1}|x_{t},x_{0}) = \mathcal{N}(x_{t-1}; \mu_{q}(x_{t},x_{0}), \sigma^{2}_{t}I)$ where

$$
    \mu_{q}(x_{t},x_{0}) = \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_{t}}}{1 - \bar{\alpha}_{t}}x_{t} + \frac{(1 - \alpha)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t}}x_{0} \quad , \quad \sigma^{2}_{t} = \frac{(1 - \alpha_{t})(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}
$$

Using the fact that $x_{t} = \sqrt{\bar{\alpha}_{t}}x_{0} + \sqrt{1 - \bar{\alpha}_{t}}\epsilon$, we can write $\mu_{q}(x_{t},x_{0})$ as $\mu_{q}(x_{t}, \epsilon)$ given by

$$
    \mu_{q}(x_{t}, \epsilon) = \frac{1}{\sqrt{\alpha_{t}}}\left(x_{t} - \frac{1 - \alpha_{t}}{\sqrt{1 - \bar{\alpha}_{t}}}\epsilon\right)
$$

Assuming for $2 \leq t \leq T$, $p_{\theta}(x_{t-1}|x_{t}) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t}, t), \sigma^{2}_{t}I)$ where

$$
    \mu_{\theta}(x_{t}, t) = \frac{1}{\sqrt{\alpha_{t}}}\left(x_{t} - \frac{1 - \alpha_{t}}{\sqrt{1 - \bar{\alpha}_{t}}}\epsilon_{\theta}(x_{t})\right)
$$

where $\epsilon_{\theta}$ is the network which when given $x_{t}$ as input predicts $\epsilon$. In this case, the KL divergence for $2 \leq t \leq T$, will be

$$
    KL\left(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t})\right) = \frac{1}{2\sigma_{t}^{2}}\frac{(1 - \alpha_{t})^{2}}{\alpha_{t}(1 - \bar{\alpha}_{t})}\Vert \epsilon_{\theta}(x_{t}) - \epsilon \Vert^{2}
$$

Instead we use the loss function

$$
    L_{simple} = -\sum_{t=1}^{T}E_{q(x_{t}|x_{0})}\left[\Vert\hat{\epsilon}_{\theta}(x_{t}, t) - \epsilon\Vert^{2}\right]
$$

For CIFAR, while sampling following authors [methodlogy](https://github.com/hojonathanho/diffusion/blob/master/scripts/run_cifar.py#L136), we use $\sigma^{2}_{t} = 1 - \alpha_{t}$ and not the one mentioned above. This can be controlled using the argument `variance_type`. If it is equal to `fixed_small`, then we choose same variance as posterior $q(x_{t-1}|x_{t},x_{0})$ or we choose $\beta_{t}$.

With the following hyper-parameters I got a FID of 11.29

```bash
python3 train_cifar10.py --timesteps 1000 --beta-start 0.0001 --beta-end 0.02 --variance-type fixed_small --seed 42 --device cuda:3 --output-dir runs/cifar10/v10/ --lr 0.00008 --epochs 2000 --eval-epochs 2000 --batch-size 128 --eval-batch-size 512 --do-ema --ema-decay 0.999 --do-grad-clip --grad-clip 1.0
```
