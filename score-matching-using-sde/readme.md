# DETAILS
In this, I'm trying to do score matching using SDE framework that is basically training in continuous time.

## OU SDE 
We will consider the diffusion given by OU process which is given by following SDE

$$
    dx = -\Sigma^{-1}(X - \mu)dt + \sqrt{2}dw
$$

and fortunately, this SDE has conditional in closed form that is

$$
    x_{t} \mid x_{0} \sim \mathcal{N}\left(\mu + e^{-t\Sigma^{-1}}(x_{0} - \mu), \Sigma - \Sigma^{-1}e^{-2t\Sigma^{-1}}\right)
$$

where the exponential is matrix exponential. So as $t$ goes to $\infty$, $x_{t}$ will converge to $\mathcal{N}(\mu, \Sigma)$.

Here, we will select a specific SDE that is choosing $\mu = 0$ and $\Sigma = I$, then the SDE is

$$
    dx = -xdt + \sqrt{2}dw
$$

and the corresponding marginal is

$$
    x_{t} \mid x_{0} \sim \mathcal{N}\left(e^{-t}x_{0}, (1 - e^{-2t})I\right)
$$


