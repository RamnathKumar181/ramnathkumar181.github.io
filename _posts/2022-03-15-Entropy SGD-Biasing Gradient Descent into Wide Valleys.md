---
layout: post
title: Entropy SGD-Biasing Gradient Descent into Wide Valleys
published: true
---

An overview of the paper “[Entropy SGD-Biasing Gradient Descent into Wide Valleys](https://arxiv.org/pdf/1810.12081.pdf)”.
<!--break-->
This paper proposes a new optimization algorithm called Entropy-SGD for training deep neural networks that is motivated by the local geometry of the energy landscape. The authors leverage upon this observation to construct a local-entropy-based objective function that favors well-generalizable solutions lying in large flat regions
of the energy landscape, while avoiding poorly-generalizable solutions located in the sharp valleys. All images and tables in this post are from their paper.

## Introduction

Local minima that generalize well and are discovered by gradient descent lie in “wide valleys” of the energy landscape, rather than in sharp, isolated minima. Almost-flat regions of the energy landscape are robust to data perturbations, noise in the activations, as well as perturbations of the parameters, all of which are widely-used techniques to achieve good generalization. Before we get into the exact proposed methodology, it would be useful to understand "Stochastic Gradient Langevin dynamics" and "Gibbs distribution" concretely.

### Stochastic Gradient Langevin dynamics

The stochastic gradient Langevin dynamics (SGLD) has close connections to the Brownian motion. Suppose we have a brownian motion sampled <img src="https://latex.codecogs.com/svg.latex?\bigl(\begin{smallmatrix}B_1,B_2,...,B_n\end{smallmatrix}\bigr)" title="\bigl(\begin{smallmatrix}B_1,B_2,...,B_n\end{smallmatrix}\bigr)" /> such that <img src="https://latex.codecogs.com/svg.latex?B_{t_1}-B_{t_2} \sim \mathcal{N}(0,t_1-t_2)" title="B_{t_1}-B_{t_2} \sim \mathcal{N}(0,t_1-t_2)" />.
The idea in SGLD has close connections to Physics. Suppose there is a force on an object downwards. Unlike the ideal scenario where the particle goes exactly downwards (similar to SGD), there would be diffusion due to particles colliding with each other. Under this case, the particle will follow a brownian motion downwards, something SGLD tries to capture. Instead of going straight downwards, it adds a random noisy term which depicts the diffusion process. Under this scneario, the weight upgrade would go as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?w_{t+1} = w_t - \alpha \bigtriangledown f(w_t) + \sqrt{\alpha}dB_t" title="w_{t+1} = w_t - \alpha \bigtriangledown f(w_t) + \sqrt{\alpha}dB_t" />
</p>

Intuitively, this works similar to using gradient descent, but adding a noise when doing so. This would provide a chance for the model to escape local optima using this noise. This approach is predominantly used with annealing of noise such that the model would be able to reach global optima when trained for enough timesteps.

### Gibbs distribution

The Gibbs distribution is also connected to statistical mechanics, where we denote the probability of observing a particle is directly proportional to the exponential of the negative energy of the system:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?p(x) \propto \exp^{-\beta h(x)}" title="p(x) \propto \exp^{-\beta h(x)}" />
</p>

where <img src="https://latex.codecogs.com/svg.latex?-\beta" title="\beta" /> is a constant, and <img src="https://latex.codecogs.com/svg.latex?h(x)" title="h(x)" /> is the energy of the system. In this work, they update the Gibbs distribution by adding a regularizer term that ensures the neighborhood is close to <img src="https://latex.codecogs.com/svg.latex?x" title="x" />.


## Local Entropy

Instead of minimizing the original loss <img src="https://latex.codecogs.com/svg.latex?f(x)" title="f(x)" />, the authors propose to maximize:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?F(x,\gamma) = \log \int _{x'\in\mathbb{R}} \text{exp}(-f(x')-\frac{\gamma}{2}\left \| x-x' \right \|_2^2)dx'" title="F(x,\gamma) = \log \int _{x'\in\mathbb{R}} \text{exp}(-f(x')-\frac{\gamma}{2}\left \| x-x' \right \|_2^2)dx'" />
</p>

The above is a log-partition function that measures both the depth of the valley at a location and it's flatness through the entropy; called 'local entropy'. The parameter <img src="https://latex.codecogs.com/svg.latex?\gamma" title="\gamma" /> biases the modified loss function towards <img src="https://latex.codecogs.com/svg.latex?x" title="x" />. In this work, they set the inverse temperature <img src="https://latex.codecogs.com/svg.latex?\beta" title="\beta" /> to 1 because <img src="https://latex.codecogs.com/svg.latex?\gamma" title="\gamma" /> affords a similar control on the Gibbs distribution as well.


Furthermore, an interesting point highlighted in the paper is that critical points with high training error are exponentially likely to be saddle points with many negative directions and all local minima are likely to have error that is very close to that of the global minimum.

<p align="center">
<b> Local entropy concentrates on wide valleys in the energy landscape.</b>
</p>
<p align="center">
<img src="/assets/Papers/5/Figure-6.png?raw=true" alt="Figure 1"/>
</p>
