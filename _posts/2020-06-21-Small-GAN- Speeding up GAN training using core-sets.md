---
layout: post
title: Small-GAN- Speeding up GAN training using core-sets
published: true
---

An overview of the paper “[Small-GAN- Speeding up GAN training using core-sets](https://arxiv.org/pdf/1910.13540)”.
<!--break-->
The author proposes a brief methodology where we extract a smaller batch which mimics the coverage of a larger batch. All images and tables in this post are from their paper.

## Introduction

It has previously been studied that training with larger batches yield better results when compared to smaller batches. However, the training on harder batches require very high computational power. Since, such a model is impractical, the authors tried to achieve the same using smaller batches. This experiment was performed on the GAN, and hence, the name. The idea is to extract a smaller batch which mimics the coverage of the larger batch.

### Generative Adversarial Networks

A Generative Adversarial Network (or GAN) is a system of two networks trained 'adversarially'. The generator <img src="https://latex.codecogs.com/svg.latex?G" title="G" />, takes input samples from a prior <img src="https://latex.codecogs.com/svg.latex?z&space;\sim&space;p(z)" title="z \sim p(z)" /> and outputs the learned distributions, <img src="https://latex.codecogs.com/svg.latex?G(z)" title="G(z)" />. The discriminator, <img src="https://latex.codecogs.com/svg.latex?D" title="D" />, receives as input, both the training examples, <img src="https://latex.codecogs.com/svg.latex?X" title="X" />, and the synthesized samples, <img src="https://latex.codecogs.com/svg.latex?G(z)" title="G(z)" />, and outputs a distribution <img src="https://latex.codecogs.com/svg.latex?D(.)" title="D(.)" /> over the possible sample source. The discriminator is trained to maximize:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L_D&space;=&space;-\mathbb{E}_{x&space;\sim&space;P_{data}}[\log&space;D(x)]&space;-&space;\mathbb{E}_{z&space;\sim&space;P_{z}}[\log(1-D(G(z)))]" title="L_D = -\mathbb{E}_{x \sim P_{data}}[\log D(x)] - \mathbb{E}_{z \sim P_{z}}[\log(1-D(G(z)))]" />
</p>
while the generator is trained to trick the discriminator by minimizing:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L_G&space;=&space;-\mathbb{E}_{z&space;\sim&space;P_{z}}[\log&space;D(G(z))]" title="L_G = -\mathbb{E}_{z \sim P_{z}}[\log D(G(z))]" />
</p>

### Inception score and Frechet Inception Distance

The FID is used to measure the effectiveness of an image synthesis model. This score is derived from using a pre-trained Imagenet classifier, and hence, the name. One further assumption is that the activations of the penultimate layer of the classifier come from a multivariate Gaussian. If the activation on real data are <img src="https://latex.codecogs.com/svg.latex?N(m,C)" title="N(m,C)" /> and the activations on the fake data are <img src="https://latex.codecogs.com/svg.latex?N(m_w,C_w)" title="N(m_w,C_w)" />, then the FID is defined as:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left&space;\|&space;m&space;-&space;m_w&space;\right&space;\|_2^2&space;&plus;&space;Tr(C&space;&plus;&space;C_w&space;-2(CC_w)^{\frac{1}{2}}))" title="\left \| m - m_w \right \|_2^2 + Tr(C + C_w -2(CC_w)^{\frac{1}{2}}))" />
</p>

### Core set Selection

A core set <img src="https://latex.codecogs.com/svg.latex?Q" title="Q" />, of a set <img src="https://latex.codecogs.com/svg.latex?P" title="P" /> is a subset <img src="https://latex.codecogs.com/svg.latex?Q" title="Q" /> belongs to <img src="https://latex.codecogs.com/svg.latex?P" title="P" />, that approximates the 'shape' of <img src="https://latex.codecogs.com/svg.latex?P" title="P" />.

## Sampling Distributions

Sampling from the prior is relatively simple. We can assume the prior distribution to be an uniform distribution. We do have the freedom to do so as well, so no issues here.

Sampling from the target distribution is more tricky. Taking pairwise distance might not work due to high concentration of images. Furthermore, simple metrics such as eucledian distance do not work as they lack any semantic significance. To avoid these issues, they created Inception embeddings of their data. They projected these embeddings on lower dimensions and applied eucledian pairwise distance to this set. The previous step further reduces the time taken computationally. We can then apply core-set sampling to these representations to select images. Once, we obtain the core-set, we need to apply the inverse embedding function to obtain the original image as well.

The process is task agnostic and could be applied to any variant of GAN (or any similar neural network in my opinion).
