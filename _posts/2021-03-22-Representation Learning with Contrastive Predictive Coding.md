---
layout: post
title: Representation Learning with Contrastive Predictive Coding
published: true
---

An overview of the paper “[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)”.
<!--break-->
The authors propose a universal unsupervised learning approach to extract useful representations from high-dimensional data, which they call Contrastive Predictive Coding. All images and tables in this post are from their paper.

The key insight of the model is to learn such representations by predicting the future in latent space by powerful autoregressive models. Despite the importance of unsupervised learning, it is yet to see a breakthrough similar to supervised learning: modeling high-level representations from raw observations remain elusive. Furthermore, it is not always clear what the ideal representation is and it it is possible that one can learn such a representation without additional supervision or specialization to a particular data modality.

One of the most common strategies for unsupervised learning has been to predict future, missing or contextual information. This idea of predictive coding is pretty old idea. Recent unsupervised learning has successfully used these ideas to learn word representations by predicting neighboring words.

## Contrastive Predicting Coding

### Motivation and Intuitions

The main intuition behind our model is to learn the representations that encode the underlying shared information between different parts of the (high-dimensional) signal. At the same time, it discards low-level information and noise that is more local. One of the challenges of predicting high-dimensional data is that unimodal losses such as meansquared error and cross-entropy are not very useful, and powerful conditional generative models which need to reconstruct every detail in the data are usually required. But these models are computationally intense, and waste capacity at modeling the complex relationships in the data <img src="https://latex.codecogs.com/svg.latex?x" title="x" />, often ignoring the context <img src="https://latex.codecogs.com/svg.latex?c" title="c" />. For example, images may contain thousands of bits of information while the high-level
latent variables such as the class label contain much less information (10 bits for 1,024 categories). This suggests that modeling <img src="https://latex.codecogs.com/svg.latex?p(x|c)" title="p(x|c)" /> directly may not be optimal for the purpose of extracting shared information between <img src="https://latex.codecogs.com/svg.latex?x" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?c" title="c" />. When predicting future information they instead encode the target <img src="https://latex.codecogs.com/svg.latex?x" title="x" /> (future) and context <img src="https://latex.codecogs.com/svg.latex?c" title="c" /> (present) into a compact distributed vector representations (via non-linear learned mappings) in a way that maximally preserves the mutual information of the original signals <img src="https://latex.codecogs.com/svg.latex?x" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?c" title="c" /> defined as
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?I(x;c)&space;=&space;\sum&space;_{x,c}&space;p(x,c)\log&space;\frac{p(x|c)}{p(x)}" title="I(x;c) = \sum _{x,c} p(x,c)\log \frac{p(x|c)}{p(x)}" />
</p>
By maximizing the mutual information between the encoded representations, they extract the underlying latent variables the inputs have in common.


### Contrastive Predictive Coding

First, a non-linear encoder <img src="https://latex.codecogs.com/svg.latex?g_{enc}" title="g_{enc}" /> maps the input sequence of observations <img src="https://latex.codecogs.com/svg.latex?x_t" title="x_t" /> to a sequence of latent representations <img src="https://latex.codecogs.com/svg.latex?z_t&space;=&space;g_{enc}(x_t)" title="z_t = g_{enc}(x_t)" />, potentially with a lower temporal resolution.
Next, an autoregressive model <img src="https://latex.codecogs.com/svg.latex?g_{ar}" title="g_{ar}" /> summarizes all <img src="https://latex.codecogs.com/svg.latex?z_{\leq&space;t}" title="z_{\leq t}" /> in the latent space and produces a context latent representation <img src="https://latex.codecogs.com/svg.latex?c_t&space;=&space;g_{ar}(z_{\leq&space;t})" title="c_t = g_{ar}(z_{\leq t})" />.
they do not predict future observations <img src="https://latex.codecogs.com/svg.latex?x_{t&plus;k}" title="x_{t+k}" /> directly with a generative model <img src="https://latex.codecogs.com/svg.latex?p_k(x_{t&plus;k}|c_t)" title="p_k(x_{t+k}|c_t)" />. Instead, they model a density ratio which preserves the mutual information between <img src="https://latex.codecogs.com/svg.latex?x_{t&plus;k}" title="x_{t+k}" /> and <img src="https://latex.codecogs.com/svg.latex?c_{t}" title="c_{t}" /> as follows:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?f_k(x_{t&plus;k},c_t)&space;\propto&space;\frac{p(x_{t&plus;k}|c_t)}{p(x_{t&plus;k})}" title="f_k(x_{t+k},c_t) \propto \frac{p(x_{t+k}|c_t)}{p(x_{t+k})}" />
</p>
they can use a simple log-bilinear model:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?f_k(x_{t&plus;k},c_t)&space;=&space;\exp(z_{t&plus;k}^T&space;W_k&space;c_t)" title="f_k(x_{t+k},c_t) = \exp(z_{t+k}^T W_k c_t)" />
</p>

By using a density ratio <img src="https://latex.codecogs.com/svg.latex?f(x_{t&plus;k},c_t)" title="f(x_{t+k},c_t)" /> and inferring <img src="https://latex.codecogs.com/svg.latex?f(x_{t&plus;k},c_t)" title="f(x_{t+k},c_t)" /> with an encoder, they relieve the model from modeling the high dimensional distribution <img src="https://latex.codecogs.com/svg.latex?x_{t_k}" title="x_{t_k}" />.

<p align="center">
<b>Overview of Contrastive Predictive Coding.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/17/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

### InfoNCE Loss and Mutual Information Estimation

Both the encoder and autoregressive model are trained to jointly optimize a loss based on NCE, which they  call InfoNCE. Given a set <img src="https://latex.codecogs.com/svg.latex?X&space;=&space;\begin{Bmatrix}&space;x_1,...,x_N&space;\end{Bmatrix}" title="X = \begin{Bmatrix} x_1,...,x_N \end{Bmatrix}" /> of <img src="https://latex.codecogs.com/svg.latex?N" title="N" /> random samples containing one positive sample from <img src="https://latex.codecogs.com/svg.latex?p(x_{t&plus;k}|c_t)" title="p(x_{t+k}|c_t)" /> and <img src="https://latex.codecogs.com/svg.latex?N-1" title="N-1" /> negative samples from the 'proposal' distribution <img src="https://latex.codecogs.com/svg.latex?p(x_{t&plus;k})" title="p(x_{t+k})" />
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L_N&space;=&space;-\mathbb{E}_{X}\begin{bmatrix}&space;\log&space;\frac{f_k(x_{t&plus;k}|c_t)}{\sum&space;_{x_j\in&space;X}f_k(x_j,c_t)}&space;\end{bmatrix}" title="L_N = -\mathbb{E}_{X}\begin{bmatrix} \log \frac{f_k(x_{t+k}|c_t)}{\sum _{x_j\in X}f_k(x_j,c_t)} \end{bmatrix}" />
</p>

Optimizing the above function will result in the final output estimating the density ratio. Furthermore, the authors showed that <img src="https://latex.codecogs.com/svg.latex?I(x_{t&plus;k},c_t)\geq&space;\log(N)&space;-&space;L_N" title="I(x_{t+k},c_t)\geq \log(N) - L_N" />, which becomes tighter as <img src="https://latex.codecogs.com/svg.latex?N" title="N" /> becomes larger.
