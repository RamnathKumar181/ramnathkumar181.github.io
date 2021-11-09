---
layout: post
title: Variational Graph Auto-Encoders
published: true
---

An overview of the paper “[Variational Graph Auto-Encoders](https://arxiv.org/pdf/1611.07308.pdf)”.
<!--break-->
The author presents an approach for unsupervised learning on graph structured data called the VGAE which is based on Variational auto-encoder. All images and tables in this post are from their paper.

## Probabilistic graph auto-encoder (VGAE) Model

This model uses GCN as the encoder, and a simple inner product as the decoder. The GCN outputs the latent variables/embeddings, which are then fed to the decoder. The decoder recreates the adjacency matrix given the embeddings using the inner product of two embeddings for each node.
The model optimizes conditional entropy(decoder) and KL divergence(output of encoder against a gaussian prior) of the output.

## Non-probabilistic graph auto-encoder (GAE) Model

For a non-probabilistic variant of the VGAE model, we calculate embeddings <img src="https://latex.codecogs.com/svg.latex?\inline&space;Z" title="Z" /> and the reconstructed adjacency matrix <img src="https://latex.codecogs.com/svg.latex?\inline&space;\widehat{A}" title="\widehat{A}" /> as follows:
<img src="https://latex.codecogs.com/svg.latex?\inline&space;\widehat{A}&space;=&space;\sigma&space;(ZZ^T)" title="\widehat{A} = \sigma (ZZ^T)" /> , with <img src="https://latex.codecogs.com/svg.latex?\inline&space;Z&space;=&space;$GCN$(X,A)" title="Z = $GCN$(X,A)" />.

## Discussion

Both VGAE and GAE achieve competitive results on the featureless task. Adding input features significantly improve performances across datasets. A gaussian prior is potentially a poor choice in combination with an inner product decoder, as the latter tries to push the away from zero-center. Nevertheless, the VGAE model achieves higher predictive performance across datasets as shown below:

<p align="center">
<b>Link prediction task in citation networks</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/10/Figure-1.png?raw=true" alt="Figure 1"/>
</p>
