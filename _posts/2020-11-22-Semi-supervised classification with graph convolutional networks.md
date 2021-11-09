---
layout: post
title: Semi-supervised classification with graph convolutional networks
published: true
---

An overview of the paper “[Semi-supervised classification with graph convolutional networks](https://arxiv.org/pdf/1609.02907.pdf)”.
<!--break-->
The author presents a scalable approach for semi-supervised learning on graph structured data that is based on an efficient variant of CNN which operate directly on graphs. All images and tables in this post are from their paper.

## Introduction

Here, the authors consider a problem of classifying nodes in a graph, where labels are only available for a small subset of nodes. This problem can be framed as graph-based semi-supervised learning, where label information is smoothed over the graph via same form of explicit graph-based regularization by using a graph laplacian regularization term in the loss function <img src="https://latex.codecogs.com/svg.latex?\inline&space;L&space;=&space;L_0&space;&plus;&space;\lambda&space;L_{reg}" title="L = L_0 + \lambda L_{reg}" /> where:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L_{reg}&space;=&space;\sum&space;_{i,j}&space;A_{ij}\left&space;\|&space;f(X_i)-f(X_j)&space;\right&space;\|^2&space;=&space;f(X)^T\bigtriangleup&space;f(X)" title="L_{reg} = \sum _{i,j} A_{ij}\left \| f(X_i)-f(X_j) \right \|^2 = f(X)^T\bigtriangleup f(X)" />
</p>

Here <img src="https://latex.codecogs.com/svg.latex?\inline&space;L_0" title="L_0" /> denotes the supervised loss w.r.t. the labeled part of the graph, <img src="https://latex.codecogs.com/svg.latex?\inline&space;f(.)" title="f(.)" /> can be a neural network like differentiable function, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\lambda" title="\lambda" /> is a weighing factor and <img src="https://latex.codecogs.com/svg.latex?\inline&space;X" title="X" /> is a matrix of node features. <img src="https://latex.codecogs.com/svg.latex?\inline&space;\bigtriangleup&space;=D-A" title="\bigtriangleup =D-A" /> denotes the unnormalized graph Laplacian of an undirected graph.
The above equation relies on the assumption that connected nodes in the graph are likely to share the same label. This assumption, however, might restrict model capacity, as graph edges need not necessarily encode node similarity, but could contain additional information.

## Fast approximate convolutions on graphs

Here, we consider a multi-layer Graph Covolutional Network with the following layer-wise propagation rule:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?H^{(l&plus;1)}=\sigma(\widetilde{D}^{-0.5}&space;\widetilde{A}&space;\widetilde{D}^{-0.5}&space;H^{(l)}&space;W^{(l)})" title="H^{(l+1)}=\sigma(\widetilde{D}^{-0.5} \widetilde{A} \widetilde{D}^{-0.5} H^{(l)} W^{(l)})" />
</p>
Here, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\widetilde{A}&space;=&space;A&space;&plus;&space;I_N" title="\widetilde{A} = A + I_N" /> is the adjacency matrix of the undirected graph with added self-connections. <img src="https://latex.codecogs.com/svg.latex?\inline&space;\widetilde{D}" title="\widetilde{D}" /> is the degree matrix of <img src="https://latex.codecogs.com/svg.latex?\inline&space;\widetilde{A}" title="\widetilde{A}" />.
A is the adjacency matrix + self loops. This is done so that each node includes its own features at its next representaion.
D is the degree matrix of A. Degree matrix is used to normalize nodes with high degrees. <img src="https://latex.codecogs.com/svg.latex?\inline&space;H^{(l)}" title="H^{(l)}" /> denotes the activations of the last layer such that <img src="https://latex.codecogs.com/svg.latex?H^{0}&space;=&space;X" title="H^{0} = X" />

### Spectral Graph convolutions

Here, we consider spectral convolutions on graphs defines as the multiplation of a signal <img src="https://latex.codecogs.com/svg.latex?\inline&space;x&space;\in&space;\mathbb{R}^N" title="x \in \mathbb{R}^N" /> (a scalar for every node) with a filter <img src="https://latex.codecogs.com/svg.latex?\inline&space;g_{\Theta}&space;=&space;diag(\Theta)" title="g_{\Theta} = diag(\Theta)" /> parametrized by <img src="https://latex.codecogs.com/svg.latex?\inline&space;\theta&space;\in&space;\mathbb{R}^N" title="\theta \in \mathbb{R}^N" />, i.e.:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?g(\theta)\star&space;x&space;=&space;UG_{\theta}U^Tx" title="g(\theta)\star x = UG_{\theta}U^Tx" />
</p>
We can understand <img src="https://latex.codecogs.com/svg.latex?\inline&space;g_{\theta}" title="g_{\theta}" /> as a function of the eigenvalues of the normalized graph Laplacian matrix <img src="https://latex.codecogs.com/svg.latex?\inline&space;L&space;=&space;I_N&space;-&space;D^{-0.5}AD^{-0.5}&space;=&space;U&space;\wedge&space;U^T" title="L = I_N - D^{-0.5}AD^{-0.5} = U \wedge U^T" />, with a diagonal matrix of its eigenvalues <img src="https://latex.codecogs.com/svg.latex?\inline&space;\wedge" title="\wedge" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;U^Tx" title="U^Tx" /> is the graph fourier transform of <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" />.
However, computing the equation <img src="https://latex.codecogs.com/svg.latex?\inline&space;g(\theta)\star&space;x" title="g(\theta)\star x" /> is computationally expensive. To circumvent this problem, it was suggested that <img src="https://latex.codecogs.com/svg.latex?\inline&space;g(\theta)" title="g(\theta)" /> can be approximated by a truncated expansion in terms of Chebyshev polynomials <img src="https://latex.codecogs.com/svg.latex?\inline&space;T_k(x)" title="T_k(x)" /> upto an order of <img src="https://latex.codecogs.com/svg.latex?\inline&space;K" title="K" />.
This approximation only depends only on nodes that are at maximum <img src="https://latex.codecogs.com/svg.latex?\inline&space;K" title="K" /> steps away from the central node.

### Layer wise linear Model

A neural network model based on graph colvolutions can therefore be built by stacking multiple convolutional layers of the form described above, each layer followed by a point-wise non-linearity. The idea is that we can recover a rich class of convolutional filter functions by stacking such layers. Intutively, such networks can alleviate the problem of overfitting on local neighborhood structures for graphs with very wide node degree distributions.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?g_{\theta}&space;\star&space;x&space;\approx&space;{\theta_0}'x&space;-&space;{\theta_1}'D^{-0.5}AD^{-0.5}x" title="g_{\theta} \star x \approx {\theta_0}'x - {\theta_1}'D^{-0.5}AD^{-0.5}x" />
</p>
In this linear formulation, we only consider immediate neighbors of the node. Successive application of such filters of this form then effectively convolve the <img src="https://latex.codecogs.com/svg.latex?\inline&space;k^{th}" title="k^{th}" /> order neigborhood of a node, where <img src="https://latex.codecogs.com/svg.latex?\inline&space;K" title="K" /> is the number of successive filterning operations or convolutional layers in a neural network model. To cut down on the number of parameters, we use <img src="https://latex.codecogs.com/svg.latex?\inline&space;\theta&space;=&space;{\theta_0}'=-{\theta_1}'" title="\theta = {\theta_0}'=-{\theta_1}'" /> such that above equation is approximated to:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?g_{\theta}&space;\star&space;x&space;\approx&space;\theta&space;(I_N&space;&plus;&space;D^{-0.5}AD^{-0.5})x" title="g_{\theta} \star x \approx \theta (I_N + D^{-0.5}AD^{-0.5})x" />
</p>
However, the term <img src="https://latex.codecogs.com/svg.latex?\inline&space;I_N&space;&plus;&space;D^{-0.5}AD^{-0.5}" title="I_N + D^{-0.5}AD^{-0.5}" /> now has eigenvalues in the range [0,2]. Repeated application of this operator can therefore lead to numerical instabilities and exploding/vanishing gradiesnt when used in a deep learning model. To alleviate this problem, they introduced the renormalization trick, of using <img src="https://latex.codecogs.com/svg.latex?\inline&space;I_N&space;&plus;&space;D^{-0.5}AD^{-0.5}&space;\rightarrow&space;\widetilde{D}^{-0.5}\widetilde{A}\widetilde{D}^{-0.5}" title="I_N + D^{-0.5}AD^{-0.5} \rightarrow \widetilde{D}^{-0.5}\widetilde{A}\widetilde{D}^{-0.5}" />, whose components have been discussed earlier.
We finally obtain
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Z&space;=&space;\widetilde{D}^{-0.5}&space;\widetilde{A}&space;\widetilde{D}^{-0.5}&space;X&space;\Theta" title="Z = \widetilde{D}^{-0.5} \widetilde{A} \widetilde{D}^{-0.5} X \Theta" />
</p>
which is intuitively similar to computing <img src="https://latex.codecogs.com/svg.latex?\inline&space;X&space;\theta" title="X \theta" /> and propagating these outputs across all nodes using the adjacency matrix.

<p align="center">
<b>Graph Convolutional Network</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/9/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

Now, all that's left is to define a loss function and update <img src="https://latex.codecogs.com/svg.latex?\Theta" title="\Theta" /> using backpropagation and gradient descent.

<p align="center">
<b>Hidden Layer Activations using TSNE</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/9/Figure-2.png?raw=true" alt="Figure 2"/>
</p>
