---
layout: post
title: Combining label propagation and simple models out-performs graph neural networks
published: true
---

An overview of the paper “[Combining label propagation and simple models out-performs graph neural networks](https://arxiv.org/pdf/2010.13993.pdf)”.
<!--break-->
The authors show that we can simply use a a shallow model that ignore the graph structure with two simple post processing steps that exploit correlation in label structure to achieve SOTA in graph node classification problems. The post-processing steps include:
* An error correlation that spreads residual errors in training data to correct errors in test data, and
* A prediction correlation that smooths the prediction on test data.
They call the entire procedure Correct and Smooth (C&S). The base model made with node features ignores the graph structure. In this framework, the graph structure is not used to learn parameters but instead as a post-processing mechanism. A major source of the performance improvements is directly using labels for predictions. All images and tables in this post are from their paper.

## Correct and Smooth Model

### Basic Notations

We assume that we have an undirected graph <img src="https://latex.codecogs.com/gif.latex?\inline&space;G&space;=&space;(V,E)" title="G = (V,E)" />, where there are <img src="https://latex.codecogs.com/gif.latex?\inline&space;n&space;=&space;\left&space;|&space;V&space;\right&space;|" title="n = \left | V \right |" /> nodes with features on each node represented by a matrix <img src="https://latex.codecogs.com/gif.latex?\inline&space;X_{n*p}" title="X_{n*p}" /> matrix.
Let <img src="https://latex.codecogs.com/gif.latex?\inline&space;A" title="A" /> be the adjacency matrix of the graph, and <img src="https://latex.codecogs.com/gif.latex?\inline&space;D" title="D" /> be the diagonal degree matrix, and <img src="https://latex.codecogs.com/gif.latex?\inline&space;S" title="S" /> be the normalized adjacency matrix <img src="https://latex.codecogs.com/gif.latex?\inline&space;D^{-\frac{1}{2}}AD^{\frac{1}{2}}" title="D^{-\frac{1}{2}}AD^{\frac{1}{2}}" />.
For the prediction problem, the node set <img src="https://latex.codecogs.com/gif.latex?\inline&space;V" title="V" /> is split into a disjoint set of unlabelled nodes <img src="https://latex.codecogs.com/gif.latex?\inline&space;U" title="U" /> and labeled nodes <img src="https://latex.codecogs.com/gif.latex?\inline&space;L" title="L" />. We represent the labels using one-hot encoding. We further split the labeled nodes into a training set and validation set.

### Basic Approach

The approach starts with a simple base predictor on node features, which does not rely on any learning over the graph. After, thy perform two types of label propagation(LP): one that corrects the base predictions by modelling correlated error and one that smooths the final prediction. An overview of the correct and smooth model is described in the figure below:

<p align="center">
<b>Overview of the Correct and Smooth model.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/11/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

### Simple base predictor

Here, a simple shallow MLP was used with softmax at the last layer and cross-entropy, as the loss function. This was trained on the training set. The validation set was used to tune hyperparameters such as learning rates, and the hidden layer dimensions. Finally, we obtain a matrix <img src="https://latex.codecogs.com/gif.latex?\inline&space;Z_{n*c}" title="Z_{n*c}" />.

### Correcting for error in base predictions with residual propagation

The key idea is that errors in base prediction to be possitively correlated along edges in the graph. In other words, an error at node <img src="https://latex.codecogs.com/gif.latex?\inline&space;i" title="i" /> increases the chances of a similar error at neighboring nodes of <img src="https://latex.codecogs.com/gif.latex?\inline&space;i" title="i" />. For this, we first define an error matrix <img src="https://latex.codecogs.com/gif.latex?\inline&space;E_{n*c}" title="E_{n*c}" />, where the error in residual on training data and zero everywhere else. We then smooth the error using label spreading technique optimizing an objective of two terms:
* First term encourages smoothness of error estimation over the graph
* The second terms keeps the solution close to the initial guess <img src="https://latex.codecogs.com/gif.latex?\inline&space;E" title="E" /> of the error.

After performing this step, we obtain the optimal <img src="https://latex.codecogs.com/gif.latex?\inline&space;E" title="E" />. Adding this <img src="https://latex.codecogs.com/gif.latex?\inline&space;E" title="E" /> to <img src="https://latex.codecogs.com/gif.latex?\inline&space;Z" title="Z" /> gives us the corrected predictions. However, the optimal error <img src="https://latex.codecogs.com/gif.latex?\inline&space;E" title="E" /> might not be of the right scale, hence, we need to scale the residuals.

* Autoscale: Intuitively, the scale of size in errors in optimal <img src="https://latex.codecogs.com/gif.latex?\inline&space;E" title="E" /> to be approximately the size of errors in <img src="https://latex.codecogs.com/gif.latex?\inline&space;E" title="E" />.
* Scaled Fixed Diffusion: Alternatively, we can use a diffusion which keeps the known erros at training nodes fixed. Intuitively, this fixes error values where we know the error (on labeled nodes <img src="https://latex.codecogs.com/gif.latex?\inline&space;L" title="L" />) while other nodes keep averaging over the values of their neighbors until convergence.

### Smoothing final predictions with prediction correlation

At this point, we have a score vector <img src="https://latex.codecogs.com/gif.latex?\inline&space;Z" title="Z" />, obtained from collecting the base predictor <img src="https://latex.codecogs.com/gif.latex?\inline&space;Z" title="Z" /> with a model for the correlated error <img src="https://latex.codecogs.com/gif.latex?\inline&space;E" title="E" />. To make a final prediction, we further smooth the corrected predictions. The motivation is that the adjacent nodes in the graph are likely to have similar labels, which is expected given homophily or assortative properties of a network. Thus, we can encourage smoothness over the distribution over labels by another label propagation. The performance of the model on the datasets is shown in the figure below.

<p align="center">
<b>Performance of the C&S model on different datasets.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/11/Figure-2.png?raw=true" alt="Figure 2"/>
</p>
