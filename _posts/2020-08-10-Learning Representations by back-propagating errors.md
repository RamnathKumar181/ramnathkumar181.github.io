---
layout: post
title: Learning Representations by back-propagating errors
published: true
---

An overview of the paper “[Learning Representations by back-propagating errors](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)”.
<!--break-->
The paper's proposes a novel approach for learning in networks. The procedure repeatedly adjusts the weights of the connections in the network so as to minimize a measure in the difference between the actual output vector of the net and the desired output vector. As a result of the weight adjustments, internal hidden units which are not part of the input or output come to represent important features of the task domain, and the regularities in the task are captured by the interactions of these units. All images and tables in this post are from their paper.

## Network

The total input, <img src="https://latex.codecogs.com/svg.latex?x_j" title="x_j" />, to unit <img src="https://latex.codecogs.com/svg.latex?j" title="j" /> is a linear function of the outputs <img src="https://latex.codecogs.com/svg.latex?y_j" title="y_j" />, of the units that are connected to <img src="https://latex.codecogs.com/svg.latex?j" title="j" /> and of weights <img src="https://latex.codecogs.com/svg.latex?w_{ji}" title="w_{ji}" /> on these connections. Units can be given biases by introducing an extra input to each unit which always has a 1. The weight on this extra input is called bias and is equivalent to a threshold of the opposite sign. Let us use a simple error function such as mean square error. To minimize <img src="https://latex.codecogs.com/svg.latex?\mathbb{E}" title="\mathbb{E}" /> by gradient descent, it is necessary to compute the partial derivative of <img src="https://latex.codecogs.com/svg.latex?\mathbb{E}" title="\mathbb{E}" /> wrt each weight in the network. For a given case, the partial derivatives of the error wrt each weight are computed in two passes. The forward pass is used to compute the output. The backward pass is more complicated. We first compute gradient with respect to <img src="https://latex.codecogs.com/svg.latex?y" title="y" /> (output variable). Then, by chain rule, we compute derivative of <img src="https://latex.codecogs.com/svg.latex?\mathbb{E}" title="\mathbb{E}" /> wrt <img src="https://latex.codecogs.com/svg.latex?x" title="x" /> as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;\mathbb{E}}{\partial&space;x}&space;=&space;\frac{\partial&space;\mathbb{E}}{\partial&space;y}&space;\frac{\partial&space;y}{\partial&space;x}" title="\frac{\partial \mathbb{E}}{\partial x} = \frac{\partial \mathbb{E}}{\partial y} \frac{\partial y}{\partial x}" />
</p>

If <img src="https://latex.codecogs.com/svg.latex?y" title="y" /> is a sigmoid function,then <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;y}{\partial&space;x}&space;=&space;y(1-y)" title="\frac{\partial y}{\partial x} = y(1-y)" />. This means that we know how a change in the total input <img src="https://latex.codecogs.com/svg.latex?x" title="x" /> to an output unit will affect the error. But, this total input is just a linear function of the states of the lower level units and it is also a linear function of the weights on the connections, so it is easy to compute how the error will be effected by changing these states and weights.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;\mathbb{E}}{\partial&space;w_{ji}}&space;=&space;\frac{\partial&space;\mathbb{E}}{\partial&space;x_{j}}&space;\frac{\partial&space;x_{j}}{\partial&space;w_{ji}}&space;=&space;\frac{\partial&space;\mathbb{E}}{\partial&space;x_{j}}y_i" title="\frac{\partial \mathbb{E}}{\partial w_{ji}} = \frac{\partial \mathbb{E}}{\partial x_{j}} \frac{\partial x_{j}}{\partial w_{ji}} = \frac{\partial \mathbb{E}}{\partial x_{j}}y_i" />
</p>

To compute <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;\mathbb{E}}{\partial&space;y}" title="\frac{\partial \mathbb{E}}{\partial y}" />, we need to solve the following:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;\mathbb{E}}{\partial&space;y_{i}}&space;=&space;\frac{\partial&space;\mathbb{E}}{\partial&space;x_{j}}&space;\frac{\partial&space;x_{j}}{\partial&space;y_{i}}&space;=&space;\frac{\partial&space;\mathbb{E}}{\partial&space;x_{j}}w_{ji}" title="\frac{\partial \mathbb{E}}{\partial y_{i}} = \frac{\partial \mathbb{E}}{\partial x_{j}} \frac{\partial x_{j}}{\partial y_{i}} = \frac{\partial \mathbb{E}}{\partial x_{j}}w_{ji}" />
</p>

Hence, <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;\mathbb{E}}{\partial&space;y_i}" title="\frac{\partial \mathbb{E}}{\partial y_i}" /> is just summation of all <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;\mathbb{E}}{\partial&space;x_j}w_{ji}" title="\frac{\partial \mathbb{E}}{\partial x_j}w_{ji}" />.
This basic formula is the basis of the backpropagation algorithm used in neural networks.
