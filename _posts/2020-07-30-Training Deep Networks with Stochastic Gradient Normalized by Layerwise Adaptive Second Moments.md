---
layout: post
title:  Training Deep Networks with Stochastic Gradient Normalized by Layerwise Adaptive Second Moments
published: true
---

An overview of the paper “[Training Deep Networks with Stochastic Gradient Normalized by Layerwise Adaptive Second Moments](https://arxiv.org/pdf/1905.11286.pdf)”.
<!--break-->

The paper's motivation is to propose a new optimization method, called NovoGrad. SGD with momentum and Adam are two of the most popular optimization methods. SGD with momentum is the preferred algorithm for computer vision, while Adam is perceived safer and more commonly used for NLP. Moreover, Adam can have its second moment vanish or explode, especially during the initial phase of training. However, the NovoGrad method equally performs well on image classification, speech recognition, machine translation, and language modeling and has strong regularization properties. All images and tables in this post are from their paper.

## Basics

NovoGrad is a type of stochastic normalized gradient descent:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?w_{t&plus;1}&space;=&space;w_t&space;-&space;\lambda_t&space;.&space;\frac{g_t}{\left&space;\|&space;g_t&space;\right&space;\|}" title="w_{t+1} = w_t - \lambda_t . \frac{g_t}{\left \| g_t \right \|}" />
</p>


Ignoring the gradient magnitude makes SNGD robust to vanishing and exploding gradients, but will be still be sensitive to "noisy" gradients during the initial training phase. One can improve SNGD by gradient averaging such as Adagrad, Adam, RmsProp. Adam however, does not take the normalization of gradients. Adam, the most popular one, uses moving averages <img src="https://latex.codecogs.com/svg.latex?m_t" title="m_t" />, <img src="https://latex.codecogs.com/svg.latex?v_t" title="v_t" />.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?m_t&space;=&space;\beta&space;_1&space;.m_{t-1}&space;&plus;&space;(1-\beta_1).g_t" title="m_t = \beta _1 .m_{t-1} + (1-\beta_1).g_t" />

<img src="https://latex.codecogs.com/svg.latex?v_t&space;=&space;\beta&space;_2&space;.v_{t-1}&space;&plus;&space;(1-\beta_2).g_t^2" title="v_t = \beta _2 .v_{t-1} + (1-\beta_2).g_t^2" />

<img src="https://latex.codecogs.com/svg.latex?w_{t&plus;1}&space;=&space;w_t&space;-&space;\lambda_t.&space;\frac{m_t}{\sqrt{v_t}&plus;\epsilon&space;}" title="w_{t+1} = w_t - \lambda_t. \frac{m_t}{\sqrt{v_t}+\epsilon }" />
</p>

## Layer wise Gradient Normalization

We can try to combine SNGD with layer-wise gradient normalization. Here, we replace <img src="https://latex.codecogs.com/svg.latex?g_t" title="g_t" /> with <img src="https://latex.codecogs.com/svg.latex?\widehat{g_t}&space;=&space;\frac{g_t^l}{\left&space;\|&space;g_t^l&space;\right&space;\|}" title="\widehat{g_t} = \frac{g_t^l}{\left \| g_t^l \right \|}" /> for layer <img src="https://latex.codecogs.com/svg.latex?l" title="l" />. We also define <img src="https://latex.codecogs.com/svg.latex?m_t^l" title="m_t^l" /> and <img src="https://latex.codecogs.com/svg.latex?v_t^l" title="v_t^l" />.

## Improving Adam Generalization

Adaptive methods like Adam generalize worse than SGD with momentum. Some suggest using Adam only for initial stages, and then switch to SGD. To improve adam regularization, AdamW was proposed which decouples the weight decay d as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?w_{t&plus;1}&space;=&space;w_t&space;-&space;\lambda_t.&space;(\frac{m_t}{\sqrt{v_t}&plus;\epsilon}&space;&plus;d.w_t)" title="w_{t+1} = w_t - \lambda_t. (\frac{m_t}{\sqrt{v_t}+\epsilon} +d.w_t)" />
</p>

## Reduction of Adam Memory Footprint

Unfortunately, since we might need to store the second moments for all layers, the memory required will be very large and almost impossible for very deep networks. Instead, they proposed an AdaFactor model which replaced the full second moment with moving averages of the row and column sums of squared gradients.

## NovoGrad Algorithm

NovoGrad combines three ideas: (1) use layer-wise second moments, (2) compute first moment with gradients normalized with layer-wise second moments,
(3) decouple weight decay. The exact math is a bit different, but all the properties are conserved. Note, that the final updation happens more like the SGD than the Adam algorithm. Also note that, NovoGrad uses a learning rate between SGD and Adam.
