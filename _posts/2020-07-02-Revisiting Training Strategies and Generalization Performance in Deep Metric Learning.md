---
layout: post
title: Revisiting Training Strategies and Generalization Performance in Deep Metric Learning
published: true
---

An overview of the paper “[Revisiting Training Strategies and Generalization Performance in Deep Metric Learning](https://arxiv.org/pdf/2002.08473.pdf)”.
<!--break-->
The authors review various deep metric learning methods and propose a simple, yet effective training regularization to boost the performance of ranking basaed DML methods. All images and tables in this post are from their paper.

## Training a deep metric learning model

Learning visual similarity is important for a wide range of vision tasks, such as image clustering, face detection, or image retrieval. Measuring similarity requires learning an embedding space which captures images  and reasonably reflects similarities using a defined distance metric. One of the most adopted classes of algorithms for this task is Deep Metric Learning (DML) which leverages deep neural networks to learn such a distance preserving embedding.
The key components of a DML model can be broken down into 3 different parts:

### Objective Function

The Deep Metric Learning, we learn an embedding function <img src="https://latex.codecogs.com/svg.latex?\phi&space;:&space;\chi&space;\mapsto&space;\phi&space;\subseteq&space;\mathbb{R}^D" title="\phi : \chi \mapsto \phi \subseteq \mathbb{R}^D" /> mapping datapoints <img src="https://latex.codecogs.com/svg.latex?x&space;\in&space;\chi" title="x \in \chi" /> into an embedding space <img src="https://latex.codecogs.com/svg.latex?\phi" title="\phi" /> which allows to measure the similarity between <img src="https://latex.codecogs.com/svg.latex?x_i&space;,&space;x_j" title="x_i , x_j" /> as <img src="https://latex.codecogs.com/svg.latex?d_{\phi}(x_i,&space;x_j)&space;=&space;d(\phi(x_i),&space;\phi(x_j))" title="d_{\phi}(x_i, x_j) = d(\phi(x_i), \phi(x_j))" /> with <img src="https://latex.codecogs.com/svg.latex?d(.,.)" title="d(.,.)" /> being a predefined distance function. In order to train <img src="https://latex.codecogs.com/svg.latex?\phi" title="\phi" /> to reflect the semantic similarity defined by given labels <img src="https://latex.codecogs.com/svg.latex?y&space;\in&space;Y" title="y \in Y" />, many objective functions have been proposed based on different concepts:
* <b>Ranking-based:</b> The most popular family are ranking-based loss functions operating on pairs, triplets or larger sets of datapoints. Learning <img src="https://latex.codecogs.com/svg.latex?\phi" title="\phi" /> is defined as an ordering task, such that the distances <img src="https://latex.codecogs.com/svg.latex?\phi" title="\phi" /> between an anchor <img src="https://latex.codecogs.com/svg.latex?x_a" title="x_a" /> and positive <img src="https://latex.codecogs.com/svg.latex?x_p" title="x_p" /> of the same class is minimized, and the distances <img src="https://latex.codecogs.com/svg.latex?d_{\phi}(x_a,&space;x_n)" title="d_{\phi}(x_a, x_n)" /> of the anchor with negative sample <img src="https://latex.codecogs.com/svg.latex?d_{\phi}(x_a,&space;x_n)" title="d_{\phi}(x_a, x_n)" /> with different class labels is maximized.
* <b>Classification-based:</b> As DML is essentially solving a discriminative task, some approaches can be derived from softmax logits <img src="https://latex.codecogs.com/svg.latex?l_i&space;=&space;W_j^T\phi&space;(x_i)&space;&plus;&space;b_j" title="l_i = W_j^T\phi (x_i) + b_j" />. The goal here is to maximize the margin between classes.
* <b>Proxy-based:</b> These methods approximate the distributions for the full class by one or more learned representatives. By considering the class representatives for computing the training loss, individual samples are directly compared to an entire class. Additionally, proxy-based methods help to alleviate the issue of tuple mining which is encoutered in ranking-based loss functions.

### Data Sampling

There are two broadly types of samplers, label samplers and embedded samplers. In label samplers, we choose a heuristic based on <img src="https://latex.codecogs.com/svg.latex?n" title="n" /> Samples per class (SPC-n). In SPC-R, we select <img src="https://latex.codecogs.com/svg.latex?b-1" title="b-1" /> samples, and the last sample is made sure to have the same label as another existing sample. This ensures, that atleast one triplet exists in the batch. In embedded samplers, we try to create batches of diverse data statistics. The criteria used for this include:
* <b>Greedy Coreset Distillation (GC):</b> This criterion finds a batch by iteratively adding samples which maximize the distance from the samples that have been already selected.
* <b>Matching of distance distributions (DDM):</b> DDM aims to preserve the distance distribution of <img src="https://latex.codecogs.com/svg.latex?\beta^*" title="\beta^*" />. We randomly select m candidate mini-batches and choose the batch <img src="https://latex.codecogs.com/svg.latex?\beta" title="\beta" /> with smallest Wasserstein distance between normalized distance histograms of <img src="https://latex.codecogs.com/svg.latex?\beta" title="\beta" /> and <img src="https://latex.codecogs.com/svg.latex?\beta^*" title="\beta^*" />.
* <b>FRD-Score Matching (FRD):</b> Very similar to the previous approach. The only difference being, that we use frechet distance instead of Wasserstein distance.

### Training parameters, regularization and architecture

* <b>Architecture:</b> In recent literature, mainly 3 architectures are used- GoogLeNet, Inception-BN and ResNet50.
* <b>Weight Decay:</b> Commonly, network optimization is regularized using weight decay/L2 Optimization.
* <b>Embedding dimensionality:</b> This is harder to compare and not justified in many works.
* <b>Data Preprocessing:</b> This definitely changes the entire model weights. Unfortunately, there is no proper comparison between different preprocessing steps.
* <b>Batchsize:</b> Batchsize determines the nature of the gradient updates to the networkHowever, it is commonly not taken into account as a influential factor of variation.
* <b>Advanced DML methodologies:</b> There are many extensions to objective functions, architectures, etc.However, although extensions are highly individual, they still rely on these components.

## Analyzing DML training strategies

### Studying DML parameters and architectures

In order to warrant unbiased comparability, equal and transparent training protocols and model architectures are essential, as even small deviations can result in large deviations in performance.

<p align="center">
<b>Evaluation of DML pipeline parameters and architectures.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/5/Figure-1.png?raw=true" alt="Figure 1"/>
</p>


### Batch sampling impacts DML training

Their study indicates that DML benefits from data diversity in mini-batches, independent of the chosen training objective. This coincides with the general benefit of larger batch sizes. While complex mining strategies may perform better, simple heuristics like SPC-2 are sufficient.

<p align="center">
<b>Comparison of mini-batch mining strategies.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/5/Figure-2.png?raw=true" alt="Figure 2"/>
</p>

### Comparing DML models

Under the same setup, performance saturates across different methods, contrasting results reported in recent literature. Carefully trained baseline models are able to outperform state-of-the-art approaches which use considerable stronger architectures. Thus, to evaluate the true benefit of proposed contributions, baseline models need to be competitive and implemented under comparable settings.

## Generalization in Deep Metric Learning

The experiments performed by the authors indicate that representation learning under considerable shifts between training and testing distribution is hurt by excessive feature compression, but may benefit from a more densely populated embedding space.

## Rho-regularization for improved generalization

This is their proposed addition to the DML training. Implicitly regularizing the number of directions of significant variance can improve generalization.
