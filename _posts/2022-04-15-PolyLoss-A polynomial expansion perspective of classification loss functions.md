---
layout: post
title: PolyLoss-A polynomial expansion perspective of classification loss functions
published: true
---

An overview of the paper “[PolyLoss-A polynomial expansion perspective of classification loss functions](https://arxiv.org/pdf/2204.12511)”.
<!--break-->
Cross Entropy loss and focal loss are the most common choices when training deep neural networks for classification problems. Generally speaking however, a good loss function *can* take on much more flexible forms and *should* be tailored for different tasks and datasets. A simple addition of the linear term brings significant improvement on majority tasks as shown in Figure 1. All images and tables in this post are from their paper.

<p align="center">
<b> PolyLoss outperforms cross-entropy and focal loss on various models and tasks.</b>
</p>
<p align="center">
<img src="/assets/Papers/5/Figure-10.png?raw=true" alt="Figure 1"/>
</p>

Unlike prior works, this paper claims to procide a unified framework for systematically designing a better classification loss function. The PolyLoss functions serves two purposes:
* **Loss for class imbalance**
* **Robust loss to label noise**
* **Learned loss functions**

## PolyLoss

The very intuition of the paper is simple and straightforward. They experiment with Table 1.

<p align="center">
<b> Comparing different losses in the PolyLoss framework.</b>
</p>
<p align="center">
<img src="/assets/Papers/5/Figure-11.png?raw=true" alt="Figure 2"/>
</p>

The final formulation they advocate is Poly-1 which only requires tuning one hyperparameter. More importantly, their work highlights the limitation of common loss functions, and simple modification could lead to improvements even on well established state-of-the-art models. These
findings encourage exploring and rethinking the loss function design beyond the commonly used cross-entropy and focal loss, as well as the simplest Poly-1 loss proposed in this work.
