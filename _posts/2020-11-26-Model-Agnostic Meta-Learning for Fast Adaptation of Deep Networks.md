---
layout: post
title: Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
published: true
---

An overview of the paper “[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400.pdf)”.
<!--break-->
The authors propose an algorithm for meta-learning that is model-agnostic. All images and tables in this post are from their paper.

The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples. Here, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task. In effect, the method trains the model to be easy to fine-tune.

<p align="center">
<b>Model-agnostic meta-learning algorithm.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/15/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

## Catastrophic Forgetting

One way to look at the problem of forgetting is through gradient interfence and gradient alignment.
Let's consider 2 tasks. Suppose, the task-wise gradients for a model's parameters conflict with each other in certain parts of the parameter space. The performance would thus degrade on the old tasks because the gradient updates made while learning a new task don't align with gradients directions for old tasks. On the right of the image is an ideal scenario, where the gradients align and therefore progress on learning a new task, which coincides with progress on the old ones. Ensuring gradient-alignment is therefore essential to make shared progress on task-wise objectives under limited availability of training-data. This kind of alignment across tasks can be achieved by exploiting some properties of meta-learning based gradient updates.

<p align="center">
<b>Gradient direction for various tasks at time t.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/15/Figure-2.png?raw=true" alt="Figure 2"/>
</p>

## Model Agnostic Meta-Learning (MAML)

Suppose, we want to train a model to be good at learning from a handful of samples from any data distribution, such that it performs well on unseen samples of distribution. We can think of this as wanting to optimize two objectives, the one we minimize when we learn on the handful of samples (or <img src="https://latex.codecogs.com/svg.latex?L_{inner}" title="L_{inner}" />) and the one we test the model once it has completed learning on the handful of samples (let's call this <img src="https://latex.codecogs.com/svg.latex?L_{outer}" title="L_{outer}" />, evaluated on unseen samples from the distribution). This intuitively makes sense, because the only way to make progress on unseen data (seen for <img src="https://latex.codecogs.com/svg.latex?L_{outer}" title="L_{outer}" />) is to somehow have the gradients on that data be aligned with the actual gradient steps taken by the model on some seen data (seen during <img src="https://latex.codecogs.com/svg.latex?L_{inner}" title="L_{inner}" />) (in this case, the few-shot samples). In the K-shot learning setting, the model is trained to learn a new task <img src="https://latex.codecogs.com/svg.latex?T_{i}" title="T_{i}" /> drawn from <img src="https://latex.codecogs.com/svg.latex?p(T)" title="p(T)" /> from only <img src="https://latex.codecogs.com/svg.latex?K" title="K" /> samples drawn from <img src="https://latex.codecogs.com/svg.latex?q_i" title="q_i" /> and feedback <img src="https://latex.codecogs.com/svg.latex?L_{T_i}" title="L_{T_i}" />. The model <img src="https://latex.codecogs.com/svg.latex?f" title="f" /> is then improved by considering how the test error on new data from <img src="https://latex.codecogs.com/svg.latex?q_i" title="q_i" /> changes with respect to parameters. In effect, the test error on sampled tasks <img src="https://latex.codecogs.com/svg.latex?T_i" title="T_i" /> serves as the training error of the meta-learning process. At the end of meta-training, new tasks sampled from <img src="https://latex.codecogs.com/svg.latex?p(T)" title="p(T)" />, and meta performance is measured after learning from <img src="https://latex.codecogs.com/svg.latex?K" title="K" /> samples.

The aim is to find model parameters that are sensitive to changes in the task, such that small changes in the parameters will produce large improvements on the loss function of any task drawn. The complete algorithm can be broken down into few simple steps:
* Create disjoint subset of tasks (eg. dogs, cats, etc.)
* Make inner updates using one task and repeat for all tasks
* Do an overall updation using all the losses, thus obtained, outer update.

<p align="center">
<b>MAML Algorithm.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/15/Figure-3.png?raw=true" alt="Figure 3"/>
</p>
