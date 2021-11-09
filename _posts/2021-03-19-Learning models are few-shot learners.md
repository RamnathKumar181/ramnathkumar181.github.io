---
layout: post
title: Learning models are few-shot learners
published: true
---

An overview of the paper “[Learning models are few-shot learners](https://arxiv.org/pdf/2005.14165v4.pdf)”.
<!--break-->
The authors show that scaling up language models greatly improves task-agnostic, few-shot performance. All images and tables in this post are from their paper.

While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions – something which current NLP systems still largely struggle to do.

## Introduction

Recently, there is a trend towards pre-trained language representations in NLP systems, applied in increasingly flexible and task-agnostic ways for downstream transfer. Firthermore, pre-trained recurent or transformer language models have been directly fine-tuned, entirely removing the need for task-specific architectures. However, a major limitation to this approach is that while the architecture is task agnostic, there is still a need for task-specific datasets and task-specific tuning: to achieve strong performance on a desired task. The authors furthermore point that meta-learning could address these issues. However, current meta-learning approaches are far inferior to fine-tuning. Another recent trend in language modelling is the evidence suggesting that log loss, correlates well with many downstream tasks, and follows a smooth trend of improvement with scale. Since, in-context learning (meta-learning) involves absorbing many skills and tasks within the parameters of the model, it is plausible to expect similarly strong gains with scale. . By presenting a broad characterization of GPT-3’s strengths and weaknesses, including these limitations, the authors hope to stimulate study of few-shot learning in language models and draw attention to where progress is most needed.

## Related Works

The basic pre-training approach, including model, and training, is relatively straightforward scaling up of model size, dataset size and diversity, and length of training. The settings of the model can be seen as lying on a spectrum of how much task-specific data they tend to rely on:

### Fine-Tuning (FT)

This has been the most common approach in recent years, and involves updating the weights of a pre-trained model by training on a supervised dataset specific to the desired task. The main advantage is strong performance on many benchmarks. The main disadvantages are the need for a new large dataset for every task, the potential for poor generalization, out-of-distribution and the potential to explot spurious features of the training data, potentially resulting in an unfair comparison with human performance.

### In-Context Learning

* <b>Few-Shot (FS):</b> This is the term used where the model is given a few demonstrations of the task at inference time as conditioning, but no weight updates are allowed.
* <b>One-Shot (OS):</b> This is the same as few shot except that only one demonstration is allowed, in addition to a natural language description of the task.
* <b>Zero-Shot (ZS):</b> This is the same as one-shot except that no demonstrations are allowed, and the model is only given a natural language instruction describing the task.

<p align="center">
<b>Summary of possible approaches.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/14/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

## Approach

### Model and architectures

The model and architecture used in this study is the same as GPT-2, including the modified initialization, pre-normalization, and reversible tokenization described therein, with the exception that they use alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to Sparse Transformer.

### Evaluation

For few shot learning, we evaluate each example in the evaluation set by randomly drawing K exampless from that task's training set as conditioning, delimited by 1 or 2 newlines depending on the task.Conditioning refers to fine-tuning the last layer of the model with a lower learning rate.

## Results

The paper computes the performance of their in-context learning model on a very wide spectrum of experiments. In some cases, the performance nearly matching the performance of state-of-the-art fine-tuned systems, as well as generating high quality samples and strong qualitative performance at tasks defined on-the-fly. The authors also documented roughly predictable trends of scaling in performance without using fine-tuning.
