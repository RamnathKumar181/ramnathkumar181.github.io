---
layout: post
title: Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
published: true
---

An overview of the paper “[Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)”.
<!--break-->
The paper successfully proposes a novel neural network model called the RNN Encoder-Decoder for phase based statistical machine translation system. All images and tables in this post are from their paper.

## RNN

An RNN has a hidden unit <img src="https://latex.codecogs.com/svg.latex?h" title="h" /> and an output <img src="https://latex.codecogs.com/svg.latex?y" title="y" />. It operates on a variable sized input <img src="https://latex.codecogs.com/svg.latex?x" title="x" />. At each step, the hidden unit is updated based on its previous value and current input. It learns to predict the next symbol in a sequence. In this case, we need a conditional probability of word <img src="https://latex.codecogs.com/svg.latex?t" title="t" /> occuring given the previous <img src="https://latex.codecogs.com/svg.latex?t-1" title="t-1" /> words.

## RNN Encoder-Decoder

The encoder is an RNN that reads each symbol of an input sequence <img src="https://latex.codecogs.com/svg.latex?x" title="x" /> sequentially. This updates the hidden state at each step. At the end, the hidden state acts as a summary of the entire sequence. The decoder is also an RNN which predicts the next symbol given the hidden state. The only difference being that the the predicted symbol and the hidden state also depend on the previous symbol. The entire net is trained to maximize log likelihood of the translation given a sequence input.  The given network can be used in two ways:
* Generate a target sequence given an input sequence
* Score a given pair of input and output sequence, where score is simply <img src="https://latex.codecogs.com/svg.latex?p(y|x)" title="p(y|x)" />.

## Hidden Unit that Adaptively Remembers and Forgets

They also proposed a new type of hidden unit. This hidden unit is motivated by the LSTM but is much simpler to compute. The idea is that, the reset gate and the update gate are learned over the iterations. When the reset gate is close to 0, the hidden state is forced to ignore the previous hidden state and reset with current input only. On the other hand, the update gate controls how much information from the previous hidden state will carry over. The intuition here is similar to LSTM. As each hiddent unit has seperate reset and update gates, each unit will learn to capture dependencies of different time scales.

<p align="center">
<b>An illustration of the proposed RNN Encoder-Decoder.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/4/Figure-3.png?raw=true" alt="Figure 3"/>
</p>
