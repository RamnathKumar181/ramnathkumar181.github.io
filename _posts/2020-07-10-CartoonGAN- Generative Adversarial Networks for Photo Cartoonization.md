---
layout: post
title: CartoonGAN- Generative Adversarial Networks for Photo Cartoonization
published: true
---

An overview of the paper “[CartoonGAN: Generative Adversarial Networks for Photo Cartoonization](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)”.
<!--break-->
The authors propose a novel approach to transform photos of real world scenes into cartoon style images. All images and tables in this post are from their paper.

## Introduction

Manually recreating real-world scenes in cartoon styles is very laborious and involves substantial artistic skills. To obtain high-quality cartoons, artists have to draw every single line and shade each color region of target scenes. Meanwhile, existing image editing software/algorithms with standard features cannot produce satisfactory results for cartoonization. Therefore, specially designed techniques that can automatically transform real world photos to high-quality cartoon style images are very helpful and for artists, tremendous amount of time can be saved so that they can focus on more creative work. Two novel losses suitable for cartoonization are proposed: (1) a semantic content loss and (2) an edge-promoting adversarial loss for preserving clear edges.

A GAN framework consists of two CNNs. One is the generator <img src="https://latex.codecogs.com/svg.latex?G" title="G" /> which is trained to produce output that fools the discriminator. The other is the discriminator <img src="https://latex.codecogs.com/svg.latex?D" title="D" /> which classifies whether the image is from the real target manifold or synthetic.

## CartoonGAN Architecture

In CartoonGAN, the generator network <img src="https://latex.codecogs.com/svg.latex?G" title="G" /> is used to map input images to the cartoon manifold. Cartoon stylization is produced once the model is trained. Complementary to the generator network, the discriminator network <img src="https://latex.codecogs.com/svg.latex?D" title="D" /> is used to judge whether the input image is a real cartoon image.

## Loss function

The loss function <img src="https://latex.codecogs.com/svg.latex?L(G,D)" title="L(G,D)" /> consists of two parts: (1) the adversarial loss <img src="https://latex.codecogs.com/svg.latex?L_{adv}(G,D)" title="L_{adv}(G,D)" /> which drives the generator network to achieve the desired manifold transformation, and (2) the content loss <img src="https://latex.codecogs.com/svg.latex?L_{con}(G,D)" title="L_{con}(G,D)" />, which preserves the image content during cartoon stylization. They then use a simple additive form for the loss function: <img src="https://latex.codecogs.com/svg.latex?L(G,D)&space;=&space;L_{adv}(G,D)&space;&plus;&space;\omega&space;L_{con}(G,D)" title="L(G,D) = L_{adv}(G,D) + \omega L_{con}(G,D)" />.

### Adversarial Loss <img src="https://latex.codecogs.com/svg.latex?L_{adv}(G,D)" title="L_{adv}(G,D)" />

Its value indicates to what extent the output image of the generator <img src="https://latex.codecogs.com/svg.latex?G" title="G" /> looks like a cartoon image. In CartoonGAN, the goal of training the discriminator <img src="https://latex.codecogs.com/svg.latex?D" title="D" /> is to maximize the probability of assigning the correct label to <img src="https://latex.codecogs.com/svg.latex?G(p_k)" title="G(p_k)" />, the cartoon images without clear edges (i.e., <img src="https://latex.codecogs.com/svg.latex?e_j&space;\in&space;S_{data}(e)" title="e_j \in S_{data}(e)" />) and the real cartoon images (i.e., <img src="https://latex.codecogs.com/svg.latex?c_i&space;\in&space;S_{data}(c)" title="c_i \in S_{data}(c)" />), such that the generator <img src="https://latex.codecogs.com/svg.latex?G" title="G" /> can be guided correctly by transforming the input to the correct manifold.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L_{adv}(G,D)&space;=&space;\mathbb{E}_{c_i\sim&space;S_{data}(c)}[\log&space;D(c_i)]&space;&plus;&space;\mathbb{E}_{e_j\sim&space;S_{data}(e)}[\log&space;(1-D(e_j))]&space;&plus;&space;\mathbb{E}_{p_k\sim&space;S_{data}(p)}[\log&space;(1-D(G(p_k)))]" title="L_{adv}(G,D) = \mathbb{E}_{c_i\sim S_{data}(c)}[\log D(c_i)] + \mathbb{E}_{e_j\sim S_{data}(e)}[\log (1-D(e_j))] + \mathbb{E}_{p_k\sim S_{data}(p)}[\log (1-D(G(p_k)))]" />
</p>

### Content Loss <img src="https://latex.codecogs.com/svg.latex?L_{con}(G,D)" title="L_{con}(G,D)" />

In addition to transformation between correct manifolds, one more important goal in cartoon stylization is to ensure the resulting cartoon images retain semantic content of the input photos. In CartoonGAN, we adopt the high-level feature maps in the VGG network pre-trained, which has been demonstrated to have good object preservation ability.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L_{con}(G,D)&space;=&space;\mathbb{E}_{p_i\sim&space;S_{data}(p)}[\left&space;\|&space;VGG_l(G(p_i))&space;-&space;VGG_l(p_i)&space;\right&space;\|_1]" title="L_{con}(G,D) = \mathbb{E}_{p_i\sim S_{data}(p)}[\left \| VGG_l(G(p_i)) - VGG_l(p_i) \right \|_1]" />
</p>

Semantic content loss is defined using the l1 sparse regularization of VGG feature maps between the input photo and the generated cartoon image. This is due to the fact that cartoon images have very different characteristics (i.e., clear edges and smooth shading) from photos. We observe that even with a suitable VGG layer that intends to capture the image content, the feature maps may still be affected by the massive style difference. Such differences often concentrate on local regions where the representation and regional characteristics change dramatically. l1 sparse regularization is able to cope with such changes much better than the standard l2 norm.

<p align="center">
<b>Ouput from CartoonGAN.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/6/Figure-1.png?raw=true" alt="Figure 1"/>
</p>
