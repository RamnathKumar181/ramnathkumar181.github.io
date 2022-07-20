---
layout: post
title: The Real and Complex Number Systems
published: true
book_title: Real Analysis
---

An overview of the chapter "The Real and Complex Number Systems" from the famous books of "[Principles of Mathematical Analysis](https://web.math.ucsb.edu/~agboola/teaching/2021/winter/122A/rudin.pdf)" written by Rudin.
<!--break-->
The authors create a brief introduction of the loopholes of rational numbers, thus introducing real and complex number systems that help fill these holes.


## Introduction

**P1:** Show that <img src="https://latex.codecogs.com/svg.latex?p^2=2" title="p^2=2" /> is not satisfied by any rational <img src="https://latex.codecogs.com/svg.latex?p" title="p" />
*Proof:* Suppose <img src="https://latex.codecogs.com/svg.latex?p" title="p" /> exists, then we can write <img src="https://latex.codecogs.com/svg.latex?p=m/n" title="p=m/n" />, where m and n are integers that are not both even. If so, <img src="https://latex.codecogs.com/svg.latex?m^2=2n^2" title="m^2=2n^2" /> would also hold, indicating that m is even. If so, m would also be divisible by 4, since m needs to be an integer. If so, n must also be even, to make up for the extra 2. This would mean that both m and n share a common divisor, which is against our initial choice of m and n.

**P2:** Show that there is no largest rational number that satisfies the condition <img src="https://latex.codecogs.com/svg.latex?p^2<2" title="p^2<2" />, and similarly no smallest number that satisfies the condition <img src="https://latex.codecogs.com/svg.latex?p^2>2" title="p^2>2" />.
*Proof:* Now, suppose <img src="https://latex.codecogs.com/svg.latex?p>0" title="p>0" />, let us define q such that:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?q = p-\frac{p^2-2}{p+2}" title="q = p-\frac{p^2-2}{p+2}" />
</p>

From the above equation, we can also show that
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?q^2-2 = \frac{2(p^2-2)}{(p+2)^2}" title="q^2-2 = \frac{2(p^2-2)}{(p+2)^2}" />
</p>
If <img src="https://latex.codecogs.com/svg.latex?p^2<2" title="p^2<2" />, then <img src="https://latex.codecogs.com/svg.latex?q>p" title="q>p" />, and <img src="https://latex.codecogs.com/svg.latex?q^2<2" title="q^2<2" />. The exact reverse is true when <img src="https://latex.codecogs.com/svg.latex?p^2>2" title="p^2>2" />.

The purpose of the above two proofs is to show the loopholes in rational numbers. The real number system fills these gaps. This is the principal reason for the fundamental role which it plays in analysis.

## Ordered Sets

**P3:** Suppose <img src="https://latex.codecogs.com/svg.latex?S" title="S" /> is an ordered set with the least-upper-bound property, <img src="https://latex.codecogs.com/svg.latex?B \subset S" title="B \subset S" />, <img src="https://latex.codecogs.com/svg.latex?B" title="B" /> is not empty, and <img src="https://latex.codecogs.com/svg.latex?B" title="B" /> is bounded below. Let <img src="https://latex.codecogs.com/svg.latex?L" title="L" /> be the set of all lower bounds of <img src="https://latex.codecogs.com/svg.latex?B" title="B" />, then
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\alpha = \sup L" title="\alpha = \sup L" />
</p>
exists in <img src="https://latex.codecogs.com/svg.latex?S" title="S" />, and <img src="https://latex.codecogs.com/svg.latex?\alpha = \inf B" title="\alpha = \inf B" /> and exists in <img src="https://latex.codecogs.com/svg.latex?S" title="S" />.
*Proof:* Since <img src="https://latex.codecogs.com/svg.latex?B" title="B" /> is bounded below, <img src="https://latex.codecogs.com/svg.latex?L" title="L" /> is not empty. Thus elements in <img src="https://latex.codecogs.com/svg.latex?B" title="B" /> are always greater than elements of <img src="https://latex.codecogs.com/svg.latex?L" title="L" />. Thus, <img src="https://latex.codecogs.com/svg.latex?L" title="L" /> is bounded above. Suppose the hypothesis holds, and <img src="https://latex.codecogs.com/svg.latex?\alpha = \sup L" title="\alpha = \sup L" />, where <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" /> exists in <img src="https://latex.codecogs.com/svg.latex?S" title="S" />.
If we take a number <img src="https://latex.codecogs.com/svg.latex?\gamma" title="\gamma" /> less than <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" />, then <img src="https://latex.codecogs.com/svg.latex?\gamma" title="\gamma" /> will not be a upper bound of <img src="https://latex.codecogs.com/svg.latex?L" title="L" />, and hence not belong to <img src="https://latex.codecogs.com/svg.latex?B" title="B" />. Thus, it follows that <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" /> also serves as the lower bound of <img src="https://latex.codecogs.com/svg.latex?B" title="B" />.


## The Real Field

**P4:** If <img src="https://latex.codecogs.com/svg.latex?x \in \mathbb{R}" title="x \in \mathbb{R}" />, <img src="https://latex.codecogs.com/svg.latex?y \in \mathbb{R}" title="y \in \mathbb{R}" />, and <img src="https://latex.codecogs.com/svg.latex?x>0" title="x>0" />, then there is a positive integer <img src="https://latex.codecogs.com/svg.latex?n" title="n" /> such that
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?nx>y" title="nx>y" />
</p>
This is called the **archimedean property** of R.
*Proof:* Let A be the set of all <img src="https://latex.codecogs.com/svg.latex?nx" title="nx" />, where <img src="https://latex.codecogs.com/svg.latex?n" title="n" /> runs through the positive integers. If the above hypothesis were false, <img src="https://latex.codecogs.com/svg.latex?y" title="y" /> would be an upper bound of A. If so, we could write <img src="https://latex.codecogs.com/svg.latex?\alpha = \sup A" title="\alpha = \sup A" />. If so, we consider another point <img src="https://latex.codecogs.com/svg.latex?\alpha -x" title="\alpha -x" />, where <img src="https://latex.codecogs.com/svg.latex?\alpha -x< \alpha" title="\alpha -x < \alpha" /> since <img src="https://latex.codecogs.com/svg.latex?x>0" title="x>0" />. Then, <img src="https://latex.codecogs.com/svg.latex?\alpha -x< \alpha" title="\alpha -x < \alpha" /> is not an upper bound and could be written as <img src="https://latex.codecogs.com/svg.latex?\alpha -x< mx" title="\alpha -x < mx" />, implying <img src="https://latex.codecogs.com/svg.latex?(m+1)x> \alpha" title="(m+1)x> \alpha" />. This isn't possible since <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" /> is the upper bound of A.

**P5:** Between any two real numbers, there is a rational one. If <img src="https://latex.codecogs.com/svg.latex?x \in \mathbb{R}" title="x \in \mathbb{R}" />, <img src="https://latex.codecogs.com/svg.latex?y \in \mathbb{R}" title="y \in \mathbb{R}" />, and <img src="https://latex.codecogs.com/svg.latex?y>x" title="y>x" />, then there exists a <img src="https://latex.codecogs.com/svg.latex?p \in Q" title="p \in Q" /> such that <img src="https://latex.codecogs.com/svg.latex?x<p<y" title="x<p<y" />.
*Proof:* Since <img src="https://latex.codecogs.com/svg.latex?x<y" title="x<y" />, we have <img src="https://latex.codecogs.com/svg.latex?y-x>0" title="y-x>0" />. From the previous proof, we can show that <img src="https://latex.codecogs.com/svg.latex?n(y-x)>1" title="n(y-x)>1" />.
From the above equation, we can clearly say that <img src="https://latex.codecogs.com/svg.latex?nx<nx+1<ny" title="nx<nx+1<ny" />. Since <img src="https://latex.codecogs.com/svg.latex?nx" title="nx" /> is a real number, there must exist an integer <img src="https://latex.codecogs.com/svg.latex?m" title="m" /> between <img src="https://latex.codecogs.com/svg.latex?nx" title="nx" />, and <img src="https://latex.codecogs.com/svg.latex?nx+1" title="nx+1" /> modifying the above inequality to the following:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?nx<m\leq nx+1<ny" title="nx<m\leq nx+1<ny" />
</p>
Since <img src="https://latex.codecogs.com/svg.latex?n>0" title="n>0" />, this could be simplified to <img src="https://latex.codecogs.com/svg.latex?x<m/n<y" title="x<m/n<y" />.

**P6:** For every real <img src="https://latex.codecogs.com/svg.latex?x>0" title="x>0" /> and every integer <img src="https://latex.codecogs.com/svg.latex?n>0" title="n>0" />, there is one and only one positive real <img src="https://latex.codecogs.com/svg.latex?y" title="y" /> such that <img src="https://latex.codecogs.com/svg.latex?y^n=x" title="y^n=x" />.
*Proof:*

## Complex Numbers

**P7:** If <img src="https://latex.codecogs.com/svg.latex?a_1, ..., a_n" title="a_1, ..., a_n" /> and <img src="https://latex.codecogs.com/svg.latex?b_1, ..., b_n" title="b_1, ..., b_n" /> are complex numbers, then
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left | \sum a_j\overline{b}_j \right |^2 \leq \sum \left | a_j \right |^2 \sum \left | b_j \right |^2" title="\left | \sum a_j\overline{b}_j \right |^2 \leq \sum \left | a_j \right |^2 \sum \left | b_j \right |^2" />
</p>
This is called the *Schwarz Inequality*.
*Proof:*
