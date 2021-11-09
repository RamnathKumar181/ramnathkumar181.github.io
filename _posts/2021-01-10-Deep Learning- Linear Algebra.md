---
layout: post
title: Deep Learning- Linear Algebra
published: true
---

An overview of the chapter “[Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)” from the famous book “[Deep Learning](https://www.deeplearningbook.org/)” written by Ian Goodfellow, Yoshua Bengio and Aaron Courville.
<!--break-->
The authors create a brief introduction of the important concepts of linear algebra that help guide machine learning. All images and tables in this post are from their book.

## Scalars, Vectors, Matrices and Tensors

The study of linear algebra involves several types of mathematical objects:
* <b>Scalars:</b> A single number. For example,  
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?n&space;=&space;10" title="n = 10" />
</p>
* <b>Vectors:</b> An array of numbers. For example,
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?x&space;=&space;\begin{bmatrix}&space;x_0&space;\\&space;x_1&space;\\&space;.&space;\\&space;.&space;\\&space;.&space;\\&space;x_n&space;\end{bmatrix}" title="x = \begin{bmatrix} x_0 \\ x_1 \\ . \\ . \\ . \\ x_n \end{bmatrix}" />
</p>To access elements 1 and 3 in array, we set <img src="https://latex.codecogs.com/svg.latex?\inline&space;S&space;=&space;\begin{Bmatrix}&space;1,3&space;\end{Bmatrix}" title="S = \begin{Bmatrix} 1,3 \end{Bmatrix}" /> and use <img src="https://latex.codecogs.com/svg.latex?\inline&space;x_S" title="x_S" />. To get all elemets other than 1 and 3, we use <img src="https://latex.codecogs.com/svg.latex?\inline&space;x_{-S}" title="x_{-S}" />
* <b>Matrices:</b> 2-D array of numbers. For example,
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?A&space;=&space;\begin{bmatrix}&space;x_{1,1}&space;&&space;x_{1,2}\\&space;x_{2,1}&space;&&space;x_{2,2}&space;\end{bmatrix}" title="A = \begin{bmatrix} x_{1,1} & x_{1,2}\\ x_{2,1} & x_{2,2} \end{bmatrix}" />
</p>
* <b>Tensors:</b> more than 2-D array. For example, <img src="https://latex.codecogs.com/svg.latex?\inline&space;A_{i,j,k}" title="A_{i,j,k}" />

One important operation on matrices is the transpose. The transpose of a matrix is the mirror image of the matrix across a diagonal line, called the main diagonal, running down and to the right, starting from its upper left corner. <img src="https://latex.codecogs.com/svg.latex?\inline&space;(A^T)_{i,j}&space;=&space;A_{j,i}" title="(A^T)_{i,j} = A_{j,i}" />

In Machine Learning, we allow addition of matrces and vectors where, <img src="https://latex.codecogs.com/svg.latex?\inline&space;C_{i,j}&space;=&space;A_{i,j}&space;&plus;&space;b{j}" title="C_{i,j} = A_{i,j} + b{j}" />
The implicit copying of b across each row is called broadcasting.

## Multiplying Matrices and Vectors

* <b>Matrix Multiplication:</b>
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?C_{m*p}&space;=&space;A_{m*n}B_{n*p}" title="C_{m*p} = A_{m*n}B_{n*p}" />
</p>
where, <img src="https://latex.codecogs.com/svg.latex?\inline&space;C_{i,j}&space;=&space;\sum&space;_{k}&space;A_{i,k}B_{k,j}" title="C_{i,j} = \sum _{k} A_{i,k}B_{k,j}" />.
* <b>Element wise product or Hadard product:</b>
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?C&space;=&space;A\odot&space;B" title="C = A\odot B" />
</p>
where, <img src="https://latex.codecogs.com/svg.latex?\inline&space;C_{i,j}&space;=&space;A_{i,j}*B_{i,j}" title="C_{i,j} = A_{i,j}*B_{i,j}" />.
* <b>Dot product:</b> Product of two vectors of same dimensionality.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?x\cdot&space;y&space;=&space;x^Ty" title="x\cdot y = x^Ty" />
</p>

Matrix multiplication is associative (<img src="https://latex.codecogs.com/svg.latex?\inline&space;A(BC)&space;=&space;(AB)C" title="A(BC) = (AB)C" />), distributive (<img src="https://latex.codecogs.com/svg.latex?\inline&space;A(B&plus;C)&space;=&space;AB&space;&plus;&space;AC" title="A(B+C) = AB + AC" />) but not commutative (<img src="https://latex.codecogs.com/svg.latex?\inline&space;AB&space;\neq&space;BA" title="AB \neq BA" />). However, the commutative property holds for vector product (<img src="https://latex.codecogs.com/svg.latex?\inline&space;x^Ty&space;=&space;y^Tx" title="x^Ty = y^Tx" />).

Also, note that <img src="https://latex.codecogs.com/svg.latex?\inline&space;(AB)^T&space;=&space;B^TA^T" title="(AB)^T = B^TA^T" />.

## Identity & Inverse Matrices

Linear algebra offers a powerful tool called matrix inversion, that allows us to analytically solve equation <img src="https://latex.codecogs.com/svg.latex?\inline&space;Ax&space;=&space;b" title="Ax = b" /> for many values of <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" />.
We denote the identity matrix that preserves n-dimensional vectors as <img src="https://latex.codecogs.com/svg.latex?\inline&space;I_{n}\in&space;\mathbb{R}^{n*n}" title="I_{n}\in \mathbb{R}^{n*n}" />, and
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?AA^{-1}=&space;I" title="AA^{-1}= I" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?\inline&space;A^{-1}" title="A^{-1}" /> is inverse of A. Also note that, <img src="https://latex.codecogs.com/svg.latex?\inline&space;I_{n}x&space;=&space;x" title="I_{n}x = x" />.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Ax&space;=&space;b" title="Ax = b" /> <br>
<img src="https://latex.codecogs.com/svg.latex?A^{-1}Ax&space;=&space;A^{-1}b" title="A^{-1}Ax = A^{-1}b" /> <br>
<img src="https://latex.codecogs.com/svg.latex?I_{n}x&space;=&space;A^{-1}b" title="I_{n}x = A^{-1}b" /> <br>
<img src="https://latex.codecogs.com/svg.latex?x&space;=&space;A^{-1}b" title="x = A^{-1}b" /><br>
</p>
Note that <img src="https://latex.codecogs.com/svg.latex?\inline&space;x&space;=&space;A^{-1}b" title="x = A^{-1}b" /> might or might not exist depending on the existence of inverse of <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" />.

## Linear Dependence and Span

Suppose <img src="https://latex.codecogs.com/svg.latex?\inline&space;Ax&space;=&space;b" title="Ax = b" /> has 2 solutions <img src="https://latex.codecogs.com/svg.latex?\inline&space;x,y" title="x,y" />, then <img src="https://latex.codecogs.com/svg.latex?\inline&space;z&space;=&space;\alpha&space;x&space;&plus;&space;(1-\alpha)y" title="z = \alpha x + (1-\alpha)y" /> is also a solution.

Formally, a linear combination of some set of vectors <img src="https://latex.codecogs.com/svg.latex?\inline&space;\begin{Bmatrix}&space;v^{(1)}&space;,...&space;,&space;v^{(n)}&space;\end{Bmatrix}" title="\begin{Bmatrix} v^{(1)} ,... , v^{(n)} \end{Bmatrix}" /> is given by multiplying each vector <img src="https://latex.codecogs.com/svg.latex?\inline&space;v^{(i)}" title="v^{(i)}" /> by a corresponding scalar coefficient and adding the results:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\sum&space;_{i}c_{i}v^{(i)}" title="\sum _{i}c_{i}v^{(i)}" />
</p>
The <b>span</b> of a set of vectors is the set of all points obtainable by linear combination of the original vectors. Determining whether <img src="https://latex.codecogs.com/svg.latex?\inline&space;Ax&space;=&space;b" title="Ax = b" /> has a solution amounts to testing whether <img src="https://latex.codecogs.com/svg.latex?\inline&space;b" title="b" /> is in the span of columns of <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" />, aka column space or range of <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" />
In order for the system <img src="https://latex.codecogs.com/svg.latex?\inline&space;Ax&space;=&space;b" title="Ax = b" /> to have a solution for all values of <img src="https://latex.codecogs.com/svg.latex?\inline&space;b&space;\in&space;\mathbb{R}^{m}" title="b \in \mathbb{R}^{m}" />, we therefore require that the column space of A be all of <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbb{R}^{m}" title="\mathbb{R}^{m}" />. If any point in <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbb{R}^{m}" title="\mathbb{R}^{m}" /> is excluded from th column space, that point is a potential value of <img src="https://latex.codecogs.com/svg.latex?\inline&space;b" title="b" /> that has no solution. The requirement that the column space of A be all of <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbb{R}^{m}" title="\mathbb{R}^{m}" />  implies immediately that <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> must have at least <img src="https://latex.codecogs.com/svg.latex?\inline&space;m" title="m" /> columns, i.e., <img src="https://latex.codecogs.com/svg.latex?\inline&space;n\geq&space;m" title="n\geq m" />. Having <img src="https://latex.codecogs.com/svg.latex?\inline&space;n\geq&space;m" title="n\geq m" /> is only a necessary condition for every point to have a solution, It is not however sufficient condition since, it is possible for some of the columns to be redundant.

A set of vectors is linearly independent if no vector in the set is a linear combination of other vectors. This means that for the column space of the matrix to encompass all of <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbb{R}^{m}" title="\mathbb{R}^{m}" />, the matrix must contain at least one set of <img src="https://latex.codecogs.com/svg.latex?\inline&space;m" title="m" /> linearly independent columns. This condition is both necessary and sufficient for <img src="https://latex.codecogs.com/svg.latex?\inline&space;Ax&space;=&space;b" title="Ax = b" /> to have a solution for every value of <img src="https://latex.codecogs.com/svg.latex?\inline&space;b" title="b" />. Note that the requirement is for a set to have exactly <img src="https://latex.codecogs.com/svg.latex?\inline&space;m" title="m" /> linear independent columns, not at least <img src="https://latex.codecogs.com/svg.latex?\inline&space;m" title="m" />.
In order for the matrix to have an inverse, we additionally need to ensure that <img src="https://latex.codecogs.com/svg.latex?\inline&space;Ax&space;=&space;b" title="Ax = b" /> has at most one solution for each value of <img src="https://latex.codecogs.com/svg.latex?\inline&space;b" title="b" />. To do so, we need to ensure that matrix has at most m columns. Otherwise, there is more than one way of parametrizing each solution.
Together, this means that the matrix must be a square, that is, we require that <img src="https://latex.codecogs.com/svg.latex?\inline&space;m&space;=&space;n" title="m = n" /> and that all of the columns must be linearly independent. A square matrix with linearly independent columns is known as <b>singular</b>. If <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> is not square or is square, but singular, it can still be possible to solve the equation. However, we cannot use the method of matrix inversion to find the solution.

## Norms

Norms are used to measure the size of a vector. Formally, the <img src="https://latex.codecogs.com/svg.latex?\inline&space;L^{p}" title="L^{p}" /> norm is given by:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left&space;\|&space;x&space;\right&space;\|_{p}&space;=&space;\left&space;(&space;\sum&space;_i\left&space;|&space;x_i&space;\right&space;|^{p}&space;\right&space;)^{\frac{1}{p}}" title="\left \| x \right \|_{p} = \left ( \sum _i\left | x_i \right |^{p} \right )^{\frac{1}{p}}" />
</p>
for <img src="https://latex.codecogs.com/svg.latex?\inline&space;p&space;\in&space;\mathbb{R}" title="p \in \mathbb{R}" />, <img src="https://latex.codecogs.com/svg.latex?\inline&space;p&space;\geq&space;1" title="p \geq 1" />.
Norm must follow 3 conditions:
* <img src="https://latex.codecogs.com/svg.latex?\inline&space;f(x)=0\Rightarrow&space;x=0" title="f(x)=0\Rightarrow x=0" />
* <img src="https://latex.codecogs.com/svg.latex?\inline&space;f(x&plus;y)&space;\leq&space;f(x)&space;&plus;&space;f(y)" title="f(x+y) \leq f(x) + f(y)" />
* <img src="https://latex.codecogs.com/gif.latex?\forall&space;\alpha&space;\in&space;\mathbb{R},&space;f(\alpha&space;x)&space;=&space;\left&space;|&space;\alpha&space;\right&space;|f(x)" title="\forall \alpha \in \mathbb{R}, f(\alpha x) = \left | \alpha \right |f(x)" />

The <img src="https://latex.codecogs.com/svg.latex?\inline&space;L^2" title="L^2" /> norm, with <img src="https://latex.codecogs.com/svg.latex?\inline&space;p=2" title="p=2" />, is known as the <b>Eucledian Norm</b> and is equal to <img src="https://latex.codecogs.com/svg.latex?\inline&space;(x^Tx)^{\frac{1}{2}}" title="(x^Tx)^{\frac{1}{2}}" />. The squared <img src="https://latex.codecogs.com/svg.latex?\inline&space;L^2" title="L^2" /> norm is more convenient to work with mathematically and computationally than the <img src="https://latex.codecogs.com/svg.latex?\inline&space;L^2" title="L^2" /> norm itself.
If data is close to the origin is of high importance, we use <img src="https://latex.codecogs.com/gif.latex?L^1" title="L^1" /> norm. The <img src="https://latex.codecogs.com/svg.latex?\inline&space;L^1" title="L^1" /> norm may be simplified as <img src="https://latex.codecogs.com/svg.latex?\inline&space;\left&space;\|&space;x&space;\right&space;\|_1&space;=&space;\sum&space;_i&space;\left&space;|x_i&space;\right&space;|" title="\left \| x \right \|_1 = \sum _i \left |x_i \right |" />.
We sometimes measure the size of the vector by counting its number of nonzero elements. Some authors refer to this function as the "<img src="https://latex.codecogs.com/svg.latex?\inline&space;L^0" title="L^0" /> norm", but this is incorrect terminology. The number of non-zero entries in a vector is not a norm, because scaling the vector by <img src="https://latex.codecogs.com/svg.latex?\inline&space;\alpha" title="\alpha" /> does not change the number of nonzero entries. The <img src="https://latex.codecogs.com/svg.latex?\inline&space;L^1" title="L^1" /> norm is often used as a substitute for the number of nonzero entries.
One other norm is the max norm or <img src="https://latex.codecogs.com/svg.latex?\inline&space;L^\infty" title="L^\infty" /> which can be simplifies to the absolute value of the element with the largest magnitude in the vector, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\left&space;\|&space;x&space;\right&space;\|_{\infty}&space;=&space;\max_{i}&space;\left&space;|&space;x_i&space;\right&space;|" title="\left \| x \right \|_{\infty} = \max_{i} \left | x_i \right |" />.
Sometimes, we also wish to measure the size of a matrix. In the context of deep learning, the most common way to do this is with the otherwise obscure <b>Frobenius norm</b>:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left&space;\|&space;A&space;\right&space;\|_F&space;=&space;\sqrt{\sum&space;_{i,j}A_{i,j}^2}" title="\left \| A \right \|_F = \sqrt{\sum _{i,j}A_{i,j}^2}" />
</p>
which is similar to <img src="https://latex.codecogs.com/svg.latex?\inline&space;L^2" title="L^2" /> norm of vectors, but for matrices.

Furthermore, the dot product of two vectors can be rewritten in terms of norms. Specifically,
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?x^Ty&space;=&space;\left&space;\|&space;x&space;\right&space;\|_2\left&space;\|&space;y&space;\right&space;\|_2\cos\theta" title="x^Ty = \left \| x \right \|_2\left \| y \right \|_2\cos\theta" />
</p>

## Special Kinds of matrices and Vectors

* <b>Diagonal Matrix:</b>
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?D_{ij}&space;=&space;\left\{\begin{matrix}&space;1&space;&,&space;i=j&space;\\&space;0&space;&,&space;otherwise&space;\end{matrix}\right." title="D_{ij} = \left\{\begin{matrix} 1 &, i=j \\ 0 &, otherwise \end{matrix}\right." />
</p>
We write <img src="https://latex.codecogs.com/svg.latex?\inline&space;diag(V)" title="diag(V)" /> to denote a square diagonal matrix whose diagonal entries are given by the entries of vector <img src="https://latex.codecogs.com/svg.latex?\inline&space;V" title="V" />. To compute <img src="https://latex.codecogs.com/svg.latex?\inline&space;diag(V)x" title="diag(V)x" />, we only need to scale each element <img src="https://latex.codecogs.com/svg.latex?\inline&space;x_i" title="x_i" /> by <img src="https://latex.codecogs.com/svg.latex?\inline&space;v_i" title="v_i" />. Not all diagonal matrices need be square. It is possible to construct a rectangular diagonal matrix.
* <b>Symmetric Matrix:</b> Matrix <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> is said to be symmetric if <img src="https://latex.codecogs.com/svg.latex?\inline&space;A^T=A" title="A^T=A" />, or <img src="https://latex.codecogs.com/svg.latex?\inline&space;A_{i,j}&space;=&space;A_{j,i}" title="A_{i,j} = A_{j,i}" />.
* <b>Unit Vector:</b> A vector with unit norm. <img src="https://latex.codecogs.com/svg.latex?\inline&space;\left&space;\|&space;x&space;\right&space;\|_2&space;=&space;1" title="\left \| x \right \|_2 = 1" />.
* <b>Orthogonal Vectors:</b> A vector <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;y" title="y" /> are said to be orthogonal if <img src="https://latex.codecogs.com/svg.latex?\inline&space;x^Ty&space;=&space;0" title="x^Ty = 0" />. If both the vectors have non-zero norm, this means that they are at a 90 degree angle to each other. In <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbb{R}^n" title="\mathbb{R}^n" />, at most <img src="https://latex.codecogs.com/svg.latex?\inline&space;n" title="n" /> vectors may be mutually orthogonal to each other with nonzero norm. If the vectors are not only orthogonal but also have a unit norm, we call them <b>Orthonormal</b>.
* <b>Orthogonal Matrix:</b> A square matrix whose rows are mutually orthonormal and whose columns are mutually orthonormal. This would mean that <img src="https://latex.codecogs.com/svg.latex?\inline&space;A^TA&space;=&space;AA^T&space;=&space;I" title="A^TA = AA^T = I" />. This would also imply that <img src="https://latex.codecogs.com/svg.latex?\inline&space;A^{-1}=A^T" title="A^{-1}=A^T" />.

## Eigendecomposition

This is a matrix decomposition method in which we decompose a matrix into a set of eigenvectors and eigenvalues.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?A\upsilon&space;=&space;\lambda&space;\upsilon" title="A\upsilon = \lambda \upsilon" />
</p>
Here, <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> is a square matrix, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\upsilon" title="\upsilon" /> is the eigenvector and <img src="https://latex.codecogs.com/svg.latex?\inline&space;\lambda" title="\lambda" /> is the eigenvalue.
If <img src="https://latex.codecogs.com/svg.latex?\inline&space;\upsilon" title="\upsilon" /> is an eigen vector, so is any rescaled vector <img src="https://latex.codecogs.com/svg.latex?\inline&space;s\upsilon" title="s\upsilon" /> where <img src="https://latex.codecogs.com/svg.latex?\inline&space;s&space;\neq&space;0" title="s \neq 0" />. Hence, we only consider unit vectors. Suppose that a matrix <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> has <img src="https://latex.codecogs.com/svg.latex?\inline&space;n" title="n" /> linearly independent eigenvectors, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\begin{Bmatrix}&space;\upsilon&space;^{(1)},&space;...,&space;\upsilon^{(n)}&space;\end{Bmatrix}" title="\begin{Bmatrix} \upsilon ^{(1)}, ..., \upsilon^{(n)} \end{Bmatrix}" />, with corresponding eigenvalues <img src="https://latex.codecogs.com/svg.latex?\inline&space;\begin{Bmatrix}&space;\lambda&space;^{(1)},&space;...,&space;\lambda&space;^{(n)}&space;\end{Bmatrix}" title="\begin{Bmatrix} \lambda ^{(1)}, ..., \lambda ^{(n)} \end{Bmatrix}" />. We may concatenate all of the eigenvectors to form a matrix <img src="https://latex.codecogs.com/svg.latex?\inline&space;V" title="V" /> with one eigenvector per column: <img src="https://latex.codecogs.com/svg.latex?\inline&space;V&space;=&space;\begin{bmatrix}&space;\upsilon&space;^{(1)},&space;...,&space;\upsilon^{(n)}&space;\end{bmatrix}" title="V = \begin{bmatrix} \upsilon ^{(1)}, ..., \upsilon^{(n)} \end{bmatrix}" />. Likewise, we concatenate the eigenvalues to form a vector <img src="https://latex.codecogs.com/svg.latex?\inline&space;\boldsymbol{\lambda}=&space;\begin{bmatrix}&space;\lambda&space;\upsilon&space;^{(1)},&space;...,&space;\lambda&space;\upsilon^{(n)}&space;\end{bmatrix}" title="\boldsymbol{\lambda}= \begin{bmatrix} \lambda \upsilon ^{(1)}, ..., \lambda \upsilon^{(n)} \end{bmatrix}" />. The <b>eigendecomposition</b> of A is then given by <img src="https://latex.codecogs.com/svg.latex?\inline&space;A&space;=&space;Vdiag(\boldsymbol{\lambda})V^{-1}" title="A = Vdiag(\boldsymbol{\lambda})V^{-1}" />.
We have seen that constructing matrices with specific eigenvalues and eigenvectors allows us to stretch space in desired directions. However, we often want to decompose matrices into their eigenvalues and eigenvectors.
Not every matrix can be decomposed into eigenvalues and eigenvectors. However, every real symmetric matrix can be decomposed into an expression using only real-valued eigenvectors and eigenvalues:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?A&space;=&space;Q&space;\wedge&space;Q^T" title="A = Q \wedge Q^T" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?\inline&space;Q" title="Q" /> is an orthogonal matrix composed of eigenvectors of <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" />, and <img src="https://latex.codecogs.com/svg.latex?\inline&space;\wedge" title="\wedge" /> is a diagonal matrix.
While any real symmetric matrix <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> is guaranteed to have an eigendecomposition, the eigendecomposition may not be unique. If any two or more eigenvectors share the same eigenvalue, then any set of orthogonal vectors lying in their span are also eigenvectors with that eigenvalue. By convention, we usually sort the entries of <img src="https://latex.codecogs.com/svg.latex?\inline&space;\wedge" title="\wedge" /> in descending order. Under this convention, the eigendecomposition is unique only if all of the eigenvalues are unique.

The eigendecomposition of a matrix tells us many useful facts about the matrix. The matrix is singular if and only if any of the eigenvalues are zero.
The eigendecomposition of a real symmetric matrix can also be used to optimize quadratic expressions of the form <img src="https://latex.codecogs.com/svg.latex?\inline&space;f(x)&space;=&space;x^TAx" title="f(x) = x^TAx" /> subject to <img src="https://latex.codecogs.com/svg.latex?\inline&space;\left&space;\|&space;x&space;\right&space;\|_2&space;=&space;1" title="\left \| x \right \|_2 = 1" />. Whenever, <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> is equal to an eigenvector of <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" />, <img src="https://latex.codecogs.com/svg.latex?\inline&space;f" title="f" /> taks on the value of corresponding eigenvalue, since:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?A&space;=&space;Q&space;\wedge&space;Q^T&space;\Leftrightarrow&space;\wedge&space;=&space;Q^TAQ" title="A = Q \wedge Q^T \Leftrightarrow \wedge = Q^TAQ" />
</p>
since <img src="https://latex.codecogs.com/svg.latex?\inline&space;Q" title="Q" /> is orthogonal and <img src="https://latex.codecogs.com/svg.latex?\inline&space;Q^{-1}&space;=&space;Q^T" title="Q^{-1} = Q^T" />.
A matrix whose eigenvalues are all positive is called <b>positive definite</b>. A matrix whose eigenvalues are all positive or zero-valued is called <b>positive semidefinite</b>. Likewise, if all eigenvalues are negative, the matrix is <b>negative definite</b>, and if all eigenvalues are negative or zero valued, it is <b>negative semidefinite</b>.
Positive semidefinite matrices are interesting for two reasons:
* They guarantee that <img src="https://latex.codecogs.com/svg.latex?\inline&space;\forall&space;x,&space;x^TAx&space;\geq&space;0" title="\forall x, x^TAx \geq 0" />
* They also guarantee that <img src="https://latex.codecogs.com/svg.latex?\inline&space;x^TAx=0&space;\Rightarrow&space;x=0" title="x^TAx=0 \Rightarrow x=0" />.

## Singular Value Decomposition

This is another method to factorize a matrix, into singular vectors and singular values. This is more generally applicable. Every real matrix has a SVD, but the same is not true for eigen decomposition. Here, we write <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> in terms of three matrices:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?A=UDV^T" title="A=UDV^T" />
</p>
Here, <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> is a <img src="https://latex.codecogs.com/svg.latex?\inline&space;m&space;*&space;n" title="m * n" /> matrix, <img src="https://latex.codecogs.com/svg.latex?\inline&space;U" title="U" /> is defined to be a <img src="https://latex.codecogs.com/svg.latex?\inline&space;m*m" title="m*m" /> matrix, <img src="https://latex.codecogs.com/svg.latex?\inline&space;D" title="D" /> to be a <img src="https://latex.codecogs.com/svg.latex?\inline&space;m*n" title="m*n" /> matrix, and <img src="https://latex.codecogs.com/svg.latex?\inline&space;V" title="V" /> to be a <img src="https://latex.codecogs.com/svg.latex?\inline&space;n*n" title="n*n" /> matrix.
The matrices <img src="https://latex.codecogs.com/svg.latex?\inline&space;U" title="U" />  and <img src="https://latex.codecogs.com/svg.latex?\inline&space;V" title="V" /> are both defined to be orthogonal matrices. The matrix <img src="https://latex.codecogs.com/svg.latex?\inline&space;D" title="D" /> is defined to be a diagonal matrix (not necessarily a square). The elements across the diagonal of <img src="https://latex.codecogs.com/svg.latex?\inline&space;D" title="D" /> are known as the <b>singular values</b> of the matrix <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" />. The columns of <img src="https://latex.codecogs.com/svg.latex?\inline&space;U" title="U" /> are known as <b>left-singular vectors</b>. The columns of <img src="https://latex.codecogs.com/svg.latex?\inline&space;V" title="V" /> are known as the <b>right-singular vectors</b>.

Furthermore, the left-singular vectors of <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> are the eigenvectors of <img src="https://latex.codecogs.com/svg.latex?\inline&space;AA^T" title="AA^T" />, whereas the right-singular vectors of <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> are the eigenvectors of <img src="https://latex.codecogs.com/svg.latex?\inline&space;A^TA" title="A^TA" />.

## Moore Penrose Pseudoinverse

Matrix inversion is not defined for matrices that are not square. Suppose <img src="https://latex.codecogs.com/svg.latex?\inline&space;A&space;\in&space;\mathbb{R}^{m*n}" title="A \in \mathbb{R}^{m*n}" />, if:
* <img src="https://latex.codecogs.com/svg.latex?\inline&space;m>n" title="m>n" />, no solutions of <img src="https://latex.codecogs.com/svg.latex?\inline&space;Ax&space;=&space;y" title="Ax = y" />
* <img src="https://latex.codecogs.com/svg.latex?\inline&space;n>m" title="n>m" />, multiple solutions of <img src="https://latex.codecogs.com/svg.latex?\inline&space;Ax&space;=&space;y" title="Ax = y" />

The <b>Moore-Penrose pseudoinverse</b> of <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> is defined as a matrix <img src="https://latex.codecogs.com/svg.latex?\inline&space;A^{&plus;}&space;=&space;VD^{&plus;}U^{T}" title="A^{+} = VD^{+}U^{T}" />, where <img src="https://latex.codecogs.com/svg.latex?\inline&space;U" title="U" />, <img src="https://latex.codecogs.com/svg.latex?\inline&space;D" title="D" />, and <img src="https://latex.codecogs.com/svg.latex?\inline&space;V" title="V" /> are the singular value decomposition of <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" />. <img src="https://latex.codecogs.com/svg.latex?\inline&space;D^{&plus;}" title="D^{+}" /> is obtained by taking the reciprocal of its non-zero elements, then taking the transpose of the resulting matrix.
When <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> has more columns than rows, then solving a linear equation using the pseudoinverse provides one of the many possible solutions. Specifically, it provides the solution <img src="https://latex.codecogs.com/svg.latex?\inline&space;x&space;=&space;A^{&plus;}y" title="x = A^{+}y" /> with minimal Eucledian norm <img src="https://latex.codecogs.com/svg.latex?\inline&space;\left&space;\|&space;x&space;\right&space;\|_2" title="\left \| x \right \|_2" /> among all possible solutions.
When <img src="https://latex.codecogs.com/svg.latex?\inline&space;A" title="A" /> has more rows than columns, it is possible for there to be no solution. In this case, using the pseudoinverse gives us the <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> for which <img src="https://latex.codecogs.com/svg.latex?\inline&space;Ax" title="Ax" /> is as close as possible to <img src="https://latex.codecogs.com/svg.latex?\inline&space;y" title="y" /> in terms of Eucledian norm <img src="https://latex.codecogs.com/svg.latex?\inline&space;\left&space;\|&space;Ax-y&space;\right&space;\|_2" title="\left \| Ax-y \right \|_2" />

## The Trace Operator

The trace operator gives the sum of all the diagonal entries of a matrix:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Tr(A)&space;=&space;\sum&space;_iA_{i,i}" title="Tr(A) = \sum _iA_{i,i}" />
</p>
The trace operator provides an alternative way of writing the Frobenius norm of a matrix: <img src="https://latex.codecogs.com/svg.latex?\inline&space;\left&space;\|&space;A&space;\right&space;\|_F&space;=&space;\sqrt{Tr(AA^T)}" title="\left \| A \right \|_F = \sqrt{Tr(AA^T)}" />.
Furthermore, the trace operator is invariant to the transpose operator: <img src="https://latex.codecogs.com/svg.latex?\inline&space;Tr(A)&space;=&space;Tr(A^T)" title="Tr(A) = Tr(A^T)" />.
Also, the trace operator is invariant to cyclic permutations, even if the output is of a different shape: <img src="https://latex.codecogs.com/svg.latex?\inline&space;Tr(ABC)&space;=&space;Tr(BCA)&space;=&space;Tr(CAB)" title="Tr(ABC) = Tr(BCA) = Tr(CAB)" />.

## The Determinant

The determinant of a square matrix, denoted by <img src="https://latex.codecogs.com/svg.latex?\inline&space;det(A)" title="det(A)" />, is a function mapping matrices to real scalars. The determinant is equal to the product of all the eigenvalues of the matrix.
The absolute value of the determinant can be thought of as a measure of how much multiplication by the matrix expands or contracts space.
If the determinant is 0, the space is contracted completely along at least one dimension, causing it to lose all of its volume. If the determinant is 1, then the transformation preserves volume.

## Principal Component Analysis

Principal component analysis, or PCA, is a technique widely used for applications such as dimensionality reduction, lossy data compression, feature extraction and data visualization. PCA can be defined as the orthogonal projection of the data onto a lower dimensional linear space, known as the principal subspace, such that the variance of the projected data is maximized. These lower dimensional linear space is obtained from the eigenvectors derived from eigendecomposition of the data.
Equivalently, it can also be defined as the linear projection that minimizer the average projection cost, defined as the mean squared distance between the data points and their projections.
