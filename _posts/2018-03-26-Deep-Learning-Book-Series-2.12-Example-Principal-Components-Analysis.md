---
bg: "river.jpg"
layout: post
mathjax: true
title: Deep Learning Book Series · 2.12 Example   Principal Components Analysis
crawlertitle: "Introduction to Principal Components Analysis (PCA) using Python/Numpy examples and drawings"
categories: posts
tags: ['linear-algebra', 'python', 'numpy', 'deep-learning-book']
author: hadrienj
jupyter: https://github.com/hadrienj/deepLearningBook-Notes/blob/master/2.12%20Example%20-%20Principal%20Components%20Analysis/2.12%20Example%20-%20Principal%20Components%20Analysis.ipynb
date: 2018-03-26 18:00:00
excerpt: This post on linear algebra is about Principal Components Analysis (PCA). We will use Python/Numpy/Matplotlib to get a better intuition and understanding of this important data analysis tool!
excerpt-image: '<img src="../../assets/images/2.12/principal-component-analysis-variance-explained.png" width="400" alt="Representation of the variance explained across directions" title="Maximizing the variance">
    <em>Projection of the data point: the line direction is the one with the largest variance</em>'
deep-learning-book-toc: true
---

*Last update: Jan. 2020*

# Introduction

This is the last chapter of this series on linear algebra! It is about Principal Components Analysis (PCA). We will use some knowledge that we acquired along the preceding chapters to understand this important data analysis tool! Feel free to check out the preceding chapters!

{% include mailchimp.html %}

# 2.12 Example - Principal Components Analysis

Dimensions are a crucial topic in data science. The dimensions are all the features of the dataset. For instance, if you are looking at a dataset containing pieces of music, dimensions could be the genre, the length of the piece, the number of instruments, the presence of a singer, etc. You can imagine all these dimensions as different columns. When there are only two dimensions, it is very convenient to plot: you can use the $x$- and $y$-axis. Add color and you can represent a third dimension. It is similar if you have tens or hundereds of dimensions, it will just be harder to visualize it.

When you have that many dimensions, it happens that some of them are correlated. For instance, we can reasonably think that the genre of a piece of music will correlate with the instruments present in the piece. One way to reduce dimensionality is simply to keep only some of them. The problem is that you loose good information. It would be nice to have a way to reduce these dimensions while keeping important informations present in the dataset.

The aim of *Principal Components Analysis* (PCA) is generaly to reduce the number of dimensions of a dataset. PCA provides us with a new set of dimensions, the *Principal Components* (PC). They are ordered: the first PC is the dimension associated with the largest variance. In addition, PC's are orthogonal. Remember that orthogonal vectors means that their dot product is equal to $0$ (see [2.6](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.6-Special-Kinds-of-Matrices-and-Vectors/)). This means that each PC is decorelated to the preceding one. You can choose to keep only the first few PC's, knowing that each PC is a linear combination of the data features. For instance, one PC could be a linear combination of the length of the muscial piece and the number of instruments playing.

### Example 1.

Unit vectors are an example of orthogonal vectors:

<img src="../../assets/images/2.12/orthogonal-vectors.png" width="200" alt="Example of orthogonal vectors" title="Orthogonal vectors">
<em>Orthogonal vectors</em>

## Describing the problem

The problem can be expressed as finding a function that converts a set of data points from $\mathbb{R}^n$ to $\mathbb{R}^l$: we want to change the number of dimensions of our dataset from $n$ to $l$. If $l<n$, the new dataset will be compressed because its number of features decreased. We also need a function that can decode back the transformed dataset into the initial one:

<img src="../../assets/images/2.12/principal-components-analysis-PCA-change-coordinates.png" width="80%" alt="Principal components analysis (PCA)" title="Principal components analysis (PCA)">
<em>Principal components analysis as a change of coordinate system</em>

The first step is to understand the shape of the data. $x^{(i)}$ is one data point containing $n$ dimensions. Let's have $m$ data points organized as column vectors (one column per point):

<div>
$
\bs{x}=
\begin{bmatrix}
    x^{(1)} & x^{(2)} & \cdots & x^{(m)}
\end{bmatrix}
$
</div>

If we deploy the $n$ dimensions of our data points we will have:

<div>
$
\bs{x}=
\begin{bmatrix}
    x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(m)} \\\
    x_2^{(1)} & x_2^{(2)} & \cdots & x_2^{(m)} \\\
    \cdots & \cdots & \cdots & \cdots \\\
    x_n^{(1)} & x_n^{(2)} & \cdots & x_n^{(m)}
\end{bmatrix}
$
</div>

We can also write:

<div>
$$
\bs{x}=
\begin{bmatrix}
    x_1\\\\
    x_2\\\\
    \cdots\\\\
    x_n
\end{bmatrix}
$$
</div>

$c$ will have the shape:

<div>
$$
\bs{c}=
\begin{bmatrix}
    c_1\\\\
    c_2\\\\
    \cdots\\\\
    c_l
\end{bmatrix}
$$
</div>

{% include essential-math-ribbon.html %}

## Adding some constraints: the decoding function

The encoding function $f(\bs{x})$ transforms $\bs{x}$ into $\bs{c}$ and the decoding function transforms back $\bs{c}$ into an approximation of $\bs{x}$. To keep things simple, PCA will respect some constraints:

### Constraint 1.

The decoding function has to be a simple matrix multiplication:

<div>
$$
g(\bs{c})=\bs{Dc}
$$
</div>

By applying the matrix $\bs{D}$ to the dataset from the new coordinates system we should get back to the initial coordinate system.

### Constraint 2.

The columns of $\bs{D}$ must be orthogonal (see [2.6](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.6-Special-Kinds-of-Matrices-and-Vectors/)).

### Constraint 3.

The columns of $\bs{D}$ must have unit norm (see [2.6](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.6-Special-Kinds-of-Matrices-and-Vectors/)).

## Finding the encoding function

Important: For now we will consider only **one data point**. Thus we will have the following dimensions for these matrices (note that $\bs{x}$ and $\bs{c}$ are column vectors):

<img src="../../assets/images/2.12/principal-components-analysis-PCA-decoding-function.png" width="250" alt="Principal components analysis (PCA) - the decoding function" title="The decoding function">
<em>The decoding function</em>

We want a decoding function which is a simple matrix multiplication. For that reason, we have $g(\bs{c})=\bs{Dc}$. We will then find the encoding function from the decoding function. We want to minimize the error between the decoded data point and the actual data point. With our previous notation, this means reducing the distance between $\bs{x}$ and $g(\bs{c})$. As an indicator of this distance, we will use the squared $L^2$ norm (see [2.5](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.5-Norms/)):

<div>
$$
\norm{\bs{x} - g(\bs{c})}_2^2
$$
</div>

This is what we want to minimize. Let's call $\bs{c}^*$ the optimal $\bs{c}$. Mathematically it can be written:

<div>
$$
\bs{c}^* = \underset{c}{\arg\min} \norm{\bs{x} - g(\bs{c})}_2^2
$$
</div>

This means that we want to find the values of the vector $\bs{c}$ such that $\norm{\bs{x} - g(\bs{c})}_2^2$ is as small as possible.

If you have a look back to [2.5](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.5-Norms/) you can see that the squared $L^2$ norm can be expressed as:

<div>
$$
\norm{\bs{y}}_2^2 = \bs{y}^\text{T}\bs{y}
$$
</div>

We have named the variable $\bs{y}$ to avoid confusion with our $\bs{x}$. Here $\bs{y}=\bs{x} - g(\bs{c})$

Thus the equation that we want to minimize becomes:

<div>
$$
(\bs{x} - g(\bs{c}))^\text{T}(\bs{x} - g(\bs{c}))
$$
</div>

Since the transpose respects addition we have:

<div>
$$
(\bs{x}^\text{T} - g(\bs{c})^\text{T})(\bs{x} - g(\bs{c}))
$$
</div>

By the distributive property (see [2.2](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.2-Multiplying-Matrices-and-Vectors/)) we can develop:

<div>
$$
\bs{x^\text{T}x} - \bs{x}^\text{T}g(\bs{c}) -  g(\bs{c})^\text{T}\bs{x} + g(\bs{c})^\text{T}g(\bs{c})
$$
</div>

The commutative property (see [2.2](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.2-Multiplying-Matrices-and-Vectors/)) tells us that $
\bs{x^\text{T}y} = \bs{y^\text{T}x}
$. Since the result of $g(\bs{c})^\text{T}\bs{x}$ is a scalar, we have $
g(\bs{c})^\text{T}\bs{x} = \bs{x}^\text{T}g(\bs{c})
$. So the equation becomes:

<div>
$$
\bs{x^\text{T}x} -\bs{x}^\text{T}g(\bs{c}) -\bs{x}^\text{T}g(\bs{c}) + g(\bs{c})^\text{T}g(\bs{c})\\\\
= \bs{x^\text{T}x} -2\bs{x}^\text{T}g(\bs{c}) + g(\bs{c})^\text{T}g(\bs{c})
$$
</div>

The first term $\bs{x^\text{T}x}$ does not depends on $\bs{c}$ and since we want to minimize the function according to $\bs{c}$ we can just get off this term. We simplify to:

<div>
$$
\bs{c}^* = \underset{c}{\arg\min} -2\bs{x}^\text{T}g(\bs{c}) + g(\bs{c})^\text{T}g(\bs{c})
$$
</div>

Since $g(\bs{c})=\bs{Dc}$:

<div>
$$
\bs{c}^* = \underset{c}{\arg\min} -2\bs{x}^\text{T}\bs{Dc} + (\bs{Dc})^\text{T}\bs{Dc}
$$
</div>

With $(\bs{Dc})^\text{T}=\bs{c}^\text{T}\bs{D}^\text{T}$ (see [2.2](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.2-Multiplying-Matrices-and-Vectors/)), we have:

<div>
$$
\bs{c}^* = \underset{c}{\arg\min} -2\bs{x}^\text{T}\bs{Dc} + \bs{c}^\text{T}\bs{D}^\text{T}\bs{Dc}
$$
</div>

As we saw in [2.6](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.6-Special-Kinds-of-Matrices-and-Vectors/), $\bs{D}^\text{T}\bs{D}=\bs{I}_l$ because $\bs{D}$ is orthogonal (actually, it is [semi-orthogonal](https://en.wikipedia.org/wiki/Semi-orthogonal_matrix) if $n \neq l$) and have unit norm columns. We can replace in the equation:

<div>
$$
\bs{c}^* = \underset{c}{\arg\min} -2\bs{x}^\text{T}\bs{Dc} + \bs{c}^\text{T}\bs{I}_l\bs{c}
$$
</div>

<div>
$$
\bs{c}^* = \underset{c}{\arg\min} -2\bs{x}^\text{T}\bs{Dc} + \bs{c}^\text{T}\bs{c}
$$
</div>

### Minimizing the function

So far so good! Now the goal is to find the minimum of the function $- 2\bs{x}^\text{T}\bs{Dc} + \bs{c}^\text{T}\bs{c}$. One widely used way of doing that is to use the **gradient descent** algorithm. It is not the focus of this chapter but we will say a word about it (see [4.3](http://www.deeplearningbook.org/contents/numerical.html) of the Deep Learning Book for more details). The main idea is that the sign of the derivative of the function at a specific value of $x$ tells you if you need to increase or decrease $x$ to reach the minimum. When the slope is near $0$, the minimum should have been reached.

<img src="../../assets/images/2.12/gradient-descent.png" width="400" alt="Mechanism of the gradient descent algorithm" title="Mechanism of the gradient descent algorithm">
<em>Gradient descent</em>

However, functions with local minima can trouble the descent:

<img src="../../assets/images/2.12/gradient-descent-local-minima.png" width="400" alt="Gradient descent in the case of local minimum" title="Gradient descent">
<em>Gradient descent can get stuck in local minima</em>

These examples are in 2 dimensions but the principle stands for higher dimensional functions. The gradient is a vector containing the partial derivatives of all dimensions. Its mathematical notation is $\nabla_xf(\bs{x})$.

### Calculating the gradient of the function

Here we want to minimize through each dimension of $\bs{c}$. We are looking for a slope of $0$. The equation is:

<div>
$$
\nabla_c(-2\bs{x}^\text{T}\bs{Dc} + \bs{c}^\text{T}\bs{c})=0
$$
</div>

Let's take these terms separately to calculate the derivative according to $\bs{c}$.

<div>
$$
\frac{d(-2\bs{x}^\text{T}\bs{Dc})}{d\bs{c}} = -2\bs{x}^\text{T}\bs{D}
$$
</div>

The second term is $\bs{c}^\text{T}\bs{c}$. We can develop the vector $\bs{c}$ and calculate the derivative for each element:

<div>
$
\begin{aligned}
\frac{d(\bs{c}^\text{T}\bs{c})}{d\bs{c}} &=
\left(\frac{d(\bs{c}_1^2 + \bs{c}_2^2 + \cdots + \bs{c}_l^2)}{d\bs{c}_1},
\frac{d(\bs{c}_1^2 + \bs{c}_2^2 + \cdots + \bs{c}_l^2)}{d\bs{c}_2},
\cdots,
\frac{d(\bs{c}_1^2 + \bs{c}_2^2 + \cdots + \bs{c}_l^2)}{d\bs{c}_l}\right) \\\
&=(2\bs{c}_1, 2\bs{c}_2, \cdots, 2\bs{c}_l) \\\
&=2(\bs{c}_1, \bs{c}_2, \cdots, \bs{c}_l) \\\
&=2\bs{c}
\end{aligned}
$
</div>

So we can progress in our derivatives:

<div>
$$
\nabla_c(-2\bs{x}^\text{T}\bs{Dc} + \bs{c}^\text{T}\bs{c})=0\\\\
-2\bs{x}^\text{T}\bs{D} + 2\bs{c}=0\\\\
\bs{c}=\bs{x}^\text{T}\bs{D}
$$
</div>

Keep in mind the dimensions of the matrix $\bs{D}$ and the vector $\bs{x}$. We want $\bs{c}$ to be a column vector of shape (l, 1), so we need to transpose $\bs{x}^\text{T}\bs{D}$.

<img src="../../assets/images/2.12/principal-components-analysis-PCA-transpose-matrix-vectors.png" width="250" alt="Checking the dimension of the encoding" title="Checking the dimensions of the encoding">
<em>Dimensions of the matrix/vector dot product</em>

We have:

<div>
$$
(\bs{x}^\text{T}\bs{D})^{\text{T}} = \bs{D}^\text{T}\bs{x}
$$
</div>

So we have:

<div>
$$
\bs{c}=\bs{D}^\text{T}\bs{x}
$$
</div>

Great! We found the encoding function! Here are its dimensions:

<img src="../../assets/images/2.12/principal-components-analysis-PCA-encoding-function.png" width="250" alt="Expression of the encoding function" title="The encoding function">
<em>The encoding function</em>

To go back from $\bs{c}$ to $\bs{x}$ we use $g(\bs{c})=\bs{Dc}$:

<div>
$$
r(\bs{x}) = g(f(\bs{x})=\bs{D}\bs{D}^\text{T}\bs{x}
$$
</div>

<img src="../../assets/images/2.12/principal-components-analysis-PCA-reconstruction-function.png" width="300" alt="Expression of the reconstruction function" title="The reconstruction function">
<em>The reconstruction function</em>

## Finding $\bs{D}$

The next step is to find the matrix $\bs{D}$. Recall that the purpose of the PCA is to change the coordinate system in order to maximize the variance along the first dimensions of the projected space. This is equivalent to minimizing the error between data points and their reconstruction ([cf here](https://stats.stackexchange.com/questions/32174/pca-objective-function-what-is-the-connection-between-maximizing-variance-and-m)). See bellow the covariance matrix to have more details.

<span class='pquote'>
    Maximizing the variance corresponds to minimizing the error of the reconstruction.
</span>

### The Frobenius norm

Since we have to take all points into account (the same matrix $\bs{D}$ will be used for all points) we will use the Frobenius norm of the errors (see [2.5](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.5-Norms/)) which is the equivalent of the $L^2$ norm for matrices. Here is the formula of the Frobenius norm:

<div>
$$
\norm{\bs{A}}_F=\sqrt{\sum_{i,j}A^2_{i,j}}
$$
</div>

It is like if you unroll the matrix to end up with a one dimensional vector and that you take the $L^2$ norm of this vector.

We will call $\bs{D}^*$ the optimal $\bs{D}$ (in the sense that the error is as small as possible). We have:

<div>
$$
\bs{D}^* = \underset{\bs{D}}{\arg\min} \sqrt{\sum_{i,j}(x_j^{(i)}-r(\bs{x}^{(i)})_j})^2
$$
</div>

With the constraint that $\bs{D}^\text{T}\bs{D}=\bs{I}_l$ because we have chosen the constraint of having the columns of $\bs{D}$ orthogonal.




### The first principal component

We will start to find only the first principal component (PC). For that reason, we will have $l=1$. So the matrix $\bs{D}$ will have the shape $(n \times 1)$: it is a simple column vector. Since it is a vector we will call it $\bs{d}$:

<img src="../../assets/images/2.12/first-principal-component.png" width="100" alt="Dimension of the first principal component" title="The first principal component">
<em>The first principal component</em>

We can therefore remove the sum over $j$ and the square root since we will take the squared $L^2$ norm:

<div>
$$
\bs{d}^* = \underset{\bs{d}}{\arg\min} \sum_{i}\norm{(\bs{x}^{(i)}-r(\bs{x}^{(i)}))}_2^2
$$
</div>


We have also seen that:

<div>
$$
r(\bs{x})=\bs{D}\bs{D}^\text{T}\bs{x}
$$
</div>

Since we are looking only for the first PC:

<div>
$$
r(\bs{x})=\bs{d}\bs{d}^\text{T}\bs{x}
$$
</div>

We can plug $r(\bs{x})$ into the equation:

<div>
$$
\bs{d}^* = \underset{\bs{d}}{\arg\min} \sum_{i}\norm{\bs{x}^{(i)}-\bs{dd}^\text{T}\bs{x}^{(i)}}_2^2
$$
</div>

Because of the constraint 3. (the columns of $\bs{D}$ have unit norms) we have $\norm{\bs{d}}_2 = 1$. $\bs{d}$ is one of the columns of $\bs{D}$ and thus has a unit norm.


Instead of using the sum along the $m$ data points $\bs{x}$ we can have the matrix $\bs{X}$ which gather all the observations:

<div>
$
\bs{X} = \begin{bmatrix}
    \bs{x}^{(1)\text{T}} \\\
    \bs{x}^{(2)\text{T}} \\\
    \cdots \\\
    \bs{x}^{(m)\text{T}}
\end{bmatrix}=
\begin{bmatrix}
    \bs{x}_1^{(1)} & \bs{x}_2^{(1)} & \cdots & \bs{x}_n^{(1)} \\\
    \bs{x}_1^{(2)} & \bs{x}_2^{(2)} & \cdots & \bs{x}_n^{(2)} \\\
    \cdots & \cdots & \cdots & \cdots \\\
    \bs{x}_0^{(m)} & \bs{x}_1^{(m)} & \cdots & \bs{x}_n^{(m)}
\end{bmatrix}
$
</div>

We want $\bs{x}^{(i)\text{T}}$ instead of $\bs{x}^{(i)}$ in our expression of $\bs{d}^*$. We can transpose the content of the norm:


<div>
$
\begin{aligned}
\bs{d}^* &= \underset{\bs{d}}{\arg\min} \sum_{i}\norm{(\bs{x}^{(i)}-\bs{dd}^\text{T}\bs{x}^{(i)})^\text{T}}_2^2 \\\
&= \underset{\bs{d}}{\arg\min} \sum_i\norm{\bs{x}^{(i)\text{T}}-\bs{x}^{(i)\text{T}}\bs{dd}^\text{T}}_2^2 \\\
\end{aligned}
$
</div>

and

<div>
$$
\bs{d}^* = \underset{\bs{d}}{\arg\min} \norm{\bs{X}-\bs{X}\bs{dd}^\text{T}}_\text{F}^2
$$
</div>

with the constraint that $\bs{d}^\text{T}\bs{d}=1$.




### Using the Trace operator

We will now use the Trace operator (see [2.10](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.10-The-Trace-Operator/)) to simplify the equation to minimize. Recall that:

<div>
$$
\norm{\bs{A}}_F=\sqrt{\Tr({\bs{AA}^T})}
$$
</div>

So here $\bs{A}=\bs{X}-\bs{X}\bs{dd}^\text{T}$. So we have:

<div>
$$
\bs{d}^* = \underset{\bs{d}}{\arg\min} \Tr{((\bs{X}-\bs{Xdd}^\text{T})}(\bs{X}-\bs{Xdd}^\text{T})^\text{T})
$$
</div>

Since we can cycle the order of the matrices in a Trace (see [2.10](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.10-The-Trace-Operator/)) we can write:

<div>
$
\begin{aligned}
\bs{d}^* &= \argmin{d} \Tr{((\bs{X}-\bs{Xdd}^\text{T})^\text{T}}(\bs{X}-\bs{Xdd}^\text{T})) \\\
&=\argmin{d} \Tr{((\bs{X}^\text{T}-(\bs{Xdd}^\text{T})^\text{T})}(\bs{X}-\bs{Xdd}^\text{T}))
\end{aligned}
$
</div>

And $(\bs{Xdd}^\text{T})^\text{T}=(\bs{d}^\text{T})^\text{T}\bs{d}^\text{T}\bs{X}^\text{T}=\bs{d}\bs{d}^\text{T}\bs{X}^\text{T}$. Let's plug that into our equation:


<div>
$
\begin{aligned}
\bs{d}^* &= \argmin{d} \Tr{(\bs{X}^\text{T}-\bs{d}\bs{d}^\text{T}\bs{X}^\text{T})}(\bs{X}-\bs{Xdd}^\text{T})) \\\
&= \argmin{d} \Tr{(\bs{X}^\text{T}\bs{X}-\bs{X}^\text{T}\bs{Xdd}^\text{T} -\bs{d}\bs{d}^\text{T}\bs{X}^\text{T}\bs{X} +\bs{d}\bs{d}^\text{T}\bs{X}^\text{T}\bs{Xdd}^\text{T}}) \\\
&= \argmin{d} \Tr{(\bs{X}^\text{T}\bs{X})} - \Tr{(\bs{X}^\text{T}\bs{Xdd}^\text{T})} - \Tr{(\bs{d}\bs{d}^\text{T}\bs{X}^\text{T}\bs{X})} + \Tr{(\bs{d}\bs{d}^\text{T}\bs{X}^\text{T}\bs{Xdd}^\text{T})}
\end{aligned}
$
</div>

We can remove the first term that not depends on $d$:

<div>
$$
\bs{d}^* = \argmin{d} - \Tr{(\bs{X}^\text{T}\bs{Xdd}^\text{T})} - \Tr{(\bs{d}\bs{d}^\text{T}\bs{X}^\text{T}\bs{X})} + \Tr{(\bs{d}\bs{d}^\text{T}\bs{X}^\text{T}\bs{Xdd}^\text{T})}
$$
</div>

Still because of the cycling property of a trace, we have

<div>
$$
\Tr{(\bs{X}^\text{T}\bs{Xdd}^\text{T})} = \Tr{(\bs{d}\bs{d}^\text{T}\bs{X}^\text{T}\bs{X})}
$$
</div>

We can simplify to:

<div>
$$
\bs{d}^* = \argmin{d} -2\Tr{(\bs{X}^\text{T}\bs{Xdd}^\text{T})} + \Tr{(\bs{d}\bs{d}^\text{T}\bs{X}^\text{T}\bs{Xdd}^\text{T})}
$$
</div>

and then

<div>
$$
\bs{d}^* = \argmin{d} -2\Tr{(\bs{X}^\text{T}\bs{Xdd}^\text{T})} + \Tr{(\bs{X}^\text{T}\bs{Xdd}^\text{T}\bs{d}\bs{d}^\text{T})}
$$
</div>

Because of the constraint $\bs{d}^\text{T}\bs{d}=1$:

<div>
$
\begin{aligned}
\bs{d}^* &= \argmin{d} -2\Tr{(\bs{X}^\text{T}\bs{Xdd}^\text{T})} + \Tr{(\bs{X}^\text{T}\bs{Xd}\bs{d}^\text{T})}\textrm{ subject to }\bs{d}^\text{T}\bs{d}=1 \\\
&= \argmin{d} -\Tr{(\bs{X}^\text{T}\bs{Xdd}^\text{T})}\textrm{ subject to }\bs{d}^\text{T}\bs{d}=1 \\\
&=\argmax{d} \Tr{(\bs{X}^\text{T}\bs{Xdd}^\text{T})}\textrm{ subject to }\bs{d}^\text{T}\bs{d}=1
\end{aligned}
$
</div>

and with the cycling property:

<div>
$$
\bs{d}^* = \argmax{d} \Tr{(\bs{d}^\text{T}\bs{X}^\text{T}\bs{Xd})} \textrm{ subject to }\bs{d}^\text{T}\bs{d}=1
$$
</div>




### Eigendecomposition

We will see that we can find the maximum of the function by calculating the eigenvectors of $\bs{X^\text{T}X}$.


### Covariance matrix

As we wrote above, the optimization problem of maximizing the variance of the components and minimizing the error between the reconstructed and the actual data are equivalent. Actually, if you look at the formula of $\bs{d}$ you can see that there is the term $\bs{X^\text{T}X}$ in the middle.

If we have centered our data around 0 (see bellow for more details about centering), $\bs{X^\text{T}X}$ is the covariance matrix (see [this Quora question](https://www.quora.com/Why-do-we-need-to-center-the-data-for-Principle-Components-Analysis)).

The covariance matrix is a $n$ by $n$ matrix ($n$ being the number of dimensions). Its diagonal is the variance of the corresponding dimensions and the other cells are the covariance between the two corresponding dimensions (the amount of redundancy).

This means that the largest covariance we have between two dimensions the more redundancy exists between these dimensions. This also means that the best-fit line is associated with small errors if the variance is hight. To maximize the variance and minimize the covariance (in order to decorrelate the dimensions) means that the ideal covariance matrix is a diagonal matrix (non-zero values in the diagonal only). Therefore the diagonalization of the covariance matrix will give us the optimal solution.

### Example 2.

As an example we will create again a 2D data set (like in [2.9](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.9-The-Moore-Penrose-Pseudoinverse/)). To see the effect of the PCA we will introduce some correlations between the two dimensions. Let's create 100 data points with 2 dimensions:


```python
np.random.seed(123)
x = 5*np.random.rand(100)
y = 2*x + 1 + np.random.randn(100)

x = x.reshape(100, 1)
y = y.reshape(100, 1)

X = np.hstack([x, y])
X.shape
```

<pre class='output'>
(100, 2)
</pre>


Let's plot the data:


```python
plt.plot(X[:,0], X[:,1], '*')
plt.show()
```


<img src="../../assets/images/2.12/dataset-correlation.png" width="300" alt="Creation of a toy dataset and plot with Python, Numpy and Matplotlib" title="Toy dataset">
<em>Toy dataset with correlated features</em>

Highly correlated data means that the dimensions are redundant. It is possible to predict one from the other without losing much information.

The first processing we will do is to center the data around 0. PCA is a regression model without intercept (see [here](https://stats.stackexchange.com/questions/22329/how-does-centering-the-data-get-rid-of-the-intercept-in-regression-and-pca)) and the first component is thus necessarly crossing the origin.

Here is a simple function that substract the mean of each column to each data point of this column. It can be used to center the data points around 0.


```python
def centerData(X):
    X = X.copy()
    X -= np.mean(X, axis = 0)
    return X
```

So let's center our data $\bs{X}$ around 0 for both dimensions:


```python
X_centered = centerData(X)
plt.plot(X_centered[:,0], X_centered[:,1], '*')
plt.show()
```

<img src="../../assets/images/2.12/dataset-centering.png" width="300" alt="Plot of the dataset with Python, Numpy and Matplotlib after centering" title="Centered data">
<em>The dataset is now centered in $0$</em>

That's better!

We can now look for PCs. We saw that they correspond to values taken by $\bs{d}$ that maximize the following function:

<div>
$$
\bs{d}^* = \argmax{d} \Tr{(\bs{d}^\text{T}\bs{X}^\text{T}\bs{Xd})} \textrm{ subject to }\bs{d}^\text{T}\bs{d}=1
$$
</div>

To find $\bs{d}$ we can calculate the eigenvectors of $\bs{X^\text{T}X}$ (see [2.7](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.7-Eigendecomposition/) for more details about eigendecomposition). So let's do that:


```python
eigVals, eigVecs = np.linalg.eig(X_centered.T.dot(X_centered))
eigVecs
```

<pre class='output'>
array([[-0.91116273, -0.41204669],
       [ 0.41204669, -0.91116273]])
</pre>


These are the vectors maximizing our function. Each column vector is associated with an eigenvalue. The vector associated with the larger eigenvalue tells us the direction associated with the larger variance in our data.

First, let's create a function `plotVectors()` to plot vectors:

```python
def plotVectors(vecs, cols, alpha=1):
    """
    Plot set of vectors.

    Parameters
    ----------
    vecs : array-like
        Coordinates of the vectors to plot. Each vectors is in an array. For
        instance: [[1, 3], [2, 2]] can be used to plot 2 vectors.
    cols : array-like
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
    alpha : float
        Opacity of vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure of the vectors
    """
    plt.figure()
    plt.axvline(x=0, color='#A9A9A9', zorder=0)
    plt.axhline(y=0, color='#A9A9A9', zorder=0)

    for i in range(len(vecs)):
        x = np.concatenate([[0,0],vecs[i]])
        plt.quiver([x[0]],
                   [x[1]],
                   [x[2]],
                   [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=cols[i],
                   alpha=alpha)
```

To check that, we will plot these vectors along with the data.


```python
orange = '#FF9A13'
blue = '#1190FF'
plotVectors(eigVecs.T, [orange, blue])
plt.plot(X_centered[:,0], X_centered[:,1], '*')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
```


<img src="../../assets/images/2.12/principal-component-analysis-eigenvectors.png" width="300" alt="Plot of the dataset and the eigenvectors of its covariance matrix with Python, Numpy and Matplotlib" title="Eigenvectors">
<em>Eigenvectors of the covariance matrix</em>

We can see that the blue vector direction corresponds to the oblique shape of our data. The idea is that if you project the data points on the line corresponding to the blue vector direction you will end up with the largest variance. This vector has the direction that maximizes variance of projected data. Have a look at the following figure:

<img src="../../assets/images/2.12/principal-component-analysis-variance-explained.png" width="400" alt="Representation of the variance explained across directions" title="Maximizing the variance">
<em>Projection of the data point: the line direction is the one with the largest variance</em>

When you project data points on the pink line there is more variance. This line has the direction that maximizes the variance of the data points. It is the same for the figure above: our blue vector has the direction of the line where data point projection has the higher variance. Then the second eigenvector is orthogonal to the first.

In our figure above, the blue vector is the second eigenvector so let's check that it is the one associated with the bigger eigenvalue:


```python
eigVals
```

<pre class='output'>
array([  18.04730409,  798.35242844])
</pre>


So yes, the second vector corresponds to the biggest eigenvalue.

Now that we have found the matrix $\bs{d}$ we will use the encoding function to rotate the data. The goal of the rotation is to end up with a new coordinate system where data is uncorrelated and thus where the basis axes gather all the variance. It is then possible to keep only few axes: this is the purpose of dimensionality reduction.

Recall that the encoding function is:

<div>
$$
\bs{c}=\bs{D}^\text{T}\bs{x}
$$
</div>

$\bs{D}$ is the matrix containing the eigenvectors that we have calculated before. In addition, this formula corresponds to only one data point where dimensions are the rows of $\bs{x}$. In our case, we will apply it to all data points and since $\bs{X}$ has dimensions on the columns we need to transpose it.


```python
X_new = eigVecs.T.dot(X_centered.T)

plt.plot(eigVecs.T.dot(X_centered.T)[0, :], eigVecs.T.dot(X_centered.T)[1, :], '*')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
```

<img src="../../assets/images/2.12/principal-component-analysis-rotation-data.png" width="300" alt="Dataset after rotation" title="Rotation of our dataset">
<em>We rotated the data in order to have the largest variance on one axis</em>

It worked! The rotation transformed our dataset that have now the more variance on one of the basis axis. You could keep only this dimension and have a fairly good representation of the data.

### About the unit norm constraint

We saw that the maximization is subject to $\bs{d}^\text{T}\bs{d}=1$. This means that the solution vector has to be a unit vector. Without this constraint, you could scale $\bs{d}$ up to the infinity to increase the function to maximize (see [here](https://stats.stackexchange.com/questions/117695/why-is-the-eigenvector-in-pca-taken-to-be-unit-norm)). For instance, let's see some vectors $\bs{x}$ that could maximize the function:


```python
d = np.array([[12], [26]])
d.T.dot(X.T).dot(X).dot(d)
```

<pre class='output'>
array([[ 4165298.04389264]])
</pre>


However this $\bs{d}$ has not a unit norm (since $\bs{d}$ is a column vector we use the transpose of $\bs{d}^\text{T}\bs{d}=1$ (see [2.2](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.2-Multiplying-Matrices-and-Vectors/)):


```python
d.T.dot(d)
```

<pre class='output'>
array([[820]])
</pre>


The eigenvectors have unit norm and thus respect the constraint:


```python
eigVecs[:,0].dot(eigVecs[:,0].T)
```

<pre class='output'>
1.0
</pre>


and


```python
eigVecs[:,1].dot(eigVecs[:,1].T)
```

<pre class='output'>
1.0
</pre>


And... This is the end! We have gone through a lot of things during this series on linear algebra! I hope that it was a useful introduction to this topic which is of large importance in the data science/machine learning/deep learning fields.

{% include mailchimp.html %}

# References

## PCA

- [A lot of intuitive explanations on PCA](https://arxiv.org/pdf/1404.1100.pdf)

- [Principal Component Analysis](https://brilliant.org/wiki/principal-component-analysis/#from-approximate-equality-to-minimizing-function)

- [Linear algebra - ncsu](http://www4.ncsu.edu/~slrace/LinearAlgebra2017/Slides/PCAPrint.pdf)

- [A one stop shop for PCA](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)

- [PCA - Ben-Gurion University](https://www.cs.bgu.ac.il/~inabd171/wiki.files/lecture14_handouts.pdf)

## Semi-orthogonal matrix

- [Wikipedia - Semi orthogonal matrix](https://en.wikipedia.org/wiki/Semi-orthogonal_matrix)

## Intuition about PCA

- [Blog George M Dallas](https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/)

## Derivatives

- [SE - Derivative of vector and vector transpose product](https://math.stackexchange.com/questions/1377764/derivative-of-vector-and-vector-transpose-product)

## Link between variance maximized and error minimized:

- [SE - What norm of the reconstruction error is minimized by the low rank approximation](https://stats.stackexchange.com/questions/130721/what-norm-of-the-reconstruction-error-is-minimized-by-the-low-rank-approximation)

- [SE - PCA objective function](https://stats.stackexchange.com/questions/32174/pca-objective-function-what-is-the-connection-between-maximizing-variance-and-m)

- [SE - Why do the leading eigenvectors of A maximize...](https://stats.stackexchange.com/questions/318625/why-do-the-leading-eigenvectors-of-a-maximize-texttrdtad)

## Centering data

- [Quora - Why do we need to center the data for PCA](https://www.quora.com/Why-do-we-need-to-center-the-data-for-Principle-Components-Analysis)

- [SE - How does centering the data get rid of the intercept in regression and PCA](https://stats.stackexchange.com/questions/22329/how-does-centering-the-data-get-rid-of-the-intercept-in-regression-and-pca)

## Unit norm constraint

- [SE - Why is the eigenvector in PCA taken to be unit norm](https://stats.stackexchange.com/questions/117695/why-is-the-eigenvector-in-pca-taken-to-be-unit-norm)

<span class='notes'>
    Feel free to drop me an email or a comment. The syllabus of this series can be found [in the introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/). All the notebooks can be found on [Github](https://github.com/hadrienj/deepLearningBook-Notes).
</span>

<span class='notes'>
    This content is part of a series following the chapter 2 on linear algebra from the [Deep Learning Book](http://www.deeplearningbook.org/) by Goodfellow, I., Bengio, Y., and Courville, A. (2016). It aims to provide intuitions/drawings/python code on mathematical theories and is constructed as my understanding of these concepts. You can check the syllabus in the [introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/) but here are the links to the other articles:
</span>

