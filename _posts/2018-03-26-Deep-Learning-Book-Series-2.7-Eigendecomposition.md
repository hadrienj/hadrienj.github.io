---
bg: "flower.jpg"
layout: post
mathjax: true
title: Deep Learning Book Series · 2.7 Eigendecomposition
crawlertitle: "Introduction to Eigendecomposition using Python/Numpy examples and drawings"
categories: posts
tags: ['linear-algebra', 'python', 'numpy', 'deep-learning-book']
author: hadrienj
jupyter: https://github.com/hadrienj/deepLearningBook-Notes/blob/master/2.7%20Eigendecomposition/2.7%20Eigendecomposition.ipynb
date: 2018-03-26 15:30:00
excerpt: This post introduces the concept of eigendecomposition. We will start with getting some intuitions on eigenvectors and eigenvalues. We will develop on the idea that a matrix can be seen as a linear transformation and that applying a matrix on its eigenvectors gives new vectors. We will also see how to visualize linear transformations in Python!
excerpt-image: <img src="../../assets/images/2.7/unit-circle-eigenvectors.png" width="250" alt="Plot of the unit circle and its transformation by the matrix A" title="Transformation of the unit circle by the matrix A">
    <em>The unit circle and its transformation by the matrix A. The vectors are the eigenvectors of A.</em>
deep-learning-book-toc: true
---

# Introduction

We will see some major concepts of linear algebra in this chapter. We will start with getting some ideas on eigenvectors and eigenvalues. We will develop on the idea that a matrix can be seen as a linear transformation and that applying a matrix on its eigenvectors gives new vectors with the same direction. Then we will see how to express quadratic equations into the matrix form. We will see that the eigendecomposition of the matrix corresponding to a quadratic equation can be used to find the minimum and maximum of this function. As a bonus, we will also see how to visualize linear transformations in Python!

{% include mailchimp.html %}

# 2.7 Eigendecomposition

The eigendecomposition is one form of matrix decomposition. Decomposing a matrix means that we want to find a product of matrices that is equal to the initial matrix. In the case of the eigendecomposition, we decompose the initial matrix into the product of its eigenvectors and eigenvalues. Before all, let's see what are eigenvectors and eigenvalues.

# Matrices as linear transformations

As we have seen in [2.3](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.3-Identity-and-Inverse-Matrices/) with the example of the identity matrix, you can think of matrices as linear transformations. Some matrices will rotate your space, others will rescale it etc. So when we apply a matrix to a vector, we end up with a transformed version of the vector. When we say that we 'apply' the matrix to the vector it means that we calculate the dot product of the matrix with the vector. We will start with a basic example of this kind of transformation.

### Example 1.


```python
A = np.array([[-1, 3], [2, -2]])
A
```

<pre class='output'>
array([[-1,  3],
       [ 2, -2]])
</pre>



```python
v = np.array([[2], [1]])
v
```

<pre class='output'>
array([[2],
       [1]])
</pre>


Let's plot this vector:


```python
plotVectors([v.flatten()], cols=['#1190FF'])
plt.ylim(-1, 4)
plt.xlim(-1, 4)
```

<pre class='output'>
(-1, 4)
</pre>

<img src="../../assets/images/2.7/simple-vector.png" width="250" alt="Example of a simple vector" title="A simple vector">
<em>A simple vector</em>


Now, we will apply the matrix $\bs{A}$ to this vector and plot the old vector (light blue) and the new one (orange):


```python
Av = A.dot(v)
print Av
plotVectors([v.flatten(), Av.flatten()], cols=['#1190FF', '#FF9A13'])
plt.ylim(-1, 4)
plt.xlim(-1, 4)
```

    [[1]
     [2]]


<pre class='output'>
(-1, 4)
</pre>


<img src="../../assets/images/2.7/simple-vector-and-transformation.png" width="250" alt="A simple vector and its transformation" title="A simple vector and its transformation">
<em>A simple vector and its transformation</em>

We can see that applying the matrix $\bs{A}$ has the effect of modifying the vector.

Now that you can think of matrices as linear transformation recipes, let's see the case of a very special type of vector: the eigenvector.

# Eigenvectors and eigenvalues

We have seen an example of a vector transformed by a matrix. Now imagine that the transformation of the initial vector gives us a new vector that has the exact same direction. The scale can be different but the direction is the same. Applying the matrix didn't change the direction of the vector. This special vector is called an eigenvector of the matrix. We will see that finding the eigenvectors of a matrix can be very useful.

<span class='pquote'>
    Imagine that the transformation of the initial vector by the matrix gives a new vector with the exact same direction. This vector is called an eigenvector of $\bs{A}$.
</span>

This means that $\bs{v}$ is a eigenvector of $\bs{A}$ if $\bs{v}$ and $\bs{Av}$ are in the same direction or to rephrase it if the vectors $\bs{Av}$ and $\bs{v}$ are parallel. The output vector is just a scaled version of the input vector. This scalling factor is $\lambda$ which is called the **eigenvalue** of $\bs{A}$.

<div>
$$
\bs{Av} = \lambda\bs{v}
$$
</div>

### Example 2.

Let's $\bs{A}$ be the following matrix:

<div>
$$
\bs{A}=
\begin{bmatrix}
    5 & 1\\\\
    3 & 3
\end{bmatrix}
$$
</div>

We know that one eigenvector of A is:

<div>
$$
\bs{v}=
\begin{bmatrix}
    1\\\\
    1
\end{bmatrix}
$$
</div>

We can check that $\bs{Av} = \lambda\bs{v}$:

<div>
$$
\begin{bmatrix}
    5 & 1\\\\
    3 & 3
\end{bmatrix}
\begin{bmatrix}
    1\\\\
    1
\end{bmatrix}=\begin{bmatrix}
    6\\\\
    6
\end{bmatrix}
$$
</div>

We can see that:

<div>
$$
6\times \begin{bmatrix}
    1\\\\
    1
\end{bmatrix} = \begin{bmatrix}
    6\\\\
    6
\end{bmatrix}
$$
</div>

which means that $\bs{v}$ is well an eigenvector of $\bs{A}$. Also, the corresponding eigenvalue is $\lambda=6$.

We can represent $\bs{v}$ and $\bs{Av}$ to check if their directions are the same:


```python
A = np.array([[5, 1], [3, 3]])
A
```

<pre class='output'>
array([[5, 1],
       [3, 3]])
</pre>



```python
v = np.array([[1], [1]])
v
```

<pre class='output'>
array([[1],
       [1]])
</pre>



```python
Av = A.dot(v)

orange = '#FF9A13'
blue = '#1190FF'

plotVectors([Av.flatten(), v.flatten()], cols=[blue, orange])
plt.ylim(-1, 7)
plt.xlim(-1, 7)
```

<pre class='output'>
(-1, 7)
</pre>

<img src="../../assets/images/2.7/eigenvector-transformation.png" width="250" alt="The direction of the eigenvector after transformation by its matrix is the same as the original vector direction" title="Eigenvector direction">
<em>Eigenvector doesn't change its direction when we apply the corresponding matrix</em>

We can see that their directions are the same!

Another eigenvector of $\bs{A}$ is

<div>
$$
\bs{v}=
\begin{bmatrix}
    1\\\\
    -3
\end{bmatrix}
$$
</div>

because

<div>
$$
\begin{bmatrix}
    5 & 1\\\\
    3 & 3
\end{bmatrix}\begin{bmatrix}
    1\\\\
    -3
\end{bmatrix} = \begin{bmatrix}
    2\\\\
    -6
\end{bmatrix}
$$
</div>

and

<div>
$$
2 \times \begin{bmatrix}
    1\\\\
    -3
\end{bmatrix} =
\begin{bmatrix}
    2\\\\
    -6
\end{bmatrix}
$$
</div>

So the corresponding eigenvalue is $\lambda=2$.


```python
v = np.array([[1], [-3]])
v
```

<pre class='output'>
array([[ 1],
       [-3]])
</pre>



```python
Av = A.dot(v)

plotVectors([Av.flatten(), v.flatten()], cols=[blue, orange])
plt.ylim(-7, 1)
plt.xlim(-1, 3)
```

<pre class='output'>
(-1, 3)
</pre>


<img src="../../assets/images/2.7/eigenvector-transformation1.png" width="250" alt="Another eigenvector and its transformation" title="Another eigenvector and its transformation">
<em>Another eigenvector and its transformation</em>


This example shows that the eigenvectors $\bs{v}$ are vectors that change only in scale when we apply the matrix $\bs{A}$ to them. Here the scales were 6 for the first eigenvector and 2 to the second but $\lambda$ can take any real or even complex value.

## Find eigenvalues and eigenvectors in Python

Numpy provides a function returning eigenvectors and eigenvalues (the first array corresponds to the eigenvalues and the second to the eigenvectors concatenated in columns):

```python
(array([ 6.,  2.]), array([[ 0.70710678, -0.31622777],
       [ 0.70710678,  0.9486833 ]]))
```

Here a demonstration with the preceding example.


```python
A = np.array([[5, 1], [3, 3]])
A
```

<pre class='output'>
array([[5, 1],
       [3, 3]])
</pre>



```python
np.linalg.eig(A)
```

<pre class='output'>
(array([ 6.,  2.]), array([[ 0.70710678, -0.31622777],
        [ 0.70710678,  0.9486833 ]]))
</pre>


We can see that the eigenvalues are the same than the ones we used before: 6 and 2 (first array).

The eigenvectors correspond to the columns of the second array. This means that the eigenvector corresponding to $\lambda=6$ is:

<div>
$$
\begin{bmatrix}
    0.70710678\\\\
    0.70710678
\end{bmatrix}
$$
</div>

The eigenvector corresponding to $\lambda=2$ is:

<div>
$$
\begin{bmatrix}
    -0.31622777\\\\
    0.9486833
\end{bmatrix}
$$
</div>

The eigenvectors look different because they have not necessarly the same scaling than the ones we gave in the example. We can easily see that the first corresponds to a scaled version of our $\begin{bmatrix}
    1\\\\
    1
\end{bmatrix}$. But the same property stands. We have still $\bs{Av} = \lambda\bs{v}$:

<div>
$$
\begin{bmatrix}
    5 & 1\\\\
    3 & 3
\end{bmatrix}
\begin{bmatrix}
    0.70710678\\\\
    0.70710678
\end{bmatrix}=
\begin{bmatrix}
    4.24264069\\\\
    4.24264069
\end{bmatrix}
$$
</div>

With $0.70710678 \times 6 = 4.24264069$. So there are an infinite number of eigenvectors corresponding to the eigenvalue $6$. They are equivalent because we are interested by their directions.

For the second eigenvector we can check that it corresponds to a scaled version of $\begin{bmatrix}
    1\\\\
    -3
\end{bmatrix}$. We can draw these vectors and see if they are parallel.


```python
v = np.array([[1], [-3]])
Av = A.dot(v)
v_np = [-0.31622777, 0.9486833]

plotVectors([Av.flatten(), v.flatten(), v_np], cols=[blue, orange, 'blue'])
plt.ylim(-7, 1)
plt.xlim(-1, 3)
```

<pre class='output'>
(-1, 3)
</pre>


<img src="../../assets/images/2.7/eigenvector-numpy.png" width="250" alt="Eigenvectors found in numpy" title="Eigenvectors found in numpy">
<em>Eigenvectors found in numpy have identical directions</em>

We can see that the vector found with Numpy (in dark blue) is a scaled version of our preceding $\begin{bmatrix}
    1\\\\
    -3
\end{bmatrix}$.

## Rescaled vectors

As we saw it with numpy, if $\bs{v}$ is an eigenvector of $\bs{A}$, then any rescaled vector $s\bs{v}$ is also an eigenvector of $\bs{A}$. The eigenvalue of the rescaled vector is the same.

Let's try to rescale

$$
\bs{v}=
\begin{bmatrix}
    1\\\\
    -3
\end{bmatrix}
$$

from our preceding example.

For instance,

<div>
$$
\bs{3v}=
\begin{bmatrix}
    3\\\\
    -9
\end{bmatrix}
$$
</div>

So we have:

<div>
$$
\begin{align*}
\begin{bmatrix}
    5 & 1\\\\
    3 & 3
\end{bmatrix}
\begin{bmatrix}
    3\\\\
    -9
\end{bmatrix}
&=
\begin{bmatrix}
    6\\\\
    18
\end{bmatrix}\\\\
&= 2 \times
\begin{bmatrix}
    3\\\\
    -9
\end{bmatrix}
\end{align*}
$$
</div>

We have well $\bs{A}\times 3\bs{v} = \lambda \times 3\bs{v}$ and the eigenvalue is still $\lambda=2$.

## Concatenating eigenvalues and eigenvectors

Now that we have an idea of what eigenvectors and eigenvalues are we can see how it can be used to decompose a matrix. All eigenvectors of a matrix $\bs{A}$ can be concatenated in a matrix with each column corresponding to each eigenvector (like in the second array return by `np.linalg.eig(A)`):

<div>
$$
\bs{V}=
\begin{bmatrix}
    1 & 1\\\\
    1 & -3
\end{bmatrix}
$$
</div>

The first column

$$
\begin{bmatrix}
    1\\\\
    1
\end{bmatrix}
$$

corresponds to $\lambda=6$ and the second

$$
\begin{bmatrix}
    1\\\\
    -3
\end{bmatrix}
$$

to $\lambda=2$.

The vector $\bs{\lambda}$ can be created from all eigenvalues:

<div>
$$
\bs{\lambda}=
\begin{bmatrix}
    6\\\\
    2
\end{bmatrix}
$$
</div>

Then the eigendecomposition is given by

<div>
$$
\bs{A}=\bs{V}\cdot diag(\bs{\lambda}) \cdot \bs{V}^{-1}
$$
</div>

<span class='pquote'>
    We can decompose the matrix $\bs{A}$ with eigenvectors and eigenvalues. It is done with: $\bs{A}=\bs{V}\cdot diag(\bs{\lambda}) \cdot \bs{V}^{-1}$
</span>

$diag(\bs{v})$ is a diagonal matrix (see [2.6](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.6-Special-Kinds-of-Matrices-and-Vectors/)) containing all the eigenvalues. Continuing with our example we have

<div>
$$
\bs{V}=\begin{bmatrix}
    1 & 1\\\\
    1 & -3
\end{bmatrix}
$$
</div>

The diagonal matrix is all zeros except the diagonal that is our vector $\bs{\lambda}$.

<div>
$$
diag(\bs{v})=
\begin{bmatrix}
    6 & 0\\\\
    0 & 2
\end{bmatrix}
$$
</div>

The inverse matrix of $\bs{V}$ can be calculated with numpy:


```python
V = np.array([[1, 1], [1, -3]])
V
```

<pre class='output'>
array([[ 1,  1],
       [ 1, -3]])
</pre>



```python
V_inv = np.linalg.inv(V)
V_inv
```

<pre class='output'>
array([[ 0.75,  0.25],
       [ 0.25, -0.25]])
</pre>


So let's plug

<div>
$$
\bs{V}^{-1}=\begin{bmatrix}
    0.75 & 0.25\\\\
    0.25 & -0.25
\end{bmatrix}
$$
</div>

into our equation:

<div>
$$
\begin{align*}
&\bs{V}\cdot diag(\bs{\lambda}) \cdot \bs{V}^{-1}\\\\
&=
\begin{bmatrix}
    1 & 1\\\\
    1 & -3
\end{bmatrix}\\\\
&\begin{bmatrix}
    6 & 0\\\\
    0 & 2
\end{bmatrix}
\begin{bmatrix}
    0.75 & 0.25\\\\
    0.25 & -0.25
\end{bmatrix}
\end{align*}
$$
</div>

If we do the dot product of the first two matrices we have:

<div>
$$
\begin{bmatrix}
    1 & 1\\\\
    1 & -3
\end{bmatrix}
\begin{bmatrix}
    6 & 0\\\\
    0 & 2
\end{bmatrix} =
\begin{bmatrix}
    6 & 2\\\\
    6 & -6
\end{bmatrix}
$$
</div>

So if we replace into the equation:

<div>
$$
\begin{align*}
&\begin{bmatrix}
    6 & 2\\\\
    6 & -6
\end{bmatrix}
\begin{bmatrix}
    0.75 & 0.25\\\\
    0.25 & -0.25
\end{bmatrix}\\\\
&=
\begin{bmatrix}
    5 & 1\\\\
    3 & 3
\end{bmatrix}=
\bs{A}
\end{align*}
$$
</div>

Let's check our result with Python:


```python
lambdas = np.diag([6,2])
lambdas
```

<pre class='output'>
array([[6, 0],
       [0, 2]])
</pre>



```python
V.dot(lambdas).dot(V_inv)
```

<pre class='output'>
array([[ 5.,  1.],
       [ 3.,  3.]])
</pre>


That confirms our previous calculation.

## Real symmetric matrix

In the case of real symmetric matrices (more details about symmetric matrices in [2.6](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.6-Special-Kinds-of-Matrices-and-Vectors/)), the eigendecomposition can be expressed as

<div>
$$
\bs{A} = \bs{Q}\Lambda \bs{Q}^\text{T}
$$
</div>

where $\bs{Q}$ is the matrix with eigenvectors as columns and $\Lambda$ is $diag(\lambda)$.

### Example 3.

<div>
$$
\bs{A}=\begin{bmatrix}
    6 & 2\\\\
    2 & 3
\end{bmatrix}
$$
</div>

This matrix is symmetric because $\bs{A}=\bs{A}^\text{T}$. Its eigenvectors are:

<div>
$$
\bs{Q}=
\begin{bmatrix}
    0.89442719 & -0.4472136\\\\
    0.4472136 & 0.89442719
\end{bmatrix}
$$
</div>

and its eigenvalues put in a diagonal matrix gives:

<div>
$$
\bs{\Lambda}=
\begin{bmatrix}
    7 & 0\\\\
    0 & 2
\end{bmatrix}
$$
</div>

So let's begin to calculate $\bs{Q\Lambda}$:

<div>
$$
\begin{align*}
\bs{Q\Lambda}&=
\begin{bmatrix}
    0.89442719 & -0.4472136\\\\
    0.4472136 & 0.89442719
\end{bmatrix}
\begin{bmatrix}
    7 & 0\\\\
    0 & 2
\end{bmatrix}\\\\
&=
\begin{bmatrix}
    0.89442719 \times 7 & -0.4472136\times 2\\\\
    0.4472136 \times 7 & 0.89442719\times 2
\end{bmatrix}\\\\
&=
\begin{bmatrix}
    6.26099033 & -0.8944272\\\\
    3.1304952 & 1.78885438
\end{bmatrix}
\end{align*}
$$
</div>

with:

<div>
$$
\bs{Q}^\text{T}=
\begin{bmatrix}
    0.89442719 & 0.4472136\\\\
    -0.4472136 & 0.89442719
\end{bmatrix}
$$
</div>

So we have:

<div>
$$
\begin{align*}
\bs{Q\Lambda} \bs{Q}^\text{T}&=
\begin{bmatrix}
    6.26099033 & -0.8944272\\\\
    3.1304952 & 1.78885438
\end{bmatrix}\\\\
&\cdot
\begin{bmatrix}
    0.89442719 & 0.4472136\\\\
    -0.4472136 & 0.89442719
\end{bmatrix}\\\\
&=
\begin{bmatrix}
    6 & 2\\\\
    2 & 3
\end{bmatrix}
\end{align*}
$$
</div>

It works! For that reason, it can useful to use symmetric matrices! Let's do the same things easily with `linalg` from numpy:


```python
A = np.array([[6, 2], [2, 3]])
A
```

<pre class='output'>
array([[6, 2],
       [2, 3]])
</pre>



```python
eigVals, eigVecs = np.linalg.eig(A)
eigVecs
```

<pre class='output'>
array([[ 0.89442719, -0.4472136 ],
       [ 0.4472136 ,  0.89442719]])
</pre>



```python
eigVals = np.diag(eigVals)
eigVals
```

<pre class='output'>
array([[ 7.,  0.],
       [ 0.,  2.]])
</pre>



```python
eigVecs.dot(eigVals).dot(eigVecs.T)
```

<pre class='output'>
array([[ 6.,  2.],
       [ 2.,  3.]])
</pre>


We can see that the result corresponds to our initial matrix.

# Quadratic form to matrix form

Eigendecomposition can be used to optimize quadratic functions. We will see that when $\bs{x}$ takes the values of an eigenvector, $f(\bs{x})$ takes the value of its corresponding eigenvalue.

<span class='pquote'>
    When $\bs{x}$ takes the values of an eigenvector, $f(\bs{x})$ takes the value of its corresponding eigenvalue.
</span>

We will see in the following points how we can show that with different methods.

Let's have the following quadratic equation:

<div>
$$
f(\bs{x}) = ax_1^2 +(b+c)x_1x_2 + dx_2^2
$$
</div>

These quadratic forms can be generated by matrices:

<div>
$$
\begin{align*}
f(\bs{x})
&= \begin{bmatrix}
    x_1 & x_2
\end{bmatrix}\begin{bmatrix}
    a & b\\\\
    c & d
\end{bmatrix}\begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}\\\\
&= \bs{x^\text{T}Ax}
\end{align*}
$$
</div>

with:

<div>
$$
\bs{x} = \begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}
$$
</div>

and

<div>
$$
\bs{A}=\begin{bmatrix}
    a & b\\\\
    c & d
\end{bmatrix}
$$
</div>

We call them matrix forms. This form is useful to do various things on the quadratic equation like constrained optimization (see bellow).

<span class='pquote'>
    Quadratic equations can be expressed under the matrix form
</span>

If you look at the relation between these forms you can see that $a$ gives you the number of $x_1^2$, $(b + c)$ the number of $x_1x_2$ and $d$ the number of $x_2^2$. This means that the same quadratic form can be obtained from infinite number of matrices $\bs{A}$ by changing $b$ and $c$ while preserving their sum.

### Example 4.

<div>
$$
\bs{x} = \begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}
$$
</div>

and

<div>
$$
\bs{A}=\begin{bmatrix}
    2 & 4\\\\
    2 & 5
\end{bmatrix}
$$
</div>

gives the following quadratic form:

<div>
$$
\begin{align*}
&2x_1^2 + (4+2)x_1x_2 + 5x_2^2\\\\
&=2x_1^2 + 6x_1x_2 + 5x_2^2
\end{align*}
$$
</div>

but if:

<div>
$$
\bs{A}=\begin{bmatrix}
    2 & -3\\\\
    9 & 5
\end{bmatrix}
$$
</div>

we still have the quadratic same form:

<div>
$$
\begin{align*}
&2x_1^2 + (-3+9)x_1x_2 + 5x_2^2\\\\
&=2x_1^2 + 6x_1x_2 + 5x_2^2
\end{align*}
$$
</div>

### Example 5

For this example, we will go from the matrix form to the quadratic form using a symmetric matrix $\bs{A}$. Let's use the matrix of the example 3.

<div>
$$
\bs{x} = \begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}
$$
</div>

and

<div>
$$\bs{A}=\begin{bmatrix}
    6 & 2\\\\
    2 & 3
\end{bmatrix}
$$
</div>

<div>
$$
\begin{align*}
\bs{x^\text{T}Ax}&=
\begin{bmatrix}
    x_1 & x_2
\end{bmatrix}
\begin{bmatrix}
    6 & 2\\\\
    2 & 3
\end{bmatrix}
\begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}\\\\
&=
\begin{bmatrix}
    x_1 & x_2
\end{bmatrix}
\begin{bmatrix}
    6 x_1 + 2 x_2\\\\
    2 x_1 + 3 x_2
\end{bmatrix}\\\\
&=
x_1(6 x_1 + 2 x_2) + x_2(2 x_1 + 3 x_2)\\\\
&=
6 x_1^2 + 4 x_1x_2 + 3 x_2^2
\end{align*}
$$
</div>

Our quadratic equation is thus:

<div>
$$
6 x_1^2 + 4 x_1x_2 + 3 x_2^2
$$
</div>

### Note

If $\bs{A}$ is a diagonal matrix (all 0 except the diagonal), the quadratic form of $\bs{x^\text{T}Ax}$ will have no cross term. Take the following matrix form:

<div>
$$
\bs{A}=\begin{bmatrix}
    a & b\\\\
    c & d
\end{bmatrix}
$$
</div>

If $\bs{A}$ is diagonal, then $b$ and $c$ are 0 and since $f(\bs{x}) = ax_1^2 +(b+c)x_1x_2 + dx_2^2$ there is no cross term. A quadratic form without cross term is called diagonal form since it comes from a diagonal matrix.

# Change of variable

A change of variable (or linear substitution) simply means that we replace a variable by another one. We will see that it can be used to remove the cross terms in our quadratic equation. Without the cross term, it will then be easier to characterize the function and eventually optimize it (i.e finding its maximum or minimum).

## With the quadratic form

### Example 6.

Let's take again our previous quadratic form:

<div>
$$
\bs{x^\text{T}Ax} = 6 x_1^2 + 4 x_1x_2 + 3 x_2^2
$$
</div>

The change of variable will concern $x_1$ and $x_2$. We can replace $x_1$ with any combination of $y_1$ and $y_2$ and $x_2$ with any combination $y_1$ and $y_2$. We will of course end up with a new equation. The nice thing is that we can find a specific substitution that will lead to a simplification of our statement. Specifically, it can be used to get rid of the cross term (in our example: $4 x_1x_2$). We will see later why it is interesting.

Actually, the right substitution is given by the eigenvectors of the matrix used to generate the quadratic form. Let's recall that the matrix form of our equation is:

<div>
$$
\bs{x} = \begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}
$$
</div>

and

<div>
$$\bs{A}=\begin{bmatrix}
    6 & 2\\\\
    2 & 3
\end{bmatrix}
$$
</div>

and that the eigenvectors of $\bs{A}$ are:

<div>
$$
\begin{bmatrix}
    0.89442719 & -0.4472136\\\\
    0.4472136 & 0.89442719
\end{bmatrix}
$$
</div>

With the purpose of simplification, we can replace these values with:

<div>
$$
\begin{bmatrix}
    \frac{2}{\sqrt{5}} & -\frac{1}{\sqrt{5}}\\\\
    \frac{1}{\sqrt{5}} & \frac{2}{\sqrt{5}}
\end{bmatrix} =
\frac{1}{\sqrt{5}}
\begin{bmatrix}
    2 & -1\\\\
    1 & 2
\end{bmatrix}
$$
</div>

So our first eigenvector is:

<div>
$$
\frac{1}{\sqrt{5}}
\begin{bmatrix}
    2\\\\
    1
\end{bmatrix}
$$
</div>

and our second eigenvector is:

<div>
$$
\frac{1}{\sqrt{5}}
\begin{bmatrix}
    -1\\\\
    2
\end{bmatrix}
$$
</div>

The change of variable will lead to:

<div>
$$
\begin{align*}
\begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}
&=
\frac{1}{\sqrt{5}}
\begin{bmatrix}
    2 & -1\\\\
    1 & 2
\end{bmatrix}
\begin{bmatrix}
    y_1\\\\
    y_2
\end{bmatrix}\\\\
&=
\frac{1}{\sqrt{5}}
\begin{bmatrix}
    2y_1 - y_2\\\\
    y_1 + 2y_2
\end{bmatrix}
\end{align*}
$$
</div>

so we have

<div>
$$
\begin{cases}
x_1 = \frac{1}{\sqrt{5}}(2y_1 - y_2)\\\\
x_2 = \frac{1}{\sqrt{5}}(y_1 + 2y_2)
\end{cases}
$$
</div>

So far so good! Let's replace that in our example:

<div>
$$
\begin{align*}
\bs{x^\text{T}Ax}
&=6 x_1^2 + 4 x_1x_2 + 3 x_2^2\\\\
&=6 [\frac{1}{\sqrt{5}}(2y_1 - y_2)]^2 + 4 [\frac{1}{\sqrt{5}}(2y_1 - y_2)\frac{1}{\sqrt{5}}(y_1 + 2y_2)] + 3 [\frac{1}{\sqrt{5}}(y_1 + 2y_2)]^2\\\\
&=
\frac{1}{5}[6 (2y_1 - y_2)^2 + 4 (2y_1 - y_2)(y_1 + 2y_2) + 3 (y_1 + 2y_2)^2]\\\\
&=
\frac{1}{5}[6 (4y_1^2 - 4y_1y_2 + y_2^2) + 4 (2y_1^2 + 4y_1y_2 - y_1y_2 - 2y_2^2) + 3 (y_1^2 + 4y_1y_2 + 4y_2^2)]\\\\
&=
\frac{1}{5}(24y_1^2 - 24y_1y_2 + 6y_2^2 + 8y_1^2 + 16y_1y_2 - 4y_1y_2 - 8y_2^2 + 3y_1^2 + 12y_1y_2 + 12y_2^2)\\\\
&=
\frac{1}{5}(35y_1^2 + 10y_2^2)\\\\
&=
7y_1^2 + 2y_2^2
\end{align*}
$$
</div>

That's great! Our new equation doesn't have any cross terms!

## With the Principal Axes Theorem

Actually there is a simpler way to do the change of variable. We can stay in the matrix form. Recall that we start with the form:

<div>
$$
f(\bs{x})=\bs{x^\text{T}Ax}
$$
</div>

The linear substitution can be wrote in these terms. We want replace the variables $\bs{x}$ by $\bs{y}$ that relates by:

<div>
$$
\bs{x}=P\bs{y}
$$
</div>

We want to find $P$ such as our new equation (after the change of variable) doesn't contain the cross terms. The first step is to replace that in the first equation:

<div>
$$
\begin{align*}
\bs{x^\text{T}Ax}
&=
(\bs{Py})^\text{T}\bs{A}(\bs{Py})\\\\
&=
\bs{y}^\text{T}(\bs{P}^\text{T}\bs{AP})\bs{y}
\end{align*}
$$
</div>

Can you see the how to transform the left hand side ($\bs{x}$) into the right hand side ($\bs{y}$)? The substitution is done by replacing $\bs{A}$ with $\bs{P^\text{T}AP}$. We also know that $\bs{A}$ is symmetric and thus that there is a diagonal matrix $\bs{D}$ containing the eigenvectors of $\bs{A}$ and such as $\bs{D}=\bs{P}^\text{T}\bs{AP}$. We thus end up with:

<div>
$$
\bs{x^\text{T}Ax}=\bs{y^\text{T}\bs{D} y}
$$
</div>

All of this implies that we can use $\bs{D}$ to simplify our quadratic equation and remove the cross terms. If you remember from example 2 we know that the eigenvalues of $\bs{A}$ are:

<div>
$$
\bs{D}=
\begin{bmatrix}
    7 & 0\\\\
    0 & 2
\end{bmatrix}
$$
</div>

<div>
$$
\begin{align*}
\bs{x^\text{T}Ax}
&=
\bs{y^\text{T}\bs{D} y}\\\\
&=
\bs{y}^\text{T}
\begin{bmatrix}
    7 & 0\\\\
    0 & 2
\end{bmatrix}
\bs{y}\\\\
&=
\begin{bmatrix}
    y_1 & y_2
\end{bmatrix}
\begin{bmatrix}
    7 & 0\\\\
    0 & 2
\end{bmatrix}
\begin{bmatrix}
    y_1\\\\
    y_2
\end{bmatrix}\\\\
&=
\begin{bmatrix}
    7y_1 +0y_2 & 0y_1 + 2y_2
\end{bmatrix}
\begin{bmatrix}
    y_1\\\\
    y_2
\end{bmatrix}\\\\
&=
7y_1^2 + 2y_2^2
\end{align*}
$$
</div>

That's nice! If you look back to the change of variable that we have done in the quadratic form, you will see that we have found the same values!

This form (without cross-term) is called the **principal axes form**.

### Summary

To summarise, the principal axes form can be found with

<div>
$$
\bs{x^\text{T}Ax} = \lambda_1y_1^2 + \lambda_2y_2^2
$$
</div>

where $\lambda_1$ is the eigenvalue corresponding to the first eigenvector and $\lambda_2$ the eigenvalue corresponding to the second eigenvector (second column of $\bs{x}$).

# Finding f(x) with eigendecomposition

We will see that there is a way to find $f(\bs{x})$ with eigenvectors and eigenvalues when $\bs{x}$ is a unit vector.

Let's start from:

<div>
$$
f(\bs{x}) =\bs{x^\text{T}Ax}
$$
</div>

We know that if $\bs{x}$ is an eigenvector of $\bs{A}$ and $\lambda$ the corresponding eigenvalue, then $
\bs{Ax}=\lambda \bs{x}
$. By replacing the term in the last equation we have:

<div>
$$
f(\bs{x}) =\bs{x^\text{T}\lambda x} = \bs{x^\text{T}x}\lambda
$$
</div>

Since $\bs{x}$ is a unit vector, $\norm{\bs{x}}_2=1$ and $\bs{x^\text{T}x}=1$ (cf. [2.5](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.5-Norms/) Norms). We end up with

<div>
$$
f(\bs{x}) = \lambda
$$
</div>

This is a usefull property. If $\bs{x}$ is an eigenvector of $\bs{A}$, $
f(\bs{x}) =\bs{x^\text{T}Ax}$ will take the value of the corresponding eigenvalue. We can see that this is working only if the euclidean norm of $\bs{x}$ is 1 (i.e $\bs{x}$ is a unit vector).

### Example 7

This example will show that $f(\bs{x}) = \lambda$. Let's take again the last example, the eigenvectors of $\bs{A}$ were

<div>
$$
\bs{Q}=
\begin{bmatrix}
    0.89442719 & -0.4472136\\\\
    0.4472136 & 0.89442719
\end{bmatrix}
$$
</div>

and the eigenvalues

<div>
$$
\bs{\Lambda}=
\begin{bmatrix}
    7 & 0\\\\
    0 & 2
\end{bmatrix}
$$
</div>

So if:

<div>
$$
\bs{x}=\begin{bmatrix}
    0.89442719 & 0.4472136
\end{bmatrix}
$$
</div>

$f(\bs{x})$ should be equal to 7. Let's check that's true.

<div>
$$
\begin{align*}
f(\bs{x}) &= 6 x_1^2 + 4 x_1x_2 + 3 x_2^2\\\\
&= 6\times 0.89442719^2 + 4\times 0.89442719\times 0.4472136 + 3 \times 0.4472136^2\\\\
&= 7
\end{align*}
$$
</div>

In the same way, if $\bs{x}=\begin{bmatrix}
    -0.4472136 & 0.89442719
\end{bmatrix}$, $f(\bs{x})$ should be equal to 2.

<div>
$$
\begin{align*}
f(\bs{x}) &= 6 x_1^2 + 4 x_1x_2 + 3 x_2^2\\\\
&= 6\times -0.4472136^2 + 4\times -0.4472136\times 0.89442719 + 3 \times 0.89442719^2\\\\
&= 2
\end{align*}
$$
</div>

# Quadratic form optimization

Depending to the context, optimizing a function means finding its maximum or its minimum. It is for instance widely used to minimize the error of cost functions in machine learning.

Here we will see how eigendecomposition can be used to optimize quadratic functions and why this can be done easily without cross terms. The difficulty is that we want a constrained optimization, that is to find the minimum or the maximum of the function for $f(\bs{x})$ being a unit vector.

### Example 7.

We want to optimize:

<div>
$$
f(\bs{x}) =\bs{x^\text{T}Ax} \textrm{ subject to }||\bs{x}||_2= 1
$$
</div>

In our last example we ended up with:

<div>
$$
f(\bs{x}) = 7y_1^2 + 2y_2^2
$$
</div>

And the constraint of $\bs{x}$ being a unit vector imply:

<div>
$$
||\bs{x}||_2 = 1 \Leftrightarrow x_1^2 + x_2^2 = 1
$$
</div>

We can also show that $\bs{y}$ has to be a unit vector if it is the case for $\bs{x}$. Recall first that $\bs{x}=\bs{Py}$:

<div>
$$
\begin{align*}
||\bs{x}||^2 &= \bs{x^\text{T}x}\\\\
&= (\bs{Py})^\text{T}(\bs{Py})\\\\
&= \bs{P^\text{T}y^\text{T}Py}\\\\
&= \bs{PP^\text{T}y^\text{T}y}\\\\
&= \bs{y^\text{T}y} = ||\bs{y}||^2
\end{align*}
$$
</div>

So $\norm{\bs{x}}^2 = \norm{\bs{y}}^2 = 1$ and thus $y_1^2 + y_2^2 = 1$

Since $y_1^2$ and $y_2^2$ cannot be negative because they are squared values, we can be sure that $2y_2^2\leq7y_2^2$. Hence:

<div>
$$
\begin{align*}
f(\bs{x}) &= 7y_1^2 + 2y_2^2\\\\
&\leq
7y_1^2 + 7y_2^2\\\\
&\leq
7(y_1^2+y_2^2)\\\\
&\leq
7
\end{align*}
$$
</div>

This means that the maximum value of $f(\bs{x})$ is 7.

The same way can lead to find the minimum of $f(\bs{x})$. $7y_1^2\geq2y_1^2$ and:

<div>
$$
\begin{align*}
f(\bs{x}) &= 7y_1^2 + 2y_2^2\\\\
&\geq
2y_1^2 + 2y_2^2\\\\
&\geq
2(y_1^2+y_2^2)\\\\
&\geq
2
\end{align*}
$$
</div>

And the minimum of $f(\bs{x})$ is 2.

### Summary

We can note that the minimum of $f(\bs{x})$ is the minimum eigenvalue of the corresponding matrix $\bs{A}$. Another useful fact is that this value is obtained when $\bs{x}$ takes the value of the corresponding eigenvector (check back the preceding paragraph). In that way, $f(\bs{x})=7$ when $\bs{x}=\begin{bmatrix}0.89442719 & 0.4472136\end{bmatrix}$. This shows how useful are the eigenvalues and eigenvector in this kind of constrained optimization.

## Graphical views

We saw that the quadratic functions $f(\bs{x}) = ax_1^2 +2bx_1x_2 + cx_2^2$ can be represented by the symmetric matrix $\bs{A}$:

<div>
$$
\bs{A}=\begin{bmatrix}
    a & b\\\\
    b & c
\end{bmatrix}
$$
</div>

Graphically, these functions can take one of three general shapes (click on the links to go to the Surface Plotter and move the shapes):

1.[Positive-definite form](https://academo.org/demos/3d-surface-plotter/?expression=x*x%2By*y&xRange=-50%2C+50&yRange=-50%2C+50&resolution=49) | 2.[Negative-definite form](https://academo.org/demos/3d-surface-plotter/?expression=-x*x-y*y&xRange=-50%2C+50&yRange=-50%2C+50&resolution=25) | 3.[Indefinite form](https://academo.org/demos/3d-surface-plotter/?expression=x*x-y*y&xRange=-50%2C+50&yRange=-50%2C+50&resolution=49)
:-------------------------:|:-------------------------:|:-------:
<img src="../../assets/images/2.7/quadratic-functions-positive-definite-form.png" alt="Quadratic function with a positive definite form" title="Quadratic function with a positive definite form"> | <img src="../../assets/images/2.7/quadratic-functions-negative-definite-form.png" alt="Quadratic function with a negative definite form" title="Quadratic function with a negative definite form"> | <img src="../../assets/images/2.7/quadratic-functions-indefinite-form.png" alt="Quadratic function with a indefinite form" title="Quadratic function with a indefinite form">



With the constraints that $\bs{x}$ is a unit vector, the minimum of the function $f(\bs{x})$ corresponds to the smallest eigenvalue and is obtained with its corresponding eigenvector. The maximum corresponds to the biggest eigenvalue and is obtained with its corresponding eigenvector.

# Conclusion

We have seen a lot of things in this chapter. We saw that linear algebra can be used to solve a variety of mathematical problems and more specifically that eigendecomposition is a powerful tool! However, it cannot be used for non square matrices. In the next chapter, we will see the Singular Value Decomposition (SVD) which is another way of decomposing matrices. The advantage of the SVD is that you can use it also with non-square matrices.

# BONUS: visualizing linear transformations

We can see the effect of eigenvectors and eigenvalues in linear transformation. We will see first how linear transformation works. Linear transformation is a mapping between an input vector and an output vector. Different operations like projection or rotation are linear transformations. Every linear transformations can be though as applying a matrix on the input vector. We will see the meaning of this graphically. For that purpose, let's start by drawing the set of unit vectors (they are all vectors with a norm of 1).


```python
t = np.linspace(0, 2*np.pi, 100)
x = np.cos(t)
y = np.sin(t)

plt.figure()
plt.plot(x, y)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()
```


<img src="../../assets/images/2.7/unit-circle.png" width="250" alt="Plot of the unit circle" title="Unit circle">
<em>Unit circle</em>

Then, we will transform each of these points by applying a matrix $\bs{A}$. This is the goal of the function bellow that takes a matrix as input and will draw

- the origin set of unit vectors
- the transformed set of unit vectors
- the eigenvectors
- the eigenvectors scalled by their eigenvalues


```python
def linearTransformation(transformMatrix):
    orange = '#FF9A13'
    blue = '#1190FF'
    # Create original set of unit vectors
    t = np.linspace(0, 2*np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)

    # Calculate eigenvectors and eigenvalues
    eigVecs = np.linalg.eig(transformMatrix)[1]
    eigVals = np.diag(np.linalg.eig(transformMatrix)[0])

    # Create vectors of 0 to store new transformed values
    newX = np.zeros(len(x))
    newY = np.zeros(len(x))
    for i in range(len(x)):
        unitVector_i = np.array([x[i], y[i]])
        # Apply the matrix to the vector
        newXY = transformMatrix.dot(unitVector_i)
        newX[i] = newXY[0]
        newY[i] = newXY[1]

    plotVectors([eigVecs[:,0], eigVecs[:,1]],
                cols=[blue, blue])
    plt.plot(x, y)

    plotVectors([eigVals[0,0]*eigVecs[:,0], eigVals[1,1]*eigVecs[:,1]],
                cols=[orange, orange])
    plt.plot(newX, newY)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()
```


```python
A = np.array([[1,-1], [-1, 4]])
linearTransformation(A)
```


<img src="../../assets/images/2.7/unit-circle-eigenvectors.png" width="250" alt="Plot of the unit circle and its transformation by the matrix A" title="Transformation of the unit circle by the matrix A">
<em>The unit circle and its transformation by the matrix A. The vectors are the eigenvectors of A.</em>

We can see the unit circle in dark blue, the non scaled eigenvectors in light blue, the transformed unit circle in green and the scaled eigenvectors in yellow.

It is worth noting that the eigenvectors are orthogonal here because the matrix is symmetric. Let's try with a non-symmetric matrix:


```python
A = np.array([[1,1], [-1, 4]])
linearTransformation(A)
```


<img src="../../assets/images/2.7/unit-circle-eigenvectors-non-symmetric-matrix.png" width="250" alt="Plot of the unit circle and its transformation by the matrix A in the case of a non symmetric matrix" title="Transformation of the unit circle by the matrix A - Non symmetric matrix">
<em>The unit circle and its transformation by the matrix A. The vectors are the eigenvectors of A (with A non symmetric).</em>

In this case, the eigenvectors are not orthogonal!

# References

## Videos of Gilbert Strang

- [Gilbert Strang, Lec21 MIT - Eigenvalues and eigenvectors](https://www.youtube.com/watch?v=lXNXrLcoerU)

- [Gilbert Strang, Lec 21 MIT, Spring 2005](https://www.youtube.com/watch?v=lXNXrLcoerU)

## Quadratic forms

- [David Lay, University of Colorado, Denver](http://math.ucdenver.edu/~esulliva/LinearAlgebra/SlideShows/07_02.pdf)

- [math.stackexchange QA](https://math.stackexchange.com/questions/2207111/eigendecomposition-optimization-of-quadratic-expressions)

## Eigenvectors

- [Victor Powell and Lewis Lehe - Interactive representation of eigenvectors](http://setosa.io/ev/eigenvectors-and-eigenvalues/)

## Linear transformations

- [Gilbert Strang - Linear transformation](http://ia802205.us.archive.org/18/items/MIT18.06S05_MP4/30.mp4)

- [Linear transformation - demo video](https://www.youtube.com/watch?v=wXCRcnbCsJA)

<span class='notes'>
    Feel free to drop me an email or a comment. The syllabus of this series can be found [in the introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/). All the notebooks can be found on [Github](https://github.com/hadrienj/deepLearningBook-Notes).
</span>

<span class='notes'>
    This content is part of a series following the chapter 2 on linear algebra from the [Deep Learning Book](http://www.deeplearningbook.org/) by Goodfellow, I., Bengio, Y., and Courville, A. (2016). It aims to provide intuitions/drawings/python code on mathematical theories and is constructed as my understanding of these concepts. You can check the syllabus in the [introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/).
</span>


