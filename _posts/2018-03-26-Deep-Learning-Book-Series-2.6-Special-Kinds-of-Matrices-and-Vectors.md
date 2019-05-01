---
bg: "tools.jpg"
layout: post
mathjax: true
title: Deep Learning Book Series · 2.6 Special Kinds of Matrices and Vectors
crawlertitle: "Deep Learning Book Series · 2.6 Special Kinds of Matrices and Vectors"
categories: posts
tags: ['linear-algebra', 'python', 'numpy', 'deep-learning-book']
author: hadrienj
jupyter: https://github.com/hadrienj/deepLearningBook-Notes/blob/master/2.6%20Special%20Kinds%20of%20Matrices%20and%20Vectors/2.6%20Special%20Kinds%20of%20Matrices%20and%20Vectors.ipynb
date: 2018-03-26 15:00:00
excerpt: We have seen in 2.3 some interesting kind of matrices. We will see other type of vectors and matrices in this chapter. It is not a big chapter but it is important to understand the next ones.
excerpt-image: <img src="../../assets/images/2.6/diagonal-and-symmetric-matrices.png" width="400" alt="Diagonal and symmetric matrices" title="Diagonal and symmetric matrices">
    <em>Example of diagonal and symmetric matrices</em>
---

<span class='notes'>
    This content is part of a series following the chapter 2 on linear algebra from the [Deep Learning Book](http://www.deeplearningbook.org/) by Goodfellow, I., Bengio, Y., and Courville, A. (2016). It aims to provide intuitions/drawings/python code on mathematical theories and is constructed as my understanding of these concepts. You can check the syllabus in the [introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/).
</span>

# Introduction

We have seen in [2.3](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.3-Identity-and-Inverse-Matrices/) some interesting kind of matrices. We will see other type of vectors and matrices in this chapter. It is not a big chapter but it is important to understand the next ones.

# 2.6 Special Kinds of Matrices and Vectors

<img src="../../assets/images/2.6/diagonal-and-symmetric-matrices.png" width="400" alt="Diagonal and symmetric matrices" title="Diagonal and symmetric matrices">
<em>Example of diagonal and symmetric matrices</em>

# Diagonal matrices

<img src="../../assets/images/2.6/diagonal-matrix.png" width="150" alt="Example of a diagonal matrix" title="Diagonal matrix">
<em>Example of a diagonal matrix</em>

A matrix $\bs{A}_{i,j}$ is diagonal if its entries are all zeros except on the diagonal (when $i=j$).

### Example 1.

<div>
$$
\bs{D}=
\begin{bmatrix}
    2 & 0 & 0 & 0\\\\
    0 & 4 & 0 & 0\\\\
    0 & 0 & 3 & 0\\\\
    0 & 0 & 0 & 1
\end{bmatrix}
$$
</div>

### Example 2.

In this case the matrix is also square but there can be non square diagonal matrices.

<div>
$$
\bs{D}=
\begin{bmatrix}
    2 & 0 & 0\\\\
    0 & 4 & 0\\\\
    0 & 0 & 3\\\\
    0 & 0 & 0
\end{bmatrix}
$$
</div>

Or

<div>
$$
\bs{D}=
\begin{bmatrix}
    2 & 0 & 0 & 0\\\\
    0 & 4 & 0 & 0\\\\
    0 & 0 & 3 & 0
\end{bmatrix}
$$
</div>

### Example 3.

The diagonal matrix can be denoted $diag(\bs{v})$ where $\bs{v}$ is the vector containing the diagonal values.

<div>
$$
\bs{D}=
\begin{bmatrix}
    2 & 0 & 0 & 0\\\\
    0 & 4 & 0 & 0\\\\
    0 & 0 & 3 & 0\\\\
    0 & 0 & 0 & 1
\end{bmatrix}
$$
</div>

In this matrix, $\bs{v}$ is the following vector:

<div>
$$
\bs{v}=
\begin{bmatrix}
    2\\\\
    4\\\\
    3\\\\
    1
\end{bmatrix}
$$
</div>

The Numpy function `diag()` can be used to create square diagonal matrices:


```python
v = np.array([2, 4, 3, 1])
np.diag(v)
```

<pre class='output'>
array([[2, 0, 0, 0],
       [0, 4, 0, 0],
       [0, 0, 3, 0],
       [0, 0, 0, 1]])
</pre>

### Example 4.

The mutliplication between a diagonal matrix and a vector is thus just a ponderation of each element of the vector by $v$:

<div>
$$
\bs{D}=
\begin{bmatrix}
    2 & 0 & 0 & 0\\\\
    0 & 4 & 0 & 0\\\\
    0 & 0 & 3 & 0\\\\
    0 & 0 & 0 & 1
\end{bmatrix}
$$
</div>

and

<div>
$$
\bs{x}=
\begin{bmatrix}
    3\\\\
    2\\\\
    2\\\\
    7
\end{bmatrix}
$$
</div>

<div>
$$
\begin{align*}
&\bs{Dx}=
\begin{bmatrix}
    2 & 0 & 0 & 0\\\\
    0 & 4 & 0 & 0\\\\
    0 & 0 & 3 & 0\\\\
    0 & 0 & 0 & 1
\end{bmatrix} \times
\begin{bmatrix}
    3\\\\
    2\\\\
    2\\\\
    7
\end{bmatrix}\\\\
&=\begin{bmatrix}
    2\times3 + 0\times2 + 0\times2 + 0\times7\\\\
    0\times3 + 4\times2 + 0\times2 + 0\times7\\\\
    0\times3 + 0\times2 + 3\times2 + 0\times7\\\\
    0\times3 + 0\times2 + 0\times2 + 1\times7
\end{bmatrix}\\\\
&=
\begin{bmatrix}
    2\times3\\\\
    4\times2\\\\
    3\times2\\\\
    1\times7
\end{bmatrix}
\end{align*}
$$
</div>

### Example 5.

Non square matrices have the same properties:

<div>
$$
\bs{D}=
\begin{bmatrix}
    2 & 0 & 0\\\\
    0 & 4 & 0\\\\
    0 & 0 & 3\\\\
    0 & 0 & 0
\end{bmatrix}
$$
</div>

and

<div>
$$
\bs{x}=
\begin{bmatrix}
    3\\\\
    2\\\\
    2
\end{bmatrix}
$$
</div>

<div>
$$
\bs{Dx}=
\begin{bmatrix}
    2 & 0 & 0\\\\
    0 & 4 & 0\\\\
    0 & 0 & 3\\\\
    0 & 0 & 0
\end{bmatrix}
\times
\begin{bmatrix}
    3\\\\
    2\\\\
    2
\end{bmatrix}
=
\begin{bmatrix}
    2\times3\\\\
    4\times2\\\\
    3\times2\\\\
    0
\end{bmatrix}
$$
</div>

The invert of a square diagonal matrix exists if all entries of the diagonal are non-zeros. If it is the case, the invert is easy to find. Also, the inverse doen't exist if the matrix is non-square.

<div>
$$
\bs{D}=
\begin{bmatrix}
    2 & 0 & 0 & 0\\\\
    0 & 4 & 0 & 0\\\\
    0 & 0 & 3 & 0\\\\
    0 & 0 & 0 & 1
\end{bmatrix}
$$
</div>

<div>
$$
\bs{D}^{-1}=
\begin{bmatrix}
    \frac{1}{2} & 0 & 0 & 0\\\\
    0 & \frac{1}{4} & 0 & 0\\\\
    0 & 0 & \frac{1}{3} & 0\\\\
    0 & 0 & 0 & \frac{1}{1}
\end{bmatrix}
$$
</div>

<div>
$$
\bs{D}=
\begin{bmatrix}
    2 & 0 & 0 & 0\\\\
    0 & 4 & 0 & 0\\\\
    0 & 0 & 3 & 0\\\\
    0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
    \frac{1}{2} & 0 & 0 & 0\\\\
    0 & \frac{1}{4} & 0 & 0\\\\
    0 & 0 & \frac{1}{3} & 0\\\\
    0 & 0 & 0 & \frac{1}{1}
\end{bmatrix}=
\begin{bmatrix}
    1 & 0 & 0 & 0\\\\
    0 & 1 & 0 & 0\\\\
    0 & 0 & 1 & 0\\\\
    0 & 0 & 0 & 1
\end{bmatrix}
$$
</div>

Let's check with Numpy that the multiplication of the matrix with its invert gives us the identity matrix:


```python
A = np.array([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]])
A
```

<pre class='output'>
array([[2, 0, 0, 0],
       [0, 4, 0, 0],
       [0, 0, 3, 0],
       [0, 0, 0, 1]])
</pre>



```python
A_inv = np.array([[1/2., 0, 0, 0], [0, 1/4., 0, 0], [0, 0, 1/3., 0], [0, 0, 0, 1/1.]])
A_inv
```

<pre class='output'>
array([[ 0.5       ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.25      ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.33333333,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
</pre>



```python
A.dot(A_inv)
```

<pre class='output'>
array([[ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]])
</pre>


Great! This gives the identity matrix

# Symmetric matrices

<img src="../../assets/images/2.6/symmetric-matrix.png" width="150" alt="Illustration of a symmetric matrix" title="Symmetric matrix">
<em>Illustration of a symmetric matrix</em>

The matrix $A$ is symmetric if it is equal to its transpose:

<div>
$$
\bs{A} = \bs{A}^\text{T}
$$
</div>

This concerns only square matrices.

### Example 6.

<div>
$$
\bs{A}=
\begin{bmatrix}
    2 & 4 & -1\\\\
    4 & -8 & 0\\\\
    -1 & 0 & 3
\end{bmatrix}
$$
</div>


```python
A = np.array([[2, 4, -1], [4, -8, 0], [-1, 0, 3]])
A
```

<pre class='output'>
array([[ 2,  4, -1],
       [ 4, -8,  0],
       [-1,  0,  3]])
</pre>



```python
A.T
```

<pre class='output'>
array([[ 2,  4, -1],
       [ 4, -8,  0],
       [-1,  0,  3]])
</pre>


# Unit vectors

A unit vector is a vector of length equal to 1. It can be denoted by a letter with a hat: $\hat{u}$

# Orthogonal vectors

Two orthogonal vectors are separated by a 90° angle. The dot product of two orthogonal vectors gives 0.

### Example 7.


```python
x = [0,0,2,2]
y = [0,0,2,-2]

plt.quiver([x[0], y[0]],
           [x[1], y[1]],
           [x[2], y[2]],
           [x[3], y[3]],
           angles='xy', scale_units='xy', scale=1)

plt.xlim(-2, 4)
plt.ylim(-3, 3)
plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')

plt.text(1, 1.5, r'$\vec{u}$', size=18)
plt.text(1.5, -1, r'$\vec{v}$', size=18)

plt.show()
plt.close()
```


<img src="../../assets/images/2.6/orthogonal-vectors.png" width="300" alt="Orthogonal vectors" title="Orthogonal vectors">
<em>Orthogonal vectors</em>

<div>
$$
\bs{x}=
\begin{bmatrix}
    2\\\\
    2
\end{bmatrix}
$$
</div>

and

<div>
$$
\bs{y}=
\begin{bmatrix}
    2\\\\
    -2
\end{bmatrix}
$$
</div>

<div>
$$
\bs{x^\text{T}y}=
\begin{bmatrix}
    2 & 2
\end{bmatrix}
\begin{bmatrix}
    2\\\\
    -2
\end{bmatrix}=
\begin{bmatrix}
    2\times2 + 2\times-2
\end{bmatrix}=0
$$
</div>

In addition, when the norm of orthogonal vectors is the unit norm they are called **orthonormal**.

<span class='pquote'>
    It is impossible to have more than $n$ vectors mutually orthogonal in $\mathbb{R}^n$.
</span>

It is impossible to have more than $n$ vectors mutually orthogonal in $\mathbb{R}^n$. For instance try to draw 3 vectors in a 2-dimensional space ($\mathbb{R}^2$) that are mutually orthogonal...


# Orthogonal matrices

Orthogonal matrices are important because they have interesting properties. A matrix is orthogonal if columns are mutually orthogonal and have a unit norm (orthonormal) and rows are mutually orthonormal and have unit norm.

<img src="../../assets/images/2.6/orthogonal-matrix.png" width="300" alt="Under the hood of an orthogonal matrix" title="Under the hood of an orthogonal matrix">
<em>Under the hood of an orthogonal matrix</em>

<div>
$$
\bs{A}=
\begin{bmatrix}
    A_{1,1} & A_{1,2}\\\\
    A_{2,1} & A_{2,2}
\end{bmatrix}
$$
</div>

This means that

<div>
$$
\begin{bmatrix}
    A_{1,1}\\\\
    A_{2,1}
\end{bmatrix}
$$
</div>

and

<div>
$$
\begin{bmatrix}
    A_{1,2}\\\\
    A_{2,2}
\end{bmatrix}
$$
</div>

are orthogonal vectors and also that the rows

<div>
$$
\begin{bmatrix}
    A_{1,1} & A_{1,2}
\end{bmatrix}
$$
</div>

and

<div>
$$
\begin{bmatrix}
    A_{2,1} & A_{2,2}
\end{bmatrix}
$$
</div>

are orthogonal vectors (cf. above for definition of orthogonal vectors).

## Property 1: $\bs{A^\text{T}A}=\bs{I}$


A orthogonal matrix has this property:

<div>
$$
\bs{A^\text{T}A}=\bs{AA^\text{T}}=\bs{I}
$$
</div>

We can see that this statement is true with the following reasoning:

Let's have the following matrix:

<div>
$$
\bs{A}=\begin{bmatrix}
    a & b\\\\
    c & d
\end{bmatrix}
$$
</div>

and thus

<div>
$$
\bs{A}^\text{T}=\begin{bmatrix}
    a & c\\\\
    b & d
\end{bmatrix}
$$
</div>

Let's do the product:

<div>
$$
\begin{align*}
&\bs{A^\text{T}A}=\begin{bmatrix}
    a & c\\\\
    b & d
\end{bmatrix}
\begin{bmatrix}
    a & b\\\\
    c & d
\end{bmatrix}
=
\begin{bmatrix}
    aa + cc & ab + cd\\\\
    ab + cd & bb + dd
\end{bmatrix}\\\\
&=
\begin{bmatrix}
    a^2 + c^2 & ab + cd\\\\
    ab + cd & b^2 + d^2
\end{bmatrix}
\end{align*}
$$
</div>

We saw in [2.5](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.5-Norms/) that the norm of the vector $\begin{bmatrix}
    a & c
\end{bmatrix}$ is equal to $a^2+c^2$ ($L^2$ or squared $L^2$). In addtion, we saw that the rows of $\bs{A}$ have a unit norm because $\bs{A}$ is orthogonal. This means that $a^2+c^2=1$ and $b^2+d^2=1$. So we now have:

<div>
$$
\bs{A^\text{T}A}=
\begin{bmatrix}
    1 & ab + cd\\\\
    ab + cd & 1
\end{bmatrix}
$$
</div>

Also, $ab+cd$ corresponds to the product of $\begin{bmatrix}
    a & c
\end{bmatrix} and \begin{bmatrix}
    b & d
\end{bmatrix}$:

<div>
$$
\begin{bmatrix}
    a & c
\end{bmatrix}
\begin{bmatrix}
    b\\\\
    d
\end{bmatrix}
=
ab+cd
$$
</div>

And we know that the columns are orthogonal which means that:

<div>
$$
\begin{bmatrix}
    a & c
\end{bmatrix}
\begin{bmatrix}
    b\\\\
    d
\end{bmatrix}=0
$$
</div>

We thus have the identity matrix:

<div>
$$
\bs{A^\text{T}A}=\begin{bmatrix}
    1 & 0\\\\
    0 & 1
\end{bmatrix}
$$
</div>

## Property 2: $\bs{A}^\text{T}=\bs{A}^{-1}$

We can show that if $\bs{A^\text{T}A}=\bs{I}$ then $
\bs{A}^\text{T}=\bs{A}^{-1}$.

If we multiply each side of the equation $\bs{A^\text{T}A}=\bs{I}$ by $\bs{A}^{-1}$ we have:

<div>
$$
(\bs{A^\text{T}A})\bs{A}^{-1}=\bs{I}\bs{A}^{-1}
$$
</div>

Recall from [2.3](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.3-Identity-and-Inverse-Matrices/) that a matrix or vector doesn't change when it is multiplied by the identity matrix. So we have:

<div>
$$
(\bs{A^\text{T}A})\bs{A}^{-1}=\bs{A}^{-1}
$$
</div>

We also saw in [2.2](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.2-Multiplying-Matrices-and-Vectors/) that matrix multiplication is associative so we can remove the parenthesis:

<div>
$$
\bs{A^\text{T}A}\bs{A}^{-1}=\bs{A}^{-1}
$$
</div>

We also know that $\bs{A}\bs{A}^{-1}=\bs{I}$ (see [2.3](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.3-Identity-and-Inverse-Matrices/)) so we can replace:

<div>
$$
\bs{A^\text{T}}\bs{I}=\bs{A}^{-1}
$$
</div>

This shows that

<div>
$$\bs{A}^\text{T}=\bs{A}^{-1}$$
</div>

You can refer to [this question](https://math.stackexchange.com/questions/1936020/why-is-the-inverse-of-an-orthogonal-matrix-equal-to-its-transpose).

### Example 8.

Sine and cosine are convenient to create orthogonal matrices. Let's take the following matrix:

<div>
$$
\bs{A}= 
\begin{bmatrix}
    cos(50) & -sin(50)\\\\
    sin(50) & cos(50)
\end{bmatrix}
$$
</div>


```python
A = np.array([[np.cos(50), -np.sin(50)], [np.sin(50), np.cos(50)]])
A
```

<pre class='output'>
array([[ 0.96496603,  0.26237485],
       [-0.26237485,  0.96496603]])
</pre>



```python
col0 = A[:, 0].reshape(A[:, 0].shape[0], 1)
col1 = A[:, 1].reshape(A[:, 1].shape[0], 1)
row0 = A[0, :].reshape(A[0, :].shape[0], 1)
row1 = A[1, :].reshape(A[1, :].shape[0], 1)
```

Let's check that rows and columns are orthogonal:


```python
col0.T.dot(col1)
```

<pre class='output'>
array([[ 0.]])
</pre>



```python
row0.T.dot(row1)
```

<pre class='output'>
array([[ 0.]])
</pre>


Let's check that:

$$
\bs{A^\text{T}A}=\bs{AA^\text{T}}=\bs{I}
$$

and thus:

$$
\bs{A}^\text{T}=\bs{A}^{-1}
$$


```python
A.T.dot(A)
```

<pre class='output'>
array([[ 1.,  0.],
       [ 0.,  1.]])
</pre>



```python
A.T
```

<pre class='output'>
array([[ 0.96496603, -0.26237485],
       [ 0.26237485,  0.96496603]])
</pre>



```python
numpy.linalg.inv(A)
```

<pre class='output'>
array([[ 0.96496603, -0.26237485],
       [ 0.26237485,  0.96496603]])
</pre>


Everything is correct!

# Conclusion

In this chapter we saw different interesting type of matrices with specific properties. It is generally useful to recall them while we deal with this kind of matrices.

In the next chapter we will saw a central idea in linear algebra: the eigendecomposition. Keep reading!

# References

## Inverse and transpose of orthogonal matrix

- [https://math.stackexchange.com/questions/1936020/why-is-the-inverse-of-an-orthogonal-matrix-equal-to-its-transpose](https://math.stackexchange.com/questions/1936020/why-is-the-inverse-of-an-orthogonal-matrix-equal-to-its-transpose)

- [https://dyinglovegrape.wordpress.com/2010/11/30/the-inverse-of-an-orthogonal-matrix-is-its-transpose/](https://dyinglovegrape.wordpress.com/2010/11/30/the-inverse-of-an-orthogonal-matrix-is-its-transpose/)

<span class='notes'>
    Feel free to drop me an email or a comment. The syllabus of this series can be found [in the introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/). All the notebooks can be found on [Github](https://github.com/hadrienj/deepLearningBook-Notes).
</span>

{% include deep-learning-book-toc.html %}