---
bg: "horse.jpg"
layout: post
mathjax: true
title: Deep Learning Book Series · 2.1 Scalars Vectors Matrices and Tensors
crawlertitle: "Introduction to Scalars Vectors Matrices and Tensors using Python/Numpy examples and drawings"
categories: posts
tags: ['linear-algebra', 'python', 'numpy', 'deep-learning-book']
author: hadrienj
jupyter: https://github.com/hadrienj/deepLearningBook-Notes/blob/master/2.1%20Scalars%2C%20Vectors%2C%20Matrices%20and%20Tensors/2.1%20Scalars%20Vectors%20Matrices%20and%20Tensors.ipynb
date: 2018-03-26 10:00:00
excerpt: This introduction to scalars, vectors, matrices and tensors presents Python/Numpy code and drawings to build a better intuition behind these linear algebra basics.
excerpt-image: <img src="../../assets/images/2.1/scalar-vector-matrix-tensor.png" width="400" alt="An example of a scalar, a vector, a matrix and a tensor" title="Difference between a scalar, a vector, a matrix and a tensor">
  <em>Difference between a scalar, a vector, a matrix and a tensor</em>
deep-learning-book-toc: true
---

# Introduction

This is the first post/notebook of a series following the syllabus of the [linear algebra chapter from the Deep Learning Book](http://www.deeplearningbook.org/contents/linear_algebra.html) by Goodfellow et al.. This work is a collection of thoughts/details/developements/examples I made while reading this chapter. It is designed to help you go through their introduction to linear algebra. For more details about this series and the syllabus, please see the [introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/).

This first chapter is quite light and concerns the basic elements used in linear algebra and their definitions. It also introduces important functions in Python/Numpy that we will use all along this series. It will explain how to create and use vectors and matrices through examples.

# 2.1 Scalars, Vectors, Matrices and Tensors

Let's start with some basic definitions:

<img src="../../assets/images/2.1/scalar-vector-matrix-tensor.png" width="400" alt="An example of a scalar, a vector, a matrix and a tensor" title="Difference between a scalar, a vector, a matrix and a tensor">
<em>Difference between a scalar, a vector, a matrix and a tensor</em>

- A scalar is a single number
- A vector is an array of numbers.

<div>
$$
\bs{x} =\begin{bmatrix}
    x_1 \\\\
    x_2 \\\\
    \cdots \\\\
    x_n
\end{bmatrix}
$$
</div>

- A matrix is a 2-D array

<div>
$$
\bs{A}=
\begin{bmatrix}
    A_{1,1} & A_{1,2} & \cdots & A_{1,n} \\\\
    A_{2,1} & A_{2,2} & \cdots & A_{2,n} \\\\
    \cdots & \cdots & \cdots & \cdots \\\\
    A_{m,1} & A_{m,2} & \cdots & A_{m,n}
\end{bmatrix}
$$
</div>

- A tensor is a $n$-dimensional array with $n>2$

We will follow the conventions used in the [Deep Learning Book](http://www.deeplearningbook.org/):

- scalars are written in lowercase and italics. For instance: $n$
- vectors are written in lowercase, italics and bold type. For instance: $\bs{x}$
- matrices are written in uppercase, italics and bold. For instance: $\bs{X}$

### Example 1.

#### Create a vector with Python and Numpy

*Coding tip*: Unlike the `matrix()` function which necessarily creates $2$-dimensional matrices, you can create $n$-dimensionnal arrays with the `array()` function. The main advantage to use `matrix()` is the useful methods (conjugate transpose, inverse, matrix operations...). We will use the `array()` function in this series.

We will start by creating a vector. This is just a $1$-dimensional array:


```python
x = np.array([1, 2, 3, 4])
x
```

<pre class='output'>
array([1, 2, 3, 4])
</pre>


### Example 2.

#### Create a (3x2) matrix with nested brackets

The `array()` function can also create $2$-dimensional arrays with nested brackets:


```python
A = np.array([[1, 2], [3, 4], [5, 6]])
A
```

<pre class='output'>
array([[1, 2],
       [3, 4],
       [5, 6]])
</pre>


### Shape

The shape of an array (that is to say its dimensions) tells you the number of values for each dimension. For a $2$-dimensional array it will give you the number of rows and the number of columns. Let's find the shape of our preceding $2$-dimensional array `A`. Since `A` is a Numpy array (it was created with the `array()` function) you can access its shape with:


```python
A.shape
```

<pre class='output'>
(3, 2)
</pre>


We can see that $\bs{A}$ has 3 rows and 2 columns.

Let's check the shape of our first vector:


```python
x.shape
```

<pre class='output'>
(4,)
</pre>


As expected, you can see that $\bs{x}$ has only one dimension. The number corresponds to the length of the array:


```python
len(x)
```

<pre class='output'>
4
</pre>


# Transposition

With transposition you can convert a row vector to a column vector and vice versa:

<img src="../../assets/images/2.1/vector-transposition.png" alt="Transposition of a vector" title="Vector transposition" width="200">
<em>Vector transposition</em>

The transpose $\bs{A}^{\text{T}}$ of the matrix $\bs{A}$ corresponds to the mirrored axes. If the matrix is a square matrix (same number of columns and rows):

<img src="../../assets/images/2.1/square-matrix-transposition.png" alt="Transposition of a square matrix" title="Square matrix transposition" width="300">
<em>Square matrix transposition</em>

If the matrix is not square the idea is the same:

<img src="../../assets/images/2.1/non-squared-matrix-transposition.png" alt="Transposition of a square matrix" title="Non square matrix transposition" width="300">
<em>Non-square matrix transposition</em>

The superscript $^\text{T}$ is used for transposed matrices.

<div>
$$
\bs{A}=
\begin{bmatrix}
    A_{1,1} & A_{1,2} \\\\
    A_{2,1} & A_{2,2} \\\\
    A_{3,1} & A_{3,2}
\end{bmatrix}
$$
</div>

<div>
$$
\bs{A}^{\text{T}}=
\begin{bmatrix}
    A_{1,1} & A_{2,1} & A_{3,1} \\\\
    A_{1,2} & A_{2,2} & A_{3,2}
\end{bmatrix}
$$
</div>

The shape ($m \times n$) is inverted and becomes ($n \times m$).

<img src="../../assets/images/2.1/dimensions-transposition-matrix.png" alt="Dimensions of matrix transposition" title="Dimensions of matrix transposition" width="300">
<em>Dimensions of matrix transposition</em>

### Example 3.

#### Create a matrix A and transpose it


```python
A = np.array([[1, 2], [3, 4], [5, 6]])
A
```

<pre class='output'>
array([[1, 2],
       [3, 4],
       [5, 6]])
</pre>



```python
A_t = A.T
A_t
```

<pre class='output'>
array([[1, 3, 5],
       [2, 4, 6]])
</pre>


We can check the dimensions of the matrices:


```python
A.shape
```

<pre class='output'>
(3, 2)
</pre>



```python
A_t.shape
```

<pre class='output'>
(2, 3)
</pre>


We can see that the number of columns becomes the number of rows with transposition and vice versa.

# Addition

<img src="../../assets/images/2.1/matrix-addition.png" alt="Addition of two matrices" title="Addition of two matrices" width="300">
<em>Addition of two matrices</em>

Matrices can be added if they have the same shape:

<div>
$$\bs{A} + \bs{B} = \bs{C}$$
</div>

Each cell of $\bs{A}$ is added to the corresponding cell of $\bs{B}$:

<div>
$$\bs{A}_{i,j} + \bs{B}_{i,j} = \bs{C}_{i,j}$$
</div>

$i$ is the row index and $j$ the column index.

<div>
$$
\begin{bmatrix}
    A_{1,1} & A_{1,2} \\\\
    A_{2,1} & A_{2,2} \\\\
    A_{3,1} & A_{3,2}
\end{bmatrix}+
\begin{bmatrix}
    B_{1,1} & B_{1,2} \\\\
    B_{2,1} & B_{2,2} \\\\
    B_{3,1} & B_{3,2}
\end{bmatrix}=
\begin{bmatrix}
    A_{1,1} + B_{1,1} & A_{1,2} + B_{1,2} \\\\
    A_{2,1} + B_{2,1} & A_{2,2} + B_{2,2} \\\\
    A_{3,1} + B_{3,1} & A_{3,2} + B_{3,2}
\end{bmatrix}
$$
</div>

The shape of $\bs{A}$, $\bs{B}$ and $\bs{C}$ are identical. Let's check that in an example:

### Example 4.

#### Create two matrices A and B and add them

With Numpy you can add matrices just as you would add vectors or scalars.


```python
A = np.array([[1, 2], [3, 4], [5, 6]])
A
```

<pre class='output'>
array([[1, 2],
       [3, 4],
       [5, 6]])
</pre>



```python
B = np.array([[2, 5], [7, 4], [4, 3]])
B
```

<pre class='output'>
array([[2, 5],
       [7, 4],
       [4, 3]])
</pre>



```python
# Add matrices A and B
C = A + B
C
```

<pre class='output'>
array([[ 3,  7],
       [10,  8],
       [ 9,  9]])
</pre>


It is also possible to add a scalar to a matrix. This means adding this scalar to each cell of the matrix.

<div>
$$
\alpha+ \begin{bmatrix}
    A_{1,1} & A_{1,2} \\\\
    A_{2,1} & A_{2,2} \\\\
    A_{3,1} & A_{3,2}
\end{bmatrix}=
\begin{bmatrix}
    \alpha + A_{1,1} & \alpha + A_{1,2} \\\\
    \alpha + A_{2,1} & \alpha + A_{2,2} \\\\
    \alpha + A_{3,1} & \alpha + A_{3,2}
\end{bmatrix}
$$
</div>

### Example 5.

#### Add a scalar to a matrix


```python
A
```

<pre class='output'>
array([[1, 2],
       [3, 4],
       [5, 6]])
</pre>



```python
# Exemple: Add 4 to the matrix A
C = A+4
C
```

<pre class='output'>
array([[ 5,  6],
       [ 7,  8],
       [ 9, 10]])
</pre>


# Broadcasting

Numpy can handle operations on arrays of different shapes. The smaller array will be extended to match the shape of the bigger one. The advantage is that this is done in `C` under the hood (like any vectorized operations in Numpy). Actually, we used broadcasting in the example 5. The scalar was converted in an array of same shape as $\bs{A}$.

Here is another generic example:

<div>
$$
\begin{bmatrix}
    A_{1,1} & A_{1,2} \\\\
    A_{2,1} & A_{2,2} \\\\
    A_{3,1} & A_{3,2}
\end{bmatrix}+
\begin{bmatrix}
    B_{1,1} \\\\
    B_{2,1} \\\\
    B_{3,1}
\end{bmatrix}
$$
</div>

is equivalent to

<div>
$$
\begin{bmatrix}
    A_{1,1} & A_{1,2} \\\\
    A_{2,1} & A_{2,2} \\\\
    A_{3,1} & A_{3,2}
\end{bmatrix}+
\begin{bmatrix}
    B_{1,1} & B_{1,1} \\\\
    B_{2,1} & B_{2,1} \\\\
    B_{3,1} & B_{3,1}
\end{bmatrix}=
\begin{bmatrix}
    A_{1,1} + B_{1,1} & A_{1,2} + B_{1,1} \\\\
    A_{2,1} + B_{2,1} & A_{2,2} + B_{2,1} \\\\
    A_{3,1} + B_{3,1} & A_{3,2} + B_{3,1}
\end{bmatrix}
$$
</div>

where the ($3 \times 1$) matrix is converted to the right shape ($3 \times 2$) by copying the first column. Numpy will do that automatically if the shapes can match.

### Example 6.

#### Add two matrices of different shapes


```python
A = np.array([[1, 2], [3, 4], [5, 6]])
A
```

<pre class='output'>
array([[1, 2],
       [3, 4],
       [5, 6]])
</pre>



```python
B = np.array([[2], [4], [6]])
B
```

<pre class='output'>
array([[2],
       [4],
       [6]])
</pre>



```python
# Broadcasting
C=A+B
C
```

<pre class='output'>
array([[ 3,  4],
       [ 7,  8],
       [11, 12]])
</pre>


You can find basics operations on matrices simply explained [here](https://www.mathsisfun.com/algebra/matrix-introduction.html).

# References

- [Broadcasting in Numpy](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)

- [Discussion on Arrays and matrices](https://stackoverflow.com/questions/4151128/what-are-the-differences-between-numpy-arrays-and-matrices-which-one-should-i-u)

- [Math is fun - Matrix introduction](https://www.mathsisfun.com/algebra/matrix-introduction.html)

<span class='notes'>
    Feel free to drop me an email or a comment. The syllabus of this series can be found [in the introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/). All the notebooks can be found on [Github](https://github.com/hadrienj/deepLearningBook-Notes).
</span>

<span class='notes'>
    This content is part of a series following the chapter 2 on linear algebra from the [Deep Learning Book](http://www.deeplearningbook.org/) by Goodfellow, I., Bengio, Y., and Courville, A. (2016). It aims to provide intuitions/drawings/python code on mathematical theories and is constructed as my understanding of these concepts. You can check the syllabus in the [introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/).
</span>

