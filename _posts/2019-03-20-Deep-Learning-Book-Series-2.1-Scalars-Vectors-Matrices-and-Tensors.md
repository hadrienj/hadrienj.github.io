---
bg: "flower.jpg"
layout: post
mathjax: true
title: Deep Learning Book Series · 2.1 Scalars Vectors Matrices and Tensors
crawlertitle: "deep learning machine learning linear algebra python getting started numpy data sciences"
categories: posts
tags: ['linear-algebra', 'python', 'numpy', 'deep-learning-book']
author: hadrienj
jupyter: https://github.com/hadrienj/deepLearningBook-Notes/tree/master/2.1%20Scalars%2C%20Vectors%2C%20Matrices%20and%20Tensors
comments: true
---

# Introduction

This is the first post/notebook of a series following the syllabus of the [linear algebra chapter from the Deep Learning Book](http://www.deeplearningbook.org/contents/linear_algebra.html) by Goodfellow et al.. This work is a collection of thoughts/details/developements/examples made while reading the Deep Learning Book. It is designed to help you go through their introduction to linear algebra. For more details about this series and check the syllabus, please see the [introduction post](https://hadrienj.github.io/posts/DeepLearningBook-LinearAlgebraSeries/).

This first chapter is quite light and concerns the basic elements used in linear algebra and their definitions. It also introduces important functions in Python/Numpy that we will use all along this series. It will explain how to create and use vectors and matrices through examples.

<span class='notes'>
    This content is part of a series following the chapter 2 on linear algebra from the [Deep Learning Book](http://www.deeplearningbook.org/) by Goodfellow, I., Bengio, Y., and Courville, A. (2016). It aims to provide intuitions/drawings/python code on mathematical theories and is constructed as my understanding of these concepts.
</span>

# Deep Learning Book Series · 2.1 Scalars Vectors Matrices and Tensors

Let's start with some basic definitions:

<img src="../../assets/images/2.1/scalar-tensor.png" width="400" alt="scalar-tensor">

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

- A tensor is a $n$-D array with $n>2$

We will follow the conventions used in the [Deep Learning Book](http://www.deeplearningbook.org/):

- scalars are written in lowercase and italics, like $n$
- vectors are written in lowercase, italics and bold type, like $\bs{x}$
- matrices are written in uppercase, italics and bold, like $\bs{X}$

### Example 1.

#### Create a vector with Python and Numpy

We will start by creating a vector with the `array()` function from Numpy. Unlike the function `matrix()` which necessarily creates 2D arrays, you can create n-dimensionnal arrays with the function `array()`. The main advantage to use `matrix()` is the useful methods (conjugate transpose, inverse, matrix operations...). In the case of a vector, this is a 1D array:


```python
x = np.array([1, 2, 3, 4])
x
```

<pre class='output'>
array([1, 2, 3, 4])
</pre>


### Example 2.

#### Create a (3x2) matrix with nested brackets

The `array()` function can create 2D array with nested brackets:


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

The shape of an array (that is to say its dimensions) tells you the number of value for each dimension. For a 2-D array it will give you the number of rows and the number of columns. Let's find the shape of our preceding 2-D array `A`. Since `A` is a Numpy array (it was created with the command `np.array`) you can access its shape with:


```python
A.shape
```

<pre class='output'>
(3, 2)
</pre>


We can see that $\bs{A}$ has indeed 3 rows and 2 columns.

In the case of our first vector, the shape can be retrieved the same way:


```python
x.shape
```

<pre class='output'>
(4,)
</pre>


You can see that there is only one dimension as expected and thus that number corresponds to the length of the array:


```python
len(x)
```

<pre class='output'>
4
</pre>


# Transposition

With transposition you can convert a row vector to a column vector and vice versa:

<img src="../../assets/images/2.1/transposeVector.png" alt="transposeVector" width="200">

The transpose $\bs{A}^{\text{T}}$ of the matrix $\bs{A}$ corresponds to the mirrored axes. If the matrix is a square matrix (same number of columns and rows):

<img src="../../assets/images/2.1/transposeMatrixSquare.png" alt="transposeMatrixSquare" width="300">

If the matrix is not square the idea is the same:

<img src="../../assets/images/2.1/transposeMatrix.png" alt="transposeMatrix" width="300">


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

<img src="../../assets/images/2.1/transposeMatrixDim.png" alt="transposeMatrixDim" width="300">

### Example 3.

#### Create a matrix A and transpose it


```python
A = np.array([[1, 2], [3, 4], [5, 6]])
A_t = A.T
```

We can check the result of the transposition and the dimensions of the matrices:


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

<img src="../../assets/images/2.1/additionMatrix.png" alt="additionMatrix" width="300">

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
B = np.array([[2, 5], [7, 4], [4, 3]])
# Add matrices A and B
C = A + B
```


```python
print("A \n%s\n" %A)
print("B \n%s\n" %B)
print("C \n%s" %C)
```

    A 
    [[1 2]
     [3 4]
     [5 6]]
    
    B 
    [[2 5]
     [7 4]
     [4 3]]
    
    C 
    [[ 3  7]
     [10  8]
     [ 9  9]]


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
# Exemple: Add 4 to the matrix A
C = A+4
```


```python
print("A \n%s\n" %A)
print("C \n%s" %C)
```

    A 
    [[1 2]
     [3 4]
     [5 6]]
    
    C 
    [[ 5  6]
     [ 7  8]
     [ 9 10]]


# Broadcasting

Numpy can handle operations on arrays of different shapes. The smaller array will be extended to match the shape of the bigger one. The advantage is that this is done in `C` under the hood (like any vectorized operations in Numpy). Actually we used broadcasting in the example 5. The scalar was converted in an array of same shape as $\bs{A}$.

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
B = np.array([[2], [4], [6]])
# Broadcasting
C=A+B
```


```python
print("C \n%s\n" %C)
print("A \n%s\n" %A)
print("B \n%s\n" %B)
```

    C 
    [[ 3  4]
     [ 7  8]
     [11 12]]
    
    A 
    [[1 2]
     [3 4]
     [5 6]]
    
    B 
    [[2]
     [4]
     [6]]
    


You can find basics operations on matrices simply explained [here](https://www.mathsisfun.com/algebra/matrix-introduction.html).

# References

- [Broadcasting in Numpy](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)

- [Discussion on Arrays and matrices](https://stackoverflow.com/questions/4151128/what-are-the-differences-between-numpy-arrays-and-matrices-which-one-should-i-u)

- [Math is fun - Matrix introduction](https://www.mathsisfun.com/algebra/matrix-introduction.html)


```python

```
