---
bg: "ecu.jpg"
layout: post
mathjax: true
title: Deep Learning Book Series 2.10 The Trace Operator
crawlertitle: "deep learning machine learning linear algebra python getting started numpy data sciences"
categories: posts
tags: ['linear-algebra', 'python', 'numpy', 'deep-learning-book']
author: hadrienj
jupyter: https://github.com/hadrienj/deepLearningBook-Notes/blob/master/2.10%20The%20Trace%20Operator.ipynb/2.10%20The%20Trace%20Operator.ipynb
date: 2018-03-24 17:00:00
skip_span: true
---

<span class='notes'>
    This content is part of a series following the chapter 2 on linear algebra from the [Deep Learning Book](http://www.deeplearningbook.org/) by Goodfellow, I., Bengio, Y., and Courville, A. (2016). It aims to provide intuitions/drawings/python code on mathematical theories and is constructed as my understanding of these concepts. You can check the syllabus in the [introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/).
</span>

# Introduction

I can assure you that you will read this chapter in 2 minutes! It is nice after the last two chapters that were quite big! We will see what is the Trace of a matrix. It will be needed for the last chapter on the Principal Component Analysis (PCA).

# 2.10 The Trace Operator

<img src="../../assets/images/2.10/trace.png" alt="trace" width="200">


The trace is the sum of all values in the diagonal of a square matrix.

<div>
$$
\bs{A}=
\begin{bmatrix}
    2 & 9 & 8 \\\\
    4 & 7 & 1 \\\\
    8 & 2 & 5
\end{bmatrix}
$$
</div>

<div>
$$
\mathrm{Tr}(\bs{A}) = 2 + 7 + 5 = 14
$$
</div>

Numpy provides the function `trace()` to calculate it:


```python
A = np.array([[2, 9, 8], [4, 7, 1], [8, 2, 5]])
A
```

<pre class='output'>
array([[2, 9, 8],
       [4, 7, 1],
       [8, 2, 5]])
</pre>



```python
A_tr = np.trace(A)
A_tr
```

<pre class='output'>
14
</pre>


GoodFellow et al. explain that the trace can be used to specify the Frobenius norm of a matrix (see [2.5](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.5-Norms/)). The Frobenius norm is the equivalent of the $L^2$ norm for matrices. It is defined by:

<div>
$$
\norm{\bs{A}}_F=\sqrt{\sum_{i,j}A^2_{i,j}}
$$
</div>

Take the square of all elements and sum them. Take the square root of the result. This norm can also be calculated with:

<div>
$$
\norm{\bs{A}}_F=\sqrt{\Tr({\bs{AA}^T})}
$$
</div>

We can check this. The first way to compute the norm can be done with the simple command `np.linalg.norm()`:


```python
np.linalg.norm(A)
```

<pre class='output'>
17.549928774784245
</pre>


The Frobenius norm of $\bs{A}$ is 17.549928774784245.

With the trace the result is identical:


```python
np.sqrt(np.trace(A.dot(A.T)))
```

<pre class='output'>
17.549928774784245
</pre>


Since the transposition of a matrix doesn't change the diagonal, the trace of the matrix is equal to the trace of its transpose:

$
\Tr(\bs{A})=\Tr(\bs{A}^T)
$

## Trace of a product

$
\Tr(\bs{ABC}) = \Tr(\bs{CAB}) = \Tr(\bs{BCA})
$


### Example 1.

Let's see an example of this property.

<div>
$$
\bs{A}=
\begin{bmatrix}
    4 & 12 \\\\
    7 & 6
\end{bmatrix}
$$
</div>

<div>
$$
\bs{B}=
\begin{bmatrix}
    1 & -3 \\\\
    4 & 3
\end{bmatrix}
$$
</div>

<div>
$$
\bs{C}=
\begin{bmatrix}
    6 & 6 \\\\
    2 & 5
\end{bmatrix}
$$
</div>


```python
A = np.array([[4, 12], [7, 6]])
B = np.array([[1, -3], [4, 3]])
C = np.array([[6, 6], [2, 5]])

np.trace(A.dot(B).dot(C))
```

<pre class='output'>
531
</pre>



```python
np.trace(C.dot(A).dot(B))
```

<pre class='output'>
531
</pre>



```python
np.trace(B.dot(C).dot(A))
```

<pre class='output'>
531
</pre>


<div>
$$
\bs{ABC}=
\begin{bmatrix}
    360 & 432 \\\\
    180 & 171
\end{bmatrix}
$$
</div>

<div>
$$
\bs{CAB}=
\begin{bmatrix}
    498 & 126 \\\\
    259 & 33
\end{bmatrix}
$$
</div>

<div>
$$
\bs{BCA}=
\begin{bmatrix}
    -63 & -54 \\\\
    393 & 594
\end{bmatrix}
$$
</div>

<div>
$$
\Tr(\bs{ABC}) = \Tr(\bs{CAB}) = \Tr(\bs{BCA}) =  531
$$
</div>

# References

- [Trace (linear algebra) - Wikipedia](https://en.wikipedia.org/wiki/Trace_(linear_algebra))

- [Numpy Trace operator](https://docs.scipy.org/doc/numpy/reference/generated/numpy.trace.html)

<span class='notes'>
    Feel free to drop me an email or a comment. The syllabus of this series can be found [in the introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/). All the notebooks can be found on [Github](https://github.com/hadrienj/deepLearningBook-Notes).
</span>