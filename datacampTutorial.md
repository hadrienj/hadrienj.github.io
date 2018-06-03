---
bg: "path.jpg"
layout: page
mathjax: true
title: datacampTutorial
---

# Syllabus

- **Norm**
- **Special matrices**
- **Eigendecomposition**
- Singular value decomposition
- Pseudoinverse
- Trace operator
- The determinant
- PCA


# Introduction

This series of tutorials will get you through the main linear algebra concepts usefull in deep learning, machine learning and data science in general. The goal is to get our hands dirty and bind code with mathematical concepts. There is no particular prerequisites but if you are not sure what a matrix is or how to do the dot product, you can see the first posts (1 to 4) of [my series]() on the deep learning book by Ian Goodfellow.

I think that having a practical tutorials on theoretical topics like linear algebra can be useful because writing and reading code is a good way to truly understand mathematical concepts. And above all I think that it can be a lot of fun!

# 1. Norms

# A practical example

This first chapter concerns an important concept for machine learning and deep learning. The norm is extensively used in the context of supervised learning, to evaluate the goodness of a model. Imagine that you want to build a model that predicts the duration of a song. You trained the model with a lot of different songs containing a lot of features (style, instruments, lyrics vs no lyrics etc.). Now you want to know if you model is good to predict the duration of a new song. One way to do it is to look at the difference between true and predicted duration for each observation. Imagine that you have the following results in seconds:

```python
errorModel1 = [22, -4, 2, 7, 6, -36, 12]
```

These differences can be tought of the error of the model. A perfect model would have only 0's while a very bad model would have huge values.

Now imagine that you try another model and you end up with the following differences between predicted and real song durations:

```python
errorModel2 = [14, 9, -13, 19, 8, -21, 4]
```

What can you do if you want to find the best model? A natural way to do it is to take the sum of the absolute value of these errors. The absolute value is used because a negative error (true duration smaller than predicted duration) is also an error. The model with the smaller total error is the better:

```python
totalErrorModel1 = np.sum(errorModel1)
totalErrorModel1
```

<pre class='output'>
9
</pre>

```python
totalErrorModel2 = np.sum(errorModel2)
totalErrorModel2
```

<pre class='output'>
20
</pre>

It looks like the model 1 is far better than the model 2.

Congratulation! What you have just done is calculate the norm of the vector of errors!

You can think of the norm as the **length** of the vector. To have an idea of the graphical representation of this, let's take again our preceding example. The error vectors are multidimensional: there is one dimension per observation. In our simple case, there is 7 observations so there is 7 dimensions. It is still quite hard to represent 7 dimensions so let's again simplify the example and keep only 2 observations:

```python
errorModel1 = [22, -4]
errorModel2 = [14, 9]
```

Now we can represent these vectors considering that the first element of the array is the x-coordinate and the second element is the y-coordinate. We will use a custom function defined bellow to plot the vectors.

```python
plotVectors([errorModel1, errorModel2], [sns.color_palette()[0], sns.color_palette()[1]])

plt.xlim(-1, 25)
plt.ylim(-5, 10)
```

<img src="../../assets/images/datacampTutorial/error-models.png" width="300" alt="Plots of the error vectors" title="Error vectors">
<em>The error vectors</em>

So we have two vectors and the way to see which one is the smaller one (hence which model has the smaller error) is to take the sum of each coordinate. It is what we have done earlier. Actually we need to take the absolute value of each coordinate because we don't want to cancel out coordinates.

# How to plot vectors with Python and Matplotlib

We just used a custom function to plot vectors in Python. Let's see how it works:

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
    plt.axvline(x=0, color='#A9A9A9', zorder=0)
    plt.axhline(y=0, color='#A9A9A9', zorder=0)

    for i in range(len(vecs)):
        if (isinstance(alpha, list)):
            alpha_i = alpha[i]
        else:
            alpha_i = alpha
        if (len(vecs[i])==2):
            x = np.concatenate([[0,0],vecs[i]])
        elif (len(vecs[i])==4):
            x = vecs[i]
        plt.quiver([x[0]],
                   [x[1]],
                   [x[2]],
                   [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=cols[i],
                  alpha=alpha_i)
```

It takes an array of vectors that we want to plot (`vecs`) and their colors as input (`cols`). If only 2 dimensions are specified in the vector, it starts at (0, 0). Under the hood, we iterate on this array of vectors and use `plt.quiver()` to plot them.

# Norm functions: definitions

We saw one way to calculate the length of the vector but the norm can be any function that maps a vector to a positive value. Different functions can be used and we will see few examples. These functions can be called norms if they are characterized by the **following properties**:

- Norms are non-negative values. If you think of the norms as a length, you easily see why it can't be negative.

- Norms are $0$ if and only if the vector is a zero vector

- Norms respect the triangle inequity. See bellow.

- $\norm{\bs{k}\cdot \bs{u}}=\norm{\bs{k}}\cdot\norm{\bs{u}}$. The norm of a vector multiplied by a scalar is equal to the absolute value of this scalar multiplied by the norm of the vector.

It is usually written with two horizontal bars: $\norm{\bs{x}}$

# The triangle inequity

The norm of the sum of some vectors is less than or equal the sum of the norms of these vectors.

<div>
$$
\norm{\bs{u}+\bs{v}} \leq \norm{\bs{u}}+\norm{\bs{v}}
$$
</div>

### Example 1.

To show what this means, we will take two vectors containing each two elements (usefull to be represented as x and y coordinates). Our vectors are:

<div>
$$
\bs{u}=
\begin{bmatrix}
    1 & 6
\end{bmatrix}
$$
</div>

```python
u = np.array([1, 6])
u
```

<pre class='output'>
array([1, 6])
</pre>

and

<div>
$$
\bs{v}=
\begin{bmatrix}
    4 & 2
\end{bmatrix}
$$
</div>

```python
v = np.array([4, 2])
v
```

<pre class='output'>
array([4, 2])
</pre>

Let's compare:

<div>
$$
\norm{\bs{u}+\bs{v}}
$$
</div>

and:

<div>
$$
\norm{\bs{u}}+\norm{\bs{v}}
$$
</div>

using the $L^2$ norm. The $L^2$ norm can be calculated with the Numpy function `np.linalg.norm()`.

<div>
$$
\norm{\bs{u}+\bs{v}} = \sqrt{(1+4)^2+(6+2)^2} = \sqrt{89} \approx 9.43
$$
</div>

```python
np.linalg.norm(u+v)
```

<pre class='output'>
9.4339811320566032
</pre>

and

<div>
$$
\norm{\bs{u}}+\norm{\bs{v}} = \sqrt{1^2+6^2}+\sqrt{4^2+2^2} = \sqrt{37}+\sqrt{20} \approx 10.55
$$
</div>

```python
np.linalg.norm(u)+np.linalg.norm(v)
```

<pre class='output'>
10.554898485297798
</pre>

We can see that the triangle inequity is respected since:

<div>
$$
\norm{\bs{u}+\bs{v}} \leq \norm{\bs{u}}+\norm{\bs{v}}
$$
</div>


**Graphical explanation**

You will see that the graphical representation of this theorem makes it quite trivial. We will plot the vectors $\bs{u}$, $\bs{v}$ and $\bs{u}+\bs{v}$.

```python
u = np.array([0,0,1,6])
v = np.array([0,0,4,2])
w = u+v

u_bis = [u[2], u[3], v[2],v[3]]


plotVectors([u, u_bis, w],
            [sns.color_palette()[0],
            sns.color_palette()[1],
            sns.color_palette()[2]])

plt.xlim(-2, 6)
plt.ylim(-2, 9)

plt.text(-1, 3.5, r'$||\vec{u}||$', color=sns.color_palette()[0], size=20)
plt.text(2.5, 7.5, r'$||\vec{v}||$', color=sns.color_palette()[1], size=20)
plt.text(2, 2, r'$||\vec{u}+\vec{v}||$', color=sns.color_palette()[2], size=20)

plt.show()
plt.close()
```


<img src="../../assets/images/2.5/triangle-inequity.png" width="300" alt="Vector illustration of the triangle inequity" title="Triangle inequity">
<em>Vector illustration of the triangle inequity</em>

The length of $\bs{u}$ plus the length of $\bs{v}$ is larger than the length of the vector $\bs{u}+\bs{b}$. Geometrically, this simply means that the shortest path between two points is a line.

# P-norms: general rules

We have seen the conditions required by the function to be called norm. This means that there are multiple functions that can be used as norms. We will see later the pros and cons of these different norms. We call p-norm the following category of functions that depends on $p$:

<div>
$$
\norm{\bs{x}}_p=(\sum_i|\bs{x}_i|^p)^{1/p}
$$
</div>

Let's dive into this equation step by step. We can see that there is a sum of elements so we can think of it as an iteration over the $i$ elements:

1. $\vert\bs{x}_i\vert$ Calculate the absolute value of the $i$th element
2. $\vert\bs{x}_i\vert^p$ Take its power $p$
3. $\sum_i\vert\bs{x}_i\vert^p$ Sum all these powered absolute values
4. $(\sum_i\vert\bs{x}_i\vert^p)^{1/p}$ Take the power $\frac{1}{p}$ of this result

This will be clear with examples using these widely used $p$-norms.

# The $L^0$ norm

If $p=0$, the formula becomes:

<div>
$$
\norm{\bs{x}}_0=(\sum_i|\bs{x}_i|^0)^{1/0}
$$
</div>

Let's see what it means. Using the power $0$ with absolute values will get you a $1$ for every non-$0$ values and a $0$ for $0$.

Therefore this norm corresponds to the number of non-zero elements in the vector. It is not really a norm because if you multiply the vector by $\alpha$, this number is the same (rule 4 above).

# The $L^1$ norm

$p=1$ so this norm is simply the sum of the absolute values:

<div>
$$
\norm{\bs{x}}_1=(\sum_i|\bs{x}_i|^1)^{1/1}=\sum_{i} |\bs{x}_i|
$$
</div>

# The Euclidean norm ($L^2$ norm)

The Euclidean norm is the $p$-norm with $p=2$. This may be the more used norm with the squared $L^2$ norm.

<div>
$$
\norm{\bs{x}}_2=(\sum_i \bs{x}_i^2)^{1/2}=\sqrt{\sum_i \bs{x}_i^2}
$$
</div>

We can note that the absolute value is not needed anymore since $x$ is squared.

Let's see an example of this norm:

### Example 2.

Graphically, the Euclidean norm corresponds to the length of the vector from the origin to the point obtained by linear combination (Pythagorean theorem). We will see an example in 2 dimensions: the vector $\bs{u}$ has two values corresponding to the $x$-coordinate and to the $y$-coordinate. If you plot the point with these coordinates and draw a vector from the origin to this point, the $L^2$ norm will be the length of this vector.

For instance:

<div>
$$
\bs{u}=
\begin{bmatrix}
    3 \\\\
    4
\end{bmatrix}
$$
</div>

Let's start by calculating the norm with the formula:

<div>
$$
\begin{align*}
\norm{\bs{u}}_2 &=\sqrt{|3|^2+|4|^2}\\\\
&=\sqrt{25}\\\\
&=5
\end{align*}
$$
</div>


So the $L^2$ norm is $5$.

By the way, the $L^2$ norm can be calculated with the `linalg.norm()` function from Numpy:


```python
np.linalg.norm([3, 4])
```

<pre class='output'>
5.0
</pre>


Here is the graphical representation of the vector:


```python
u = np.array([3, 4])

plt.ylim(-1, 5)
plt.xlim(-1, 5)
plotVectors([u], [sns.color_palette()[0]])
```


<img src="../../assets/images/datacampTutorial/example-l2-norm-vectors.png" width="300" alt="Vector illustration of the l2 norm" title="Vectors of the example 2.">
<em>Vector from the example 2.</em>

We can see that the vector goes from the origin (0, 0) to (3, 4) and that its length is 5.

In this case, the vector is in a 2-dimensional space but this stands also for more dimensions.

<div>
$$
\bs{u}=
\begin{bmatrix}
    u_1\\\\
    u_2\\\\
    \cdots \\\\
    u_n
\end{bmatrix}
$$
</div>

<div>
$$
\norm{\bs{u}}_2 = \sqrt{u_1^2+u_2^2+\cdots+u_n^2}
$$
</div>


# The squared Euclidean norm (squared $L^2$ norm)

<div>
$$
\norm{\bs{u}}_2^2 = (\sqrt{\sum_i \bs{x}_i^2})^2 = \sum_i\bs{x}_i^2
$$
</div>


The squared $L^2$ norm is convenient because it removes the square root and we end up with the simple sum of every squared values of the vector.

The squared Euclidean norm is widely used in machine learning partly because it can be calculated with the vector operation $\bs{x}^\text{T}\bs{x}$. There can be performance gain due to optimization. See [here](https://softwareengineering.stackexchange.com/questions/312445/why-does-expressing-calculations-as-matrix-multiplications-make-them-faster) and [here](https://www.quora.com/What-makes-vector-operations-faster-than-for-loops) for more details.

### Example 3.

We will see in this example that the squared Euclidean norm can be calculated with vectorized operations. Let's start with a vector $\bs{x}$:

<div>
$$
\bs{x}=
\begin{bmatrix}
    2 \\\\
    5 \\\\
    3 \\\\
    3
\end{bmatrix}
$$
</div>

Now let's take the transpose of this vector. This will just convert the initial column vector to a row vector:

<div>
$$
\bs{x}^\text{T}=
\begin{bmatrix}
    2 & 5 & 3 & 3
\end{bmatrix}
$$
</div>

The dot product of $\bs{x}$ and $\bs{x}^\text{T}$ (see [here](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.2-Multiplying-Matrices-and-Vectors/) if you need some reminder about the dot product) corresponds actually to the multiplication of each element by itself:

<div>
$$
\begin{align*}
\bs{x}^\text{T}\bs{x}&=
\begin{bmatrix}
    2 & 5 & 3 & 3
\end{bmatrix} \times
\begin{bmatrix}
    2 \\\\
    5 \\\\
    3 \\\\
    3
\end{bmatrix}\\\\
&= 2\times 2 + 5\times 5 + 3\times 3 + 3\times 3= 47
\end{align*}
$$
</div>

This is exactly the definition of the squared Euclidean norm!

As usual we will use code to check the process. First let's create our Numpy vector $\bs{x}$:


```python
x = np.array([[2], [5], [3], [3]])
x
```

<pre class='output'>
array([[2],
       [5],
       [3],
       [3]])
</pre>

Then, we will calculate the transpose of $\bs{x}$ with the `T` method of Numpy objects and calculate its dot product with $\bs{x}$. Remember (and try it to convince yourself) that the order the vector in the dot product matters. All of this can be done in one line by chaining the Numpy functions:

```python
euclideanNorm = x.T.dot(x)
euclideanNorm
```

<pre class='output'>
array([[47]])
</pre>

It should be our squared Euclidean norm! Let's calculate it from the $L^2$ norm to check it:


```python
np.linalg.norm(x)**2
```

<pre class='output'>
47.0
</pre>


It works! The possibility to use a vectorized operation is a huge advantage over the other norms.

## Derivative of the squared $L^2$ norm

We have seen that the norms can be used to evaluate the goodness of a model by summarizing the vectors of errors (see above).

Now we want to go a step further and know how we can change the parameters of our model to reduce the overall error. To do that we can use a cost function that associates the error of the model in function of the parameters values. The gradient descent algorithm can be used to find the minimum of this function. The gradient descent is done by calculating the derivatives according to each parameter (partial derivatives = gradients). This is why this is crucial to be able to calculate the derivative efficiently.

Indeed, a big advantage of the squared $L^2$ norm is that its partial derivative is easily computed. Let's have the following vector:

<div>
$$
\bs{u}=
\begin{bmatrix}
    u_1\\\\
    u_2\\\\
    \cdots \\\\
    u_n
\end{bmatrix}
$$
</div>

We have seen that its squared $L^2$ norm is calculated with:

<div>
$$
\norm{\bs{u}}_2^2 = u_1^2+u_2^2+\cdots+u_n^2
$$
</div>

Then, to calculate the partial derivatives, we consider all other variables as constant. For instance, the partial derivative according to $u_1$ is the derivative of $u_1^2+a$. For this reason we have the following partial derivatives:

<div>
$$
\begin{cases}
\dfrac{d\norm{\bs{u}}_2^2}{du_1} = 2u_1\\\\
\dfrac{d\norm{\bs{u}}_2^2}{du_2} = 2u_2\\\\
\cdots\\\\
\dfrac{d\norm{\bs{u}}_2^2}{du_n} = 2u_n
\end{cases}
$$
</div>

What is great about the gradients of squared $L^2$ norm is that the derivatives do not depend on the other variables. We will see that it is not the case of the $L^2$ norm.

## Derivative of the $L^2$ norm

In the case of the $L^2$ norm, the derivative is more complicated and takes every elements of the vector into account. Let's take the last vector $\bs{u}$ as an example. The $L^2$ norm is:

<div>
$$
\norm{\bs{u}}_2 = \sqrt{(u_1^2+u_2^2+\cdots+u_n^2)} = (u_1^2+u_2^2+\cdots+u_n^2)^{\frac{1}{2}}
$$
</div>

Let's calculate the derivative of it according to $u_1$:

<div>
$$
\begin{align*}
\dfrac{d\norm{\bs{u}}_2}{du_1} &=
\dfrac{1}{2}(u_1^2+u_2^2+\cdots+u_n^2)^{\frac{1}{2}-1}\cdot
\dfrac{d}{du_1}(u_1^2+u_2^2+\cdots+u_n^2)\\\\
&=\dfrac{1}{2}(u_1^2+u_2^2+\cdots+u_n^2)^{-\frac{1}{2}}\cdot
\dfrac{d}{du_1}(u_1^2+u_2^2+\cdots+u_n^2)\\\\
&=\dfrac{1}{2}\cdot\dfrac{1}{(u_1^2+u_2^2+\cdots+u_n^2)^{\frac{1}{2}}}\cdot
\dfrac{d}{du_1}(u_1^2+u_2^2+\cdots+u_n^2)\\\\
&=\dfrac{1}{2}\cdot\dfrac{1}{(u_1^2+u_2^2+\cdots+u_n^2)^{\frac{1}{2}}}\cdot
2\cdot u_1\\\\
&=\dfrac{u_1}{\sqrt{(u_1^2+u_2^2+\cdots+u_n^2)}}\\\\
\end{align*}
$$
</div>

We can see that the partial derivative of $u_1$ still contains $u_2...u_n$. The other gradients follow the same structure:

<div>
$$
\begin{cases}
\dfrac{d\norm{\bs{u}}_2}{du_1} = \dfrac{u_1}{\sqrt{(u_1^2+u_2^2+\cdots+u_n^2)}}\\\\
\dfrac{d\norm{\bs{u}}_2}{du_2} = \dfrac{u_2}{\sqrt{(u_1^2+u_2^2+\cdots+u_n^2)}}\\\\
\cdots\\\\
\dfrac{d\norm{\bs{u}}_2}{du_n} = \dfrac{u_n}{\sqrt{(u_1^2+u_2^2+\cdots+u_n^2)}}\\\\
\end{cases}
$$
</div>

# Other considerations

The squared $L^2$ norm is great but one problem with it is that it hardly discriminates between 0 and small values because the function increases slowly.

We can see this by graphically comparing the squared $L^2$ norm with the $L^2$ norm. The $z$-axis corresponds to the value of the norm and the $x$- and $y$-axis correspond to two parameters. The same thing is true with more than 2 dimensions but it would be hard to visualize it.

$L^2$ norm:

<img src="../../assets/images/2.5/l2-norm.png" width="500" alt="Representation of the L2 norm" title="The L2 norm">
<em>The L2 norm</em>

Squared $L^2$ norm:

<img src="../../assets/images/2.5/squared-l2-norm.png" width="500" alt="Representation of the squared L2 norm" title="The squared L2 norm">
<em>The squared L2 norm</em>

For comparison, here is the $L^1$ norm:

<img src="../../assets/images/2.5/l1-norm.png" alt="Representation of the L1 norm" title="The L1 norm" width="500">
<em>The L1 norm</em>

These plots have been done with the help of this [website](https://academo.org/demos/3d-surface-plotter/). Go and plot these norms if you need to move them in order to catch their shape.

# The max norm

It is the $L^\infty$ norm and corresponds to the absolute value of the greatest element of the vector.

<div>
$$
\norm{\bs{x}}_\infty = \max\limits_i|x_i|
$$
</div>

# Matrix norms: the Frobenius norm

<div>
$$
\norm{\bs{A}}_F=\sqrt{\sum_{i,j}A^2_{i,j}}
$$
</div>

This is equivalent to take the $L^2$ norm of the matrix after flattening.

The same Numpy function can be use:


```python
A = np.array([[1, 2], [6, 4], [3, 2]])
A
```

<pre class='output'>
array([[1, 2],
       [6, 4],
       [3, 2]])
</pre>



```python
np.linalg.norm(A)
```

<pre class='output'>
8.3666002653407556
</pre>


# Expression of the dot product with norms

<div>
$$
\bs{x}^\text{T}\bs{y} = \norm{\bs{x}}_2\cdot\norm{\bs{y}}_2\cos\theta
$$
</div>

The dot product between the vectors $\bs{x}$ and $\bs{y}$ can be retrieved with the $L^2$ norms of these vectors. $\theta$ is the angle between the two vectors.

### Example 4.

Let's take two vectors in 2 dimensions:

<div>
$$
\bs{x}=
\begin{bmatrix}
    0 \\\\
    2
\end{bmatrix}
$$
</div>

and

<div>
$$
\bs{y}=
\begin{bmatrix}
    2 \\\\
    2
\end{bmatrix}
$$
</div>

The following plot shows their graphical representation:

```python
x = [0,0,0,2]
y = [0,0,2,2]

plotVectors([x, y], [sns.color_palette()[0], sns.color_palette()[1]])

plt.xlim(-1, 3)
plt.ylim(-1, 3)

plt.text(-0.5, 1, r'$\vec{x}$', size=18, color=sns.color_palette()[0])
plt.text(1.5, 0.5, r'$\vec{y}$', size=18, color=sns.color_palette()[1])
```


<img src="../../assets/images/datacampTutorial/dot-product-expression-norms.png" width="300" alt="Expression of the dot product with norms" title="Expression of the dot product with norms">
<em>Expression of the dot product with norms</em>

We took this example for its simplicity. As we can see, the angle $\theta$ is equal to 45°.

First, let's calculate the dot product of the vectors:

<div>
$$
\bs{x^\text{T}y}=
\begin{bmatrix}
    0 & 2
\end{bmatrix} \cdot
\begin{bmatrix}
    2 \\\\
    2
\end{bmatrix} =
0\times2+2\times2 = 4
$$
</div>

```python
x = np.array([0, 2])
y = np.array([2, 2])

x.dot(y)
```

<pre class='output'>
4
</pre>

Great! And now let's calculate their norms:

<div>
$$
\norm{\bs{x}}_2=\sqrt{0^2+2^2}=\sqrt{4}=2
$$
</div>

and

<div>
$$
\norm{\bs{y}}_2=\sqrt{2^2+2^2}=\sqrt{8}
$$
</div>

So with the above formula, we have:

<div>
$$
2\times\sqrt{8}\times cos(45)=4
$$
</div>

This is the same results as with the dot product. Here are the operations using numpy. Just note that we use the function `deg2rad` from Numpy because `np.cos` takes the angle in radian so we have to do the conversion.


```python
# Note: np.cos take the angle in radian
np.cos(np.deg2rad(45))*2*np.sqrt(8)
```

<pre class='output'>
4.0000000000000009
</pre>

Nice!

# 2. Special Kinds of Matrices and Vectors

We will see interesting types of vectors and matrices in this chapter:

- Diagonal matrices
- Symmetric matrices
- Unit vectors
- Orthogonal vectors
- Orthogonal matrices

<img src="../../assets/images/2.6/diagonal-and-symmetric-matrices.png" width="400" alt="Diagonal and symmetric matrices" title="Diagonal and symmetric matrices">
<em>Example of diagonal and symmetric matrices</em>

# Diagonal matrices

<img src="../../assets/images/2.6/diagonal-matrix.png" width="150" alt="Example of a diagonal matrix" title="Diagonal matrix">
<em>Example of a diagonal matrix</em>

If you look at the above matrix, you can see that it contains two kind of values: zeros in light blue and non-zeros in dark blue. A matrix $\bs{A}_{i,j}$ is diagonal if its entries are all zeros except on the diagonal (when $i=j$).

### Example 1.

<div>
$$
\bs{D}=
\begin{bmatrix}
    2 & 0 & 0 & 0\\\\
    0 & 4 & 0 & 0\\\\
    0 & 0 & -3 & 0\\\\
    0 & 0 & 0 & 1
\end{bmatrix}
$$
</div>

In this case the matrix is also square (there is the same number of rows and columns). We can see that all the non diagonal values are $0$.

### Example 2.

There also can be non-square diagonal matrices. For instance with more rows than columns:

<div>
$$
\bs{D}=
\begin{bmatrix}
    -2 & 0 & 0\\\\
    0 & 4 & 0\\\\
    0 & 0 & 3\\\\
    0 & 0 & 0
\end{bmatrix}
$$
</div>

Or with more columns than rows:

<div>
$$
\bs{D}=
\begin{bmatrix}
    -2 & 0 & 0 & 0\\\\
    0 & 4 & 0 & 0\\\\
    0 & 0 & 3 & 0
\end{bmatrix}
$$
</div>

### Example 3.

A diagonal matrix can be denoted $diag(\bs{v})$ where $\bs{v}$ is the vector containing the diagonal values.

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

The mutliplication between a diagonal matrix and a vector is thus just a ponderation of each element of the vector by $\bs{v}$ (which is the vector containing the diagonal value of $\bs{D}$. Let's have the following matrix $\bs{D}$:

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

and the following vector $\bs{x}$:

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

The dot product of $\bs{D}$ with $\bs{x}$ gives:

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

It corresponds to a ponderation of the vector $\bs{v}$ by $\bs{x}$.

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

## Invert of a square diagonal matrix

If you need some details about the inversion of a matrix checkout my blog post [here](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.3-Identity-and-Inverse-Matrices/). The invert of a square diagonal matrix exists if all entries of the diagonal are non-zeros. If it is the case, the invert is easy to find. Also, the inverse doen't exist if the matrix is non-square. Having the following matrix $\bs{D}$:

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

```python
A = np.array([[2, 0, 0, 0],
  [0, 4, 0, 0],
  [0, 0, 3, 0],
  [0, 0, 0, 1]])
A
```

<pre class='output'>
array([[2, 0, 0, 0],
       [0, 4, 0, 0],
       [0, 0, 3, 0],
       [0, 0, 0, 1]])
</pre>

The inverse $\bs{D}^{-1}$ is:

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

```python
A_inv = np.linalg.inv(A)
A_inv
```

<pre class='output'>
array([[ 0.5       ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.25      ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.33333333,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
</pre>

We can check that the dot product of $\bs{D}$ and $\bs{D}^{-1}$ gives us the identity matrix (more details about the identity matrix [here](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.3-Identity-and-Inverse-Matrices/)).

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

```python
A.dot(A_inv)
```

<pre class='output'>
array([[ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]])
</pre>

Great!

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

Here is an example of a symmetric matrix:

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

We will check that it corresponds to its transpose:

```python
A.T
```

<pre class='output'>
array([[ 2,  4, -1],
       [ 4, -8,  0],
       [-1,  0,  3]])
</pre>

Hooray!

# Unit vectors

This one is simple! A unit vector is a vector of length equal to 1. It can be denoted by a letter with a hat: $\hat{u}$

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

Let's calculate their dot product:

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
<em>An orthogonal matrix</em>

For instance, let's have the following matrix:

<div>
$$
\bs{A}=
\begin{bmatrix}
    A_{1,1} & A_{1,2}\\\\
    A_{2,1} & A_{2,2}
\end{bmatrix}
$$
</div>

It is orthogonal if the column vectors:

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

are orthogonal and if the row vectors:

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

are also orthogonal (cf. above for definition of orthogonal vectors).

## Property 1: $\bs{A^\text{T}A}=\bs{I}$

<div>
$$
\bs{A^\text{T}A}=\bs{AA^\text{T}}=\bs{I}
$$
</div>

We will see that the dot product of an orthogonal matrix with its transpose gives the identity matrix. Let's see why!

Let's have the following matrix:

<div>
$$
\bs{A}=\begin{bmatrix}
    a & b\\\\
    c & d
\end{bmatrix}
$$
</div>

The transpose of this matrix is:

<div>
$$
\bs{A}^\text{T}=\begin{bmatrix}
    a & c\\\\
    b & d
\end{bmatrix}
$$
</div>

Let's do their product:

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

We know from the norm chapter the norm of the vector $\begin{bmatrix}
    a & c
\end{bmatrix}$ is equal to $a^2+c^2$ (squared $L^2$).

In addition, we just saw that the rows of $\bs{A}$ have a unit norm because $\bs{A}$ is orthogonal. This means that the length of the vector $\begin{bmatrix}
    a & c
\end{bmatrix}$ is equal to $1$ and thus that $a^2+c^2=1$ and $b^2+d^2=1$. So we now have:

<div>
$$
\bs{A^\text{T}A}=
\begin{bmatrix}
    1 & ab + cd\\\\
    ab + cd & 1
\end{bmatrix}
$$
</div>

Also, $ab+cd$ corresponds to the dot product of $\begin{bmatrix}
    a & c
\end{bmatrix} and \begin{bmatrix}
    b & d
\end{bmatrix}$. Look:

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

The inverse of an orthogonal matrix is equal to its transpose. We have just seen that $\bs{A^\text{T}A}=\bs{I}$ and we will see that it is related to the fact that $\bs{A}^\text{T}=\bs{A}^{-1}$.

The first step is to multiply each side of the equation $\bs{A^\text{T}A}=\bs{I}$ by $\bs{A}^{-1}$. We have:

<div>
$$
(\bs{A^\text{T}A})\bs{A}^{-1}=\bs{I}\bs{A}^{-1}
$$
</div>

Recall that a matrix or a vector doesn't change when it is multiplied by the identity matrix (see [here](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.3-Identity-and-Inverse-Matrices/)). So we have:

<div>
$$
(\bs{A^\text{T}A})\bs{A}^{-1}=\bs{A}^{-1}
$$
</div>

We also know that matrix multiplication is associative so we can remove the parenthesis:

<div>
$$
\bs{A^\text{T}A}\bs{A}^{-1}=\bs{A}^{-1}
$$
</div>

We also know from above that $\bs{A}\bs{A}^{-1}=\bs{I}$ so we can replace:

<div>
$$
\bs{A^\text{T}}\bs{I}=\bs{A}^{-1}
$$
</div>

This shows that

<div>
$$\bs{A}^\text{T}=\bs{A}^{-1}$$
</div>

You can refer to [this question](https://math.stackexchange.com/questions/1936020/why-is-the-inverse-of-an-orthogonal-matrix-equal-to-its-transpose) for more details.

### Example 8.

Sine and cosine are convenient to create orthogonal matrices. Let's create the following matrix for our example:

<div>
$$
\bs{A}=
\begin{bmatrix}
    cos(50) & -sin(50)\\\\
    sin(50) & cos(50)
\end{bmatrix}
$$
</div>

In Python, we will create the matrix like that:


```python
angle = np.deg2rad(50)
A = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
A
```

<pre class='output'>
array([[ 0.64278761, -0.76604444],
       [ 0.76604444,  0.64278761]])
</pre>

Then, for simplicity we will create variables for each row and each column:

```python
col0 = A[:, 0].reshape(A[:, 0].shape[0], 1)
col1 = A[:, 1].reshape(A[:, 1].shape[0], 1)
row0 = A[0, :].reshape(A[0, :].shape[0], 1)
row1 = A[1, :].reshape(A[1, :].shape[0], 1)
```

Let's now check that rows and columns are orthogonal. If it is the case, the dot product of the rows between them and the dot product of the columns between them have to be equal to $0$:

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


We can also check that:

$$
\bs{A^\text{T}A}=\bs{AA^\text{T}}=\bs{I}
$$

```python
A.T.dot(A)
```

<pre class='output'>
array([[ 1.,  0.],
       [ 0.,  1.]])
</pre>

and that:

$$
\bs{A}^\text{T}=\bs{A}^{-1}
$$


```python
A.T
```

<pre class='output'>
array([[ 0.64278761,  0.76604444],
       [-0.76604444,  0.64278761]])
</pre>


```python
numpy.linalg.inv(A)
```

<pre class='output'>
array([[ 0.64278761,  0.76604444],
       [-0.76604444,  0.64278761]])
</pre>


Everything is correct!

# Conclusion

In this chapter we saw different different type of matrices with specific properties. It is generally useful to recall these properties while we deal with this kind of matrices to simplify equations for instance.

In the next chapter we will saw a central idea in linear algebra: the eigendecomposition. Keep reading!


# 2.7 Eigendecomposition

We will see some major concepts of linear algebra in this chapter so hang on it worth it! We will start with getting some ideas on eigenvectors and eigenvalues. We will develop on the idea that a matrix can be seen as a linear transformation and that applying a matrix on its eigenvectors gives new vectors with the same direction. Then we will see how to express quadratic equations into the matrix form. We will see that the eigendecomposition of the matrix corresponding to a quadratic equation can be used to find the minimum and maximum of this function. As a bonus, we will also see how to visualize linear transformations in Python!

The eigendecomposition is one form of matrix decomposition. Decomposing a matrix means that we want to find a product of matrices that is equal to the initial matrix. In the case of the eigendecomposition, we decompose the initial matrix into the product of its eigenvectors and eigenvalues. Before all, let's see what are eigenvectors and eigenvalues.

# Matrices as linear transformations

The main thing you have to understand here is that matrices can be seen as linear transformations. Some matrices will rotate your space, others will rescale it etc. So when we apply a matrix to a vector, we end up with a transformed version of the vector. When we say that we 'apply' the matrix to the vector it means that we calculate the dot product of the matrix with the vector. We will start with a basic example of this kind of transformation.

### Example 1.

First, create the matrix $\bs{A}$:

```python
A = np.array([[-1, 3], [2, -2]])
A
```

<pre class='output'>
array([[-1,  3],
       [ 2, -2]])
</pre>

and the vector $\bs{v}$:

```python
v = np.array([[2], [1]])
v
```

<pre class='output'>
array([[2],
       [1]])
</pre>


Now, plot the vector $\bs{v}$ with our custom function:


```python
plotVectors([v.flatten()], cols=['#1190FF'])
plt.ylim(-1, 4)
plt.xlim(-1, 4)
```

<img src="../../assets/images/2.7/simple-vector.png" width="250" alt="Example of a simple vector" title="A simple vector">
<em>A simple vector</em>

That's great! You have everything to see the effect of the matrix $\bs{A}$ on our vector.

In order to apply the matrix $\bs{A}$ to the vector $\bs{v}$ you need to multiply them. And then, plot the old vector (light blue) and the new one (orange):


```python
Av = A.dot(v)
print 'Av:\n', Av
plotVectors([v.flatten(), Av.flatten()], cols=['#1190FF', '#FF9A13'])
plt.ylim(-1, 4)
plt.xlim(-1, 4)
```


<pre class='output'>
Av:
[[1]
 [2]]
</pre>


<img src="../../assets/images/2.7/simple-vector-and-transformation.png" width="250" alt="A simple vector and its transformation" title="A simple vector and its transformation">
<em>A simple vector and its transformation</em>

We can see that applying the matrix $\bs{A}$ has the effect of modifying the vector. This modification is tied to the matrix. Another matrix will induce another transformation!

Now that you can think of matrices as linear transformation recipes, let's see the case of a very special type of vector: the eigenvector.

# Eigenvectors and eigenvalues

We have seen an example of a vector transformed by a matrix. Now imagine that the transformation of the initial vector gives us a new vector that has the exact same direction. The scale can be different but the direction is the same. Applying the matrix didn't change the direction of the vector. This special vector is called an eigenvector of the matrix. We will see that finding the eigenvectors of a matrix can be very useful for different purposes.

<span class='pquote'>
    Imagine that the transformation of the initial vector by the matrix gives a new vector with the exact same direction. This vector is called an eigenvector of $\bs{A}$.
</span>

This means that $\bs{v}$ is a eigenvector of $\bs{A}$ if $\bs{v}$ and $\bs{Av}$ are in the same direction or to rephrase it if the vectors $\bs{Av}$ and $\bs{v}$ are parallel. The output vector is just a scaled version of the input vector. This scalling factor is $\lambda$ which is called the **eigenvalue** of $\bs{A}$.

<div>
$$
\bs{Av} = \lambda\bs{v}
$$
</div>

Don't worry, we will see all of this in examples!

### Example 2.

First, create a new matrix $\bs{A}$ like that:

<div>
$$
\bs{A}=
\begin{bmatrix}
    5 & 1\\\\
    3 & 3
\end{bmatrix}
$$
</div>

```python
A = np.array([[5, 1], [3, 3]])
A
```

<pre class='output'>
array([[5, 1],
       [3, 3]])
</pre>


For this first try, I'll give you the eigenvector $\bs{v}$ of $\bs{A}$. It is:

<div>
$$
\bs{v}=
\begin{bmatrix}
    1\\\\
    1
\end{bmatrix}
$$
</div>

So you can create the vector:

```python
v = np.array([[1], [1]])
v
```

<pre class='output'>
array([[1],
       [1]])
</pre>

Just in case I'm wrong, please check $\bs{Av} = \lambda\bs{v}$:

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

and since:

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

we have:

<div>
$$
\begin{bmatrix}
    5 & 1\\\\
    3 & 3
\end{bmatrix}
\begin{bmatrix}
    1\\\\
    1
\end{bmatrix}=6\times \begin{bmatrix}
    1\\\\
    1
\end{bmatrix}
$$
</div>

Looks good! It corresponds to our formula:

<div>
$$
\bs{Av} = \lambda\bs{v}
$$
</div>

This means that $\bs{v}$ is well an eigenvector of $\bs{A}$. Also, the corresponding eigenvalue is $\lambda=6$.

We can represent $\bs{v}$ and $\bs{Av}$ to check if their directions are the same:

```python
Av = A.dot(v)

orange = '#FF9A13'
blue = '#1190FF'

plotVectors([Av.flatten(), v.flatten()], cols=[blue, orange])
plt.ylim(-1, 7)
plt.xlim(-1, 7)
```

<img src="../../assets/images/2.7/eigenvector-transformation.png" width="250" alt="The direction of the eigenvector after transformation by its matrix is the same as the original vector direction" title="Eigenvector direction">
<em>Eigenvector doesn't change its direction when we apply the corresponding matrix</em>

We can see that their directions are the same!

Now, I have news for you: there is another eigenvector.

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

We have:

<div>
$$
\begin{bmatrix}
    5 & 1\\\\
    3 & 3
\end{bmatrix}\begin{bmatrix}
    1\\\\
    -3
\end{bmatrix} = 2 \times \begin{bmatrix}
    1\\\\
    -3
\end{bmatrix}
$$
</div>

So the corresponding eigenvalue is $\lambda=2$.

Let's create the vector:

```python
v = np.array([[1], [-3]])
v
```

<pre class='output'>
array([[ 1],
       [-3]])
</pre>

And see its graphical representation:


```python
Av = A.dot(v)

plotVectors([Av.flatten(), v.flatten()], cols=[blue, orange])
plt.ylim(-7, 1)
plt.xlim(-1, 3)
```


<img src="../../assets/images/2.7/eigenvector-transformation1.png" width="250" alt="Another eigenvector and its transformation" title="Another eigenvector and its transformation">
<em>Another eigenvector and its transformation</em>


This example shows that the eigenvectors $\bs{v}$ are vectors that change only in scale when we apply the matrix $\bs{A}$ to them. Here the scales were 6 for the first eigenvector and 2 to the second but $\lambda$ can take any real or even complex value.

## Find eigenvalues and eigenvectors in Python

Numpy provides a function returning eigenvectors and eigenvalues. Here a demonstration with the preceding example:


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

The first array corresponds to the eigenvalues and the second to the eigenvectors concatenated in columns.

We can see that the eigenvalues are the same than the ones we used before: 6 and 2 (first array). The eigenvectors correspond to the columns of the second array. This means that the eigenvector corresponding to $\lambda=6$ is:

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

You can notice that these vectors are not the ones we used in the example. They look different because they have not necessarly the same scaling than the ones we gave in the example. We can easily see that the first corresponds to a scaled version of our $\begin{bmatrix}
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

Now that we have an idea of what eigenvectors and eigenvalues are we can see how it can be used to decompose a matrix. Let's have a first look at the formula:

<div>
$$
\bs{A}=\bs{V}\cdot diag(\bs{\lambda}) \cdot \bs{V}^{-1}
$$
</div>


First, what is $\bs{V}$? All eigenvectors of a matrix $\bs{A}$ can be concatenated in a matrix with each column corresponding to each eigenvector (like in the second array return by `np.linalg.eig(A)`):

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

is the eigenvector corresponding to $\lambda=6$ and the second

$$
\begin{bmatrix}
    1\\\\
    -3
\end{bmatrix}
$$

is the eigenvector corresponding to $\lambda=2$.

Second, the vector $\bs{\lambda}$ is a vector containing all eigenvalues:

<div>
$$
\bs{\lambda}=
\begin{bmatrix}
    6\\\\
    2
\end{bmatrix}
$$
</div>

So $diag(\bs{v})$ is a diagonal matrix containing all the eigenvalues in the diagonal (and $0$ everywhere else).

Finally, $\bs{V}^{-1}$ is the inverse matrix of $\bs{V}$.

Let's now have a second look to the formula:

<div>
$$
\bs{A}=\bs{V}\cdot diag(\bs{\lambda}) \cdot \bs{V}^{-1}
$$
</div>

Now that we have seen the formula used to decompose the matrix, let's try to do it for $\bs{A}$ from the last examples. We will do it in the reverse order: since we know the eigenvalues and eigenvectors of $\bs{A}$, we will use them as in the formula to see if we end up with the matrix $\bs{A}$.

Let's create the matrix $\bs{V}$:

<div>
$$
\bs{V}=\begin{bmatrix}
    1 & 1\\\\
    1 & -3
\end{bmatrix}
$$
</div>

```python
V = np.array([[1, 1], [1, -3]])
V
```

<pre class='output'>
array([[ 1,  1],
       [ 1, -3]])
</pre>

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

This can be created with the Numpy function `np.diag()`:

```python
lambdas = np.diag([6, 2])
lambdas
```

<pre class='output'>
array([[6, 0],
       [0, 2]])
</pre>

We also need to calculate the inverse matrix of $\bs{V}$. We will use the `np.linalg.inv()` function from Numpy:

```python
V_inv = np.linalg.inv(V)
V_inv
```

<pre class='output'>
array([[ 0.75,  0.25],
       [ 0.25, -0.25]])
</pre>

We now have all the elements of the equation $\bs{V}\cdot diag(\bs{\lambda}) \cdot \bs{V}^{-1}$. If we replace each elements we have:

<div>
$$
\bs{V}\cdot diag(\bs{\lambda}) \cdot \bs{V}^{-1}
=
\begin{bmatrix}
    1 & 1\\\\
    1 & -3
\end{bmatrix}
\begin{bmatrix}
    6 & 0\\\\
    0 & 2
\end{bmatrix}
\begin{bmatrix}
    0.75 & 0.25\\\\
    0.25 & -0.25
\end{bmatrix}
$$
</div>

Before doing the matrix product in Python (it is a one-liner), we will have a look to the details. We will start to do the dot product of the first two matrices:

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
\bs{V}\cdot diag(\bs{\lambda}) \cdot \bs{V}^{-1}&=
\begin{bmatrix}
    6 & 2\\\\
    6 & -6
\end{bmatrix}
\begin{bmatrix}
    0.75 & 0.25\\\\
    0.25 & -0.25
\end{bmatrix}\\\\
&=\begin{bmatrix}
    5 & 1\\\\
    3 & 3
\end{bmatrix}=
\bs{A}
\end{align*}
$$
</div>

With Python:

```python
V.dot(lambdas).dot(V_inv)
```

<pre class='output'>
array([[ 5.,  1.],
       [ 3.,  3.]])
</pre>


Hooray! It is $\bs{A}$. That confirms our previous calculation.

## Real symmetric matrix

In the case of real symmetric matrices (more details about symmetric matrices in the chapter 2.), the eigendecomposition can be expressed as

<div>
$$
\bs{A} = \bs{Q}\Lambda \bs{Q}^\text{T}
$$
</div>

where $\bs{Q}$ is the matrix with eigenvectors as columns and $\Lambda$ is $diag(\lambda)$.

The notation can be confusing because the name of the matrix of eigenvectors and the name of the matrix containing the eigenvalues have changed. However the idea is the same: the only difference is that instead of using the inverse of the matrix of eigenvectors we use its transpose.

### Example 3.

Let's take an example of a symmetric matrix.

<div>
$$
\bs{A}=\begin{bmatrix}
    6 & 2\\\\
    2 & 3
\end{bmatrix}
$$
</div>

This matrix is symmetric (we can see that $\bs{A}=\bs{A}^\text{T}$).

```python
A = np.array([[6, 2], [2, 3]])
A
```

<pre class='output'>
array([[6, 2],
       [2, 3]])
</pre>

Now we will calculate its eigenvectors. Remember that the function `np.linalg.eig()` outputs two matrices: the first one is an array containing the eigenvalues and the second an array containing the eigenvectors.

```python
eigVals, eigVecs = np.linalg.eig(A)
eigVecs
```

<pre class='output'>
array([[ 0.89442719, -0.4472136 ],
       [ 0.4472136 ,  0.89442719]])
</pre>

So the eigenvectors of $\bs{A}$ are:

<div>
$$
\bs{Q}=
\begin{bmatrix}
    0.89442719 & -0.4472136\\\\
    0.4472136 & 0.89442719
\end{bmatrix}
$$
</div>

We will now put its eigenvalues in a diagonal matrix:

```python
eigVals = np.diag(eigVals)
eigVals
```

<pre class='output'>
array([[ 7.,  0.],
       [ 0.,  2.]])
</pre>

It gives us the following matrix:

<div>
$$
\bs{\Lambda}=
\begin{bmatrix}
    7 & 0\\\\
    0 & 2
\end{bmatrix}
$$
</div>

We have everything to calculate $\bs{A}$ from $\bs{Q}$ and $\Lambda$ and  $\bs{Q}^\text{T}$. Let's begin to calculate $\bs{Q\Lambda}$:

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

Let's calculate $\bs{Q}^\text{T}$:


```python
eigVecs.T
```

<pre class='output'>
array([[ 0.89442719,  0.4472136 ],
       [-0.4472136 ,  0.89442719]])
</pre>

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

It is easier to do it in Python:

```python
eigVecs.dot(eigVals).dot(eigVecs.T)
```

<pre class='output'>
array([[ 6.,  2.],
       [ 2.,  3.]])
</pre>

We can see that the result corresponds to our initial matrix.

# Quadratic form to matrix form

A quadratic function is a function of the form: $f(x)=ax^2+bx+c$ (with $a$, $b$, and $c$ real number and $a\neq0$). For instance $f(x)=x^2$ is a quadratic function. It looks like this:

```python
x = np.arange(-11, 11)
y = x**2

plt.figure()
plt.plot(x, y)
plt.xlim(-10, 10)
plt.ylim(0, 100)
plt.show()
```

<img src="../../assets/images/datacampTutorial/x2.png" width="250" alt="Example of a quadratic function" title="Example of a quadratic function">
<em>Example of a quadratic function: $f(x)=x^2$</em>

In this case the function is an univariate quadratic function but there can be multivariate ones (with more than one variable). For instance, a bivariate quadratic function can look like a bowl in 3 dimensions.

We will see that eigendecomposition can be used to optimize quadratic functions, that is to say to find the minimum or the maximum of the function. Indeed, when $\bs{x}$ takes the values of an eigenvector, $f(\bs{x})$ takes the value of its corresponding eigenvalue. And we will see that these points associated with the eigenvectors are special.

But the first thing to understand is the link between all the matrix manipulations we have done so far and these quadratic equations. The answer is that we can write quadratic functions under the matrix form!

Look at this matrix product:

<div>
$$
f(x_1, x_2)
= \begin{bmatrix}
    x_1 & x_2
\end{bmatrix}\begin{bmatrix}
    a & b\\\\
    c & d
\end{bmatrix}\begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}
$$
</div>

Note that there are 2 dimensions $x_1$ and $x_2$ and thus that it corresponds to a bivariate quadratic function. $a$, $b$, $c$, and $d$ are coefficients in the equation. Let's do the dot product to see that more clearly:

<div>
$$
\begin{align*}
f(x_1, x_2)
&= \begin{bmatrix}
    x_1 & x_2
\end{bmatrix}\begin{bmatrix}
    a & b\\\\
    c & d
\end{bmatrix}\begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}\\\\
&=
\begin{bmatrix}
    ax_1 + cx_2 & bx_1 + dx_2
\end{bmatrix}
\begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}\\\\
&=x_1(ax_1 + cx_2) + x_2(bx_1 + dx_2)\\\\
&=ax_1^2 + cx_1x_2 + bx_1x_2 + dx_2^2
\end{align*}
$$
</div>

The matrices correspond to a quadratic function:

<div>
$$
f(\bs{x}) = ax_1^2 +(b+c)x_1x_2 + dx_2^2
$$
</div>

The matrix form is just another way of writing the function.

We can concatenate the variables $x_1$ and $x_2$ into a vector $\bs{x}$:

<div>
$$
\bs{x} = \begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}
$$
</div>

and let's call $\bs{A}$ the matrix of coefficients:

<div>
$$
\bs{A}=\begin{bmatrix}
    a & b\\\\
    c & d
\end{bmatrix}
$$
</div>

Now we can write:

<div>
$$
f(\bs{x})= \bs{x^\text{T}Ax}
$$
</div>

This is pretty sweet way of writing a quadratic function! We call them matrix forms. This form is useful to do various things on the quadratic equation like constrained optimization (see bellow).

### Example 4.

If you look at the relation between these forms you can see that $a$ gives you the number of $x_1^2$, $(b + c)$ the number of $x_1x_2$ and $d$ the number of $x_2^2$. This means that the same quadratic form can be obtained from infinite number of matrices $\bs{A}$ by changing $b$ and $c$ while preserving their sum.

In this example, we will see how two different matrices can lead to the same equation.

Let's start with:

<div>
$$
\bs{x} = \begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}
$$
</div>

and:

<div>
$$
\bs{A}=\begin{bmatrix}
    2 & 4\\\\
    2 & 5
\end{bmatrix}
$$
</div>

Remember that you can find the function $f(\bs{x}) = ax_1^2 +(b+c)x_1x_2 + dx_2^2$ by replacing the coefficients directly from:

<div>
$$
\bs{A}=\begin{bmatrix}
    a & b\\\\
    c & d
\end{bmatrix}
$$
</div>

This gives the following quadratic form:

<div>
$$
\begin{align*}
&2x_1^2 + (4+2)x_1x_2 + 5x_2^2\\\\
&=2x_1^2 + 6x_1x_2 + 5x_2^2
\end{align*}
$$
</div>

Now we will try with another matrix that has the same total for $b+c$. For the first matrix, $b+c=6$, let's take for example $-3$ and $9$ for $b$ and $c$ that has also a total of $6$:

<div>
$$
\bs{A}=\begin{bmatrix}
    2 & -3\\\\
    9 & 5
\end{bmatrix}
$$
</div>

We still have the quadratic same form:

<div>
$$
\begin{align*}
&2x_1^2 + (-3+9)x_1x_2 + 5x_2^2\\\\
&=2x_1^2 + 6x_1x_2 + 5x_2^2
\end{align*}
$$
</div>

### Example 5

We can also try to find the quadratic form from the matrix form. For this example, we will go from the matrix form to the quadratic form using the symmetric matrix $\bs{A}$ from the example 3.

<div>
$$\bs{A}=\begin{bmatrix}
    6 & 2\\\\
    2 & 3
\end{bmatrix}
$$
</div>

and

<div>
$$
\bs{x} = \begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}
$$
</div>

The equation can be retrieved:

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

### Note on diagonal matrices

If $\bs{A}$ is a diagonal matrix (all 0 except the diagonal), the quadratic form of $\bs{x^\text{T}Ax}$ will have no cross term. The cross term of a bivariate quadratic equation is the term containing both variables. In the example 5. the equation was $6 x_1^2 + 4 x_1x_2 + 3 x_2^2$. The cross term was $4 x_1x_2$.

Take the following matrix form:

<div>
$$
\bs{A}=\begin{bmatrix}
    a & b\\\\
    c & d
\end{bmatrix}
$$
</div>

If $\bs{A}$ is diagonal, then $b$ and $c$ are 0 and since $f(\bs{x}) = ax_1^2 +(b+c)x_1x_2 + dx_2^2$ there is no cross term. A quadratic form without cross term is called **diagonal form** since it comes from a diagonal matrix. We will see along this tutorial that it can be very usefull to work with equation with no cross terms.

# Change of variable

Now that we have seen the link between the matrix form and the quadratic form of an equation, we will build on that and see how we can manipulate the equation in order to simplify it. The simplification we want to make is to get rid of the cross term of the equation (see above). Without the cross term, it will then be easier to characterize the function and eventually optimize it (i.e finding its maximum or minimum).

One way to get rid of the cross term of the equation is to use a change of variable. A change of variable (or linear substitution) simply means that we replace a variable by another one. If we stay in the quadratic functions we used previously, we want to find the variables $y_1$ and $y_2$ that are a linear combination of $x_1$ and $x_2$ such as the new equation $f(y_1, y_2)$ has no cross term.


## 1. With the quadratic form

### Example 6.

Let's take again our previous quadratic form:

<div>
$$
\bs{x^\text{T}Ax} = 6 x_1^2 + 4 x_1x_2 + 3 x_2^2
$$
</div>

The change of variable will concern $x_1$ and $x_2$. We can replace $x_1$ with any combination of $y_1$ and $y_2$ and $x_2$ with any combination $y_1$ and $y_2$. We will of course end up with a new equation. The nice thing is that we can find a specific substitution that will lead to a simplification of our statement. Specifically, it can be used to get rid of the cross term (in our example: $4 x_1x_2$). We will see later in much details why it is interesting.

You will see why it can be usefull to be able to go from the quadratic form to the matrix form. The matrix $\bs{A}$ containing the coefficients can be used to find the right substitution of variables that will remove the cross term! The right substitution is given by... the eigenvectors of $\bs{A}$. That's matrix wizardry! By the way, it is usefull to know these special kind of vectors, right?!


Let's recall that the matrix form of our equation is:

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

To do the change of variable we will apply the vector $\bs{y}$ to the matrix of eigenvectors. This will gives us how much $y_1$ and $y_2$ need to replace $x_1$ and $x_2$:

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

So far so good! To summarize, it means that to get rid of the cross term $4 x_1x_2$ of the equation $6 x_1^2 + 4 x_1x_2 + 3 x_2^2$ we can replace $x_1$ by $\frac{1}{\sqrt{5}}(2y_1 - y_2)$ and $x_2$ by $\frac{1}{\sqrt{5}}(y_1 + 2y_2)$.

Let's do that:

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

## 2. With the Principal Axes Theorem

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

We will see that there is a way to find $f(\bs{x})$ from the eigenvectors and the eigenvalues when $\bs{x}$ is a unit vector.

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

It is finally time to see why removing the cross terms of the quadratic equation makes the process of optimization easy. Using matrix form means that we can do the eigendecomposition of the matrix. Here we bind all our recently acquired knowledge and see how eigendecomposition can be used to optimize quadratic functions. We will use functions without cross-term to show why it is easy and why we learned to remove this term.

### Example 8.

We will see a case of constraint optimization. This means that we want to find the minimum or the maximum of the function while the variables respect certain constraints. In our example we want to find the minimum and the maximum of the function $f(\bs{x})$ while the norm of $\bs{x}$ is equal to $1$ (a unit vector).

To do that we will start from our last example and use the transformation that removed the cross terms. You will see that it makes the process straightforward!

Here is the function that we want to optimize:

<div>
$$
f(\bs{x}) =\bs{x^\text{T}Ax} \textrm{ subject to }||\bs{x}||_2= 1
$$
</div>

After the change of variable we ended up with this equation (no cross term):

<div>
$$
f(\bs{x}) = 7y_1^2 + 2y_2^2
$$
</div>

Furthermore, the constraint of $\bs{x}$ being a unit vector imply:

<div>
$$
||\bs{x}||_2 = 1 \Leftrightarrow x_1^2 + x_2^2 = 1
$$
</div>

You might think that it concerns $\bs{x}$ but not $\bs{y}$. However, we will show that $\bs{y}$ has to be a unit vector if it is the case for $\bs{x}$. Recall first that $\bs{x}=\bs{Py}$:

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

So $\norm{\bs{x}}^2 = \norm{\bs{y}}^2 = 1$ and thus $y_1^2 + y_2^2 = 1$.

Great! We have everything we need to do the optimization. Here is the reasoning: since $y_1^2$ and $y_2^2$ cannot be negative (they are squared values), we can be sure that $2y_2^2\leq7y_2^2$, right?. And if:

<div>
$$
2y_2^2\leq7y_2^2
$$
</div>

then:

<div>
$$
2y_2^2 + 7y_1^2 \leq 7y_1^2 + 7y_2^2
$$
</div>

Hence:

<div>
$$
\begin{align*}
f(\bs{x}) &= 7y_1^2 + 2y_2^2\\\\
&\leq 7y_1^2 + 7y_2^2\\\\
&\leq
7(y_1^2+y_2^2)\\\\
&\leq
7
\end{align*}
$$
</div>

This means that the maximum value of $f(\bs{x})$ is 7.

The same reasoning can lead to find the minimum of $f(\bs{x})$. $7y_1^2\geq2y_1^2$ and:

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

So the minimum of $f(\bs{x})$ is 2.

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

We have seen a lot of things in this chapter. We saw that linear algebra can be used to solve a variety of mathematical problems and more specifically that eigendecomposition is a powerful tool! I hope that you can now see a matrix as a linear transformation recipe and to easily visualize what matrix do what kind of transformation, have a look at the BONUS!

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
    """
    Plot the transformation of the unit circle by a matrix along with the eigenvectors scaled by the eigenvalues. Depends on the function plotVectors().

    Parameters
    ----------
    transformMatrix : array-like
        Matrix to apply to the unit circle. For instance: np.array([[1, 3], [2, 2]]).

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure showing the unit circle, the transformed circled and the eigenvectors.
    """
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

    # Plot the unit circle
    plt.plot(x, y)
    # Plot the eigenvectors rescaled by their respective eigenvalues
    plotVectors([eigVals[0,0]*eigVecs[:,0], eigVals[1,1]*eigVecs[:,1]],
                cols=[sns.color_palette()[1], sns.color_palette()[1]])
    # Plot the transformed circle
    plt.plot(newX, newY)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()
```


```python
A = np.array([[1,-1], [-1, 4]])
linearTransformation(A)
```


<img src="../../assets/images/datacampTutorial/transformed-circle.png" width="250" alt="Plot of the unit circle and its transformation by the matrix A" title="Transformation of the unit circle by the matrix A">
<em>The unit circle and its transformation by the matrix A. The vectors are the eigenvectors of A.</em>

We can see the unit circle in dark blue, the transformed unit circle and the scaled eigenvectors in green.

It is worth noting that the eigenvectors are orthogonal here because the matrix is symmetric. Let's try with a non-symmetric matrix:


```python
A = np.array([[1,1], [-1, 4]])
linearTransformation(A)
```


<img src="../../assets/images/datacampTutorial/transformed-circle-non-symmetric.png" width="250" alt="Plot of the unit circle and its transformation by the matrix A in the case of a non symmetric matrix" title="Transformation of the unit circle by the matrix A - Non symmetric matrix">
<em>The unit circle and its transformation by the matrix A. The vectors are the eigenvectors of A (with A non symmetric).</em>

In this case, the eigenvectors are not orthogonal!




# References

## 1. Norms

- [Norm - Wikipedia](https://en.wikipedia.org/wiki/Norm_(mathematics))

- [3D plots](https://academo.org/demos/3d-surface-plotter/)

## 2. Special Kinds of Matrices and Vectors

- [https://math.stackexchange.com/questions/1936020/why-is-the-inverse-of-an-orthogonal-matrix-equal-to-its-transpose](https://math.stackexchange.com/questions/1936020/why-is-the-inverse-of-an-orthogonal-matrix-equal-to-its-transpose)

- [https://dyinglovegrape.wordpress.com/2010/11/30/the-inverse-of-an-orthogonal-matrix-is-its-transpose/](https://math.stackexchange.com/questions/1936020/why-is-the-inverse-of-an-orthogonal-matrix-equal-to-its-transpose)

# 3 Eigendecomposition

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
