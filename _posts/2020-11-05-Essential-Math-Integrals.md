---
bg: "Essential-Math-for-Data-Science-Update/bridge.jpg"
layout: post
mathjax: true
categories: posts
tags: ['essential-math', 'python', 'numpy']
author: hadrienj
date: 2020-11-05
excerpt: ""
excerpt-image: <img src="../../assets/images/ch02_integrals/ch02_area_under_the_curve.png" width=200><em>Area under the curve.</em>
twitterImg: "Essential-Math-for-Data-Science-Update/output_ch06_139_0"
title: "Essential Math for Data Science: Integrals And Area Under The Curve"
crawlertitle: ""
essential-math-sample: true
---


*Integration* is the inverse operation of differentiation. Take a
function $f(x)$ and calculate its derivative $f'(x)$, the *indefinite
integral* (also called *antiderivative*) of $f'(x)$ gives you back
$f(x)$ (up to a constant, as you'll soon see).

You can use integration to calculate the *area under the curve*, which
is the area of the shape delimited by the function, as shown in Figure
1.

![Figure 1: Area under the
curve.](../../assets/images/ch02_integrals/ch02_area_under_the_curve.png){:width="200px"}
<em>Figure 1: Area under the
curve.</em>

A *definite integral* is the integral over a specific interval. It
corresponds to the area under the curve in this interval.

### Example

You'll see through this example how to understand the relationship
between the integral of a function and the area under the curve. To
illustrate the process, you'll approximate the integral of the function
$g(x) = 2x$ using discretization of the area under the curve.

#### Example Description

Let's take again the example of the moving train. You saw that speed as
a function of time was the derivative of distance as a function of time.
These functions are represented in Figure 2.

![Figure 2: The left panel shows $f(x)$ which is the distance as a
function of time, and the right panel its derivative $g(x)$, which is
the speed as a function of
time.](../../assets/images/ch02_integrals/ch02_distance_speed.png){:width="500px"}
<em>Figure 2: The left panel shows $f(x)$ which is the distance as a
function of time, and the right panel its derivative $g(x)$, which is
the speed as a function of
time.</em>

The function shown in the left panel of Figure
2 is defined as $f(x) = x^2$. Its derivative
is defined as $g(x)=2x$.

In this example, you'll learn how to find an approximation of the area
under the curve of $g(x)$.

#### Slicing the Function

To approximate the area of a shape, you can use the slicing method: you
cut the shape into small slices with an easy shape like rectangles,
calculate the area of each of these slices and sum them.

You'll do exactly that to find an approximation of the area under the
curve of $g(x)$.

![Figure 3: Approximation of the area under the curve by discretizing
the area under the curve of speed as a function of
time.](../../assets/images/ch02_integrals/ch02_speed_function_slices.png){:width="300px"}
<em>Figure 3: Approximation of the area under the curve by discretizing
the area under the curve of speed as a function of
time.</em>

Figure 3 shows the area under the
curve of $f'(x)$ sliced as one-second rectangles (let's call this
difference $\Delta x$). Note that we underestimate the area (look at the
missing triangles), but we'll fix that later.

Let's try to understand the meaning of the slices. Take the first one:
its area is defined as $2 \cdot 1$. The height of the slice is the speed
at one second (the value is 2). So there are two units of speed by one
unit of time for this first slice. The area corresponds to a
multiplication between speed and time: this is a distance.

For instance, if you drive at 50 miles per hour (speed) for two hour
(time), you traveled $50 \cdot 2 = 100$ miles (distance). This is
because the unit of speed corresponds to a ratio between distance and
time (like miles *per* hour). You get:

$$
\frac{\text{distance}}{\text{time}} \cdot \text{time} = \text{distance}
$$

To summarize, the derivative of the distance by time function is the
speed by time function, and the area under the curve of the speed by
time function (its integral) gives you a distance. This is how
derivatives and integrals are related.

#### Implementation

Let's use slicing to approximate the integral of the function $g(x)=2x$.
First, let's define the function $g(x)$:

```python
def g_2x(x):
    return 2 * x
```

As illustrated in Figure 3, you'll
consider that the function is discrete and take a step of
$\Delta x = 1$. You can create an $x$-axis with values from zero to six,
and apply the function `g_2x()` for each of these values. You can use
the Numpy method `arange(start, stop, step)` to create an array filled
with values from `start` to `stop` (not included):

```python
delta_x = 1
x = np.arange(0, 7, delta_x)
x
```

    array([0, 1, 2, 3, 4, 5, 6])

```python
y = g_2x(x)
y
```

    array([ 0,  2,  4,  6,  8, 10, 12])

You can then calculate the slice's areas by iterating and multiplying
the width ($\Delta_x$) by the height (the value of $y$ at this point).
of the slice. As you saw, this area (`delta_x * y[i-1]` in the code
below) corresponds to a distance (the distance of the moving train
traveled during the $i$th slice). You can finally append the results to
an array (`slice_area_all` in the code below).

Note that the index of `y` is `i-1` because the rectangle is on the left
of the $x$ value we estimate. For instance, the area is zero for $x=0$
and $x=1$.

```python
slice_area_all = np.zeros(y.shape[0])
for i in range(1, len(x)):
    slice_area_all[i] = delta_x * y[i-1]
slice_area_all
```

    array([ 0.,  0.,  2.,  4.,  6.,  8., 10.])

These values are the slice's areas.

To calculate the distance traveled from the beginning to the
corresponding time point (and not corresponding to each slice), you can
calculate the cumulative sum of `slice_area_all` with the Numpy function
`cumsum()`:

```python
slice_area_all = slice_area_all.cumsum()
slice_area_all
```

    array([ 0.,  0.,  2.,  6., 12., 20., 30.])

This is the estimated values of the area under the curve of $g(x)$ as a
function of $x$. We know that the function $g(x)$ is the derivative of
$f(x)=x^2$, so we should get back $f(x)$ by integration of $g(x)$.

Let's plot our estimation and $f(x)$, which we'll call the "true
function", to compare them:

```python
plt.plot(x, x ** 2, label='True')
plt.plot(x, slice_area_all, label='Estimated')

```

![Figure 4: Comparison of estimated and original
function.](../../assets/images/ch02_integrals/ch02_integrals_14_0.png){:width="300px"}
<em>Figure 4: Comparison of estimated and original
function.</em>

Let's estimate the integral function with $\Delta x = 0.1$:

```python
delta_x = 0.1
x = np.arange(0, 7, delta_x)
y = g_2x(x)
#  [...] Calculate and plot slice_area_all

```

![Figure 6: Smaller slice widths lead to a better estimation of the
original
function.](../../assets/images/ch02_integrals/ch02_integrals_16_0.png){:width="200px"}
<em>Figure 6: Smaller slice widths lead to a better estimation of the
original
function.</em>

### Riemann Sum {#sec:ch03_section_riemann_sum}

Approximating of an integral using this slicing method is called a
*Riemann sum*. Riemann sums can be calculated in different ways, as you
can see in Figure 8.

![Figure 8: Four kinds of Riemann sums for integral
approximation.](../../assets/images/ch02_integrals/ch02_riemann.png){:width="250px"}
<em>Figure 8: Four kinds of Riemann sums for integral
approximation.</em>

As pictured in Figure 8, with the left Riemann sum,
the curve is aligned with the left corner of the rectangle. With the
right Riemann sum, the curve is aligned with the right corner of the
rectangle. With the midpoint rule, the curve is aligned with the center
of the rectangle. With the trapezoidal rule, a trapezoidal shape is used
instead of a rectangle. The curve crosses both top corners of the
trapezoid.

### Mathematical Definition

In the last section, you saw the relationship between area under the
curve and integration (you got back the original function from the
derivative). Let's see now the mathematical definition of integrals.

The integrals of the function $f(x)$ with respect to $x$ is denoted as
following:

$$
\int f(x) \: dx
$$

The symbol $dx$ is called the *differential* of $x$ and refers to the
idea of an infinitesimal change of $x$. It is a difference in $x$ that
approaches 0. The main idea of integrals is to sum an infinite number of
slices which have an infinitely small width.

The symbol $\int$ is the integral sign and refers to the sum of an
infinite number of slices.

The height of each slice is the value $f(x)$. The multiplication of
$f(x)$ and $dx$ is thus the area of each slice. Finally,
$\int f(x) \: dx$ is the sum of the slice areas over an infinite number
of slices (the width of the slices tending to zero). This is the *area
under the curve*.

You saw in the last section how to approximate function integrals. But
if you know the derivative of a function, you can retrieve the integral
knowing that it is the inverse operation. For example, if you know that:

$$
\frac{d (x^2)}{dx} = 2x
$$

You can conclude that the integral of $2x$ is $x^2$. However, there is a
problem. If you add a constant to our function the derivative is the
same because the derivative of a constant is zero. For instance,

$$
\frac{d (x^2 + 3)}{dx} = 2x
$$

It is impossible to know the value of the constant. For this reason, you
need to add an unknown constant to the expression, as following:

$$
\int 2x \: dx = x^2 + c
$$

with $c$ being a constant.

##### Definite Integrals

In the case of *definite integrals*, you denote the interval of
integration with numbers below and above the integral symbol, as
following:

$$
\int _{a}^{b} f(x) \: dx
$$

It corresponds to the area under the curve of the function $f(x)$
between $x=a$ and $x=b$, as illustrated in Figure
9.

![Figure 9: Area under the curve between $x=a$ and
$x=b$.](../../assets/images/ch02_integrals/ch02_integrals_area_under_the_curve.png){:width="150px"}
<em>Figure 9: Area under the curve between $x=a$ and
$x=b$.</em>
