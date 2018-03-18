---
bg: "road.jpg"
layout: post
mathjax: true
title: Linear Algebra for deep learning with practical code and visualizations
crawlertitle: "deep learning machine learning linear algebra python getting started numpy data sciences"
categories: posts
tags: ['linear algebra', 'deep learning', 'machine learning']
author: hadrienj
---

<img src="../../assets/images/dpl_cover.jpg" width="1000">

<span class='pquote'>
    Boost your data sciences skills. Learn linear algebra.
</span>

I'd like to introduce a series of blog posts and their corresponding Python Notebooks gathering notes on [the Deep Learning Book](http://www.deeplearningbook.org/) from Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016). The aim of these notebooks is to help beginners/advanced beginners to grasp linear algebra concepts underlying deep learning and machine learning. Acquiring these skills can boost your ability to understand and apply various data sciences algorithms. You can think of it as one of the bedrock of machine learning, deep learning and data sciences.

These notes cover the chapter 2 on Linear Algebra. I liked this chapter because it gives a sense of what is most used in the domain of machine learning and deep learning. It is thus a great syllabus for anyone who want to dive in deep learning and acquire get the concepts of linear algebra useful to better understand deep learning algorithms.

You can find all the notebooks [on Github](https://github.com/hadrienj/deepLearningBook-Notes).

# Getting started with linear algebra

The goal of this series is to provide content for beginners who wants to understand enough linear algebra to be confortable with machine learning and deep learning. However, I think that the linear algebra ressource proposed in the Deep Learning book is a bit tough for beginners. I tried to produce a lot of code, examples and drawings on each part of this chapter in order to add steps that may not be obvious for beginners. I also think that you can convey as much information and knowledge through examples than through general definitions. The illustrations are a way to see the big picture of an idea. Finally, I think that coding is a great tool to experiment concretly these abstract mathematical notions. With pen and paper, it adds a layer of what you can try to push your understanding through new horizons.

<span class='pquote'>
    Coding is a great tool to concretly experiment abstract mathematical notions
</span>

# The use of Python/Numpy

In addition, I noticed that creating and reading examples is really helpful to understand the theory. It is why I built Python notebooks instead or just gather mathematical explanations. The goal is two folds:

1. Provide a starting point to use Python/Numpy to apply linear algebra concepts. And since the final goal is to use linear algebra concepts for deep learning it seems natural to continuously go between theory and code. All you will need is a working Python installation with major librairies like Numpy/Scipy/Matplotlib.

2. Give a more concrete vision of the underlying concepts. I found immensely useful to play and experiment with these notebooks to build my understanding of somewhat complicated theoretical concepts or notations. I hope that reading them will be as useful.

# Syllabus

The syllabus follow exactly the [Deep Learning Book so you can find more details if you can't understand one specific point while you are reading it.

1. [Scalars, Vectors, Matrices and Tensors]()
2. [Multiplying Matrices and Vectors]()
3. [Identity and Inverse Matrices]()
4. [Linear Dependence and Span]()
5. [Norms]()
6. [Special Kinds of Matrices and Vectors]()
7. [Eigendecomposition]()
8. [Singular Value Decomposition]()
9. [The Moore-Penrose Pseudoinverse]()
10. [The Trace Operator]()
11. [The Determinant]]()
12. [Example: Principal Components Analysis]]()

# Setup



# Enjoy

I tried to be as accurate as I could and double checked all statements. However, it is definitely possible that you encounter errors/misunderstandings/typos/english weirdness... Please report it! It is very valuable! and I will try to fix it. You can send me emails or open issues and pull request in the notebooks gihub.

# References

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.
