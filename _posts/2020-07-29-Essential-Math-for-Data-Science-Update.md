---
bg: "Essential-Math-for-Data-Science-Update/bridge.jpg"
layout: post
mathjax: true
title: "Essential Math for Data Science: New Chapters"
crawlertitle: "Essential Math for Data Science Update"
categories: posts
tags: ['essential-math', 'python', 'numpy']
author: hadrienj
date: 2020-07-29 13:30:00
excerpt: Update concerning "Essential Math for Data Science" with O'Reilly and complete table of content of the book.
excerpt-image: <img src="../../assets/images/Essential-Math-for-Data-Science-Update/output_ch06_139_0.png" width="500" alt="Contour L1 Regularization" title="L1 Regularization"><em>L1 Regularization</em>
twitterImg: "Essential-Math-for-Data-Science-Update/output_ch06_139_0"
---


<a href="https://learning.oreilly.com/library/view/essential-math-for/9781098115555/">
	<img src="../../assets/images/cover_col.jpg" width="20%" alt="Cover of the book Essential Math for Data Science" title="Cover of the book Essential Math for Data Science">
	<em>Essential Math for Data Science</em>
</a>

I'm glad to announce a few updates concerning my book <a href="https://learning.oreilly.com/library/view/essential-math-for/9781098115555/">"Essential Math for Data Science"</a> with O'Reilly:
- First, the first half of the book (chapters 02 to 06) is now available as Early Release on <a href="https://learning.oreilly.com/library/view/essential-math-for/9781098115555/">O'Reilly</a>. If you don't have an account, you can <a href="https://learning.oreilly.com/get-learning/?code=JEAN20">go here</a> to get a free 30-day access. The missing chapters (07 to 11) should be released in the next few weeks.

For some of you that accessed the previous version of the book, you'll notice that the first chapter on basic algebra has been removed. Part of old chapter 02 has been merged in the linear algebra part.

- I restructured the table of content: I removed some content about very basic math (like what is an equation or a function) to have more space to cover slighly more advanced contents. The part *Statistics and Probability* is now at the beginning of the book (just after a first part on Calculus). Have a look at the new TOC below to have more details.
- There is now <i class="material-icons right col-gray">build</i> *hands-on projects* for each chapter. Hands-on projects are sections where you can apply the math you just learned to a practical machine learning problem (like gradient descent or regularization, for instance). The difficulty of math and code in each of these hands-on project is variable, so you should find something at the right point of your learning curve.

Here is the table of content. Chapters available in the Early Release are in blue and the other in red.


<!--
The goal of the book is to provide an introduction to the mathematics needed for data science and machine learning. The idea is to use a hands-on approach using examples in Python <img src="../../assets/images/icons/python.png" style="display: inline-block; margin: 0; padding: 0;" width="30" alt="python icon" title="Python">, with Numpy <img src="../../assets/images/icons/np.png" style="display: inline-block; margin: 0; padding: 0;" width="30" alt="numpy icon" title="Numpy">, Matplotlib <img src="../../assets/images/icons/matplotlib.png" style="display: inline-block; margin: 0; padding: 0;" width="30" alt="Matplotlib icon" title="Matplotlib">, and Sklearn <img src="../../assets/images/icons/sklearn.png" style="display: inline-block; margin: 0; padding: 0;" width="30" alt="sklearn icon" title="Scikit Learn"> to get mathematical insights that will ease the every day life of a data engineer or data scientist. You will be able to experiment on the math concepts and gain intuition through code and visualizations.

What you'll learn in this book is selected to be actionable knowledge in the context of data science and machine learning / deep learning. The content is more about getting the intuition that will unlock your skills rather than providing mathematical proofs of theorems. It is aimed at people without a background in mathematics but who want to understand tools and algorithms used in data science and machine learning, like developers moving towards data science.

Here are some details about the first two parts of the book.

The first three chapters are about the basics, from equations and functions to integrals and derivatives, and they summarize what mathematical bedrock tools you need for data science and the next chapters.


##### Ch01. How to use this book

Chapter 01 is a very short introduction about the structure of the book, how to get the Jupyter notebooks (all the code will be available on Github), how to setup your computer etc.

-->

### Table of Content

<ul class="collapsible">
	<li>
    	<div class="collapsible-header collapsible-h1 no-click">
    		<i class="material-icons right more white-text">expand_more</i>
			<i class="material-icons right less white-text" style="display: none">expand_less</i>
    		<span class="col-red">01</span>. How to use this Book
    	</div>
	</li>
</ul>

#### PART 1. Calculus

Machine learning and data science require some experience with calculus. You'll be introduced here to derivatives and integrals and how they are useful in machine learning and data science.

<ul class="collapsible">
	<li>
    	<div class="collapsible-header collapsible-h1">
    		<i class="material-icons right more">expand_more</i>
    		<i class="material-icons right less" style="display: none">expand_less</i>
    		<span class="col-blue">02</span>. Calculus: Derivatives and Integrals
    	</div>
    	<div class="collapsible-body">
			<div class="collapse-out">
	    		<div class="collapse-left">
	    			<img src="../../assets/images/Essential-Math-for-Data-Science-Update/ch02_integrals_segments_non_linear.png" width="100%" alt="Area under the curve" title="Area under the curve">
					<em>Area under the curve</em>
	    		</div>
				<ul class="collapsible collapse-in collapse-right">
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  2.1 Derivatives
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 2.1.1 Introduction</p>
						<p class="subsection-toc"> 2.1.2 Mathematical Definition of Derivatives</p>
						<p class="subsection-toc"> 2.1.3 Derivatives of Linear And Nonlinear Functions</p>
						<p class="subsection-toc"> 2.1.4 Derivative Rules</p>
						<p class="subsection-toc"> 2.1.5 Partial Derivatives And Gradients</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  2.2 Integrals And Area Under The Curve
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 2.2.1 Example</p>
						<p class="subsection-toc"> 2.2.2 Riemann Sum</p>
						<p class="subsection-toc"> 2.2.3 Mathematical Definition</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
		    			  <i class="material-icons right col-gray">build</i>2.3 Hands-On Project: Gradient Descent
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 2.3.1 Cost function</p>
						<p class="subsection-toc"> 2.3.2 Derivative of the Cost Function</p>
						<p class="subsection-toc"> 2.3.3 Implementing Gradient Descent</p>
						<p class="subsection-toc"> 2.3.4 MSE Cost Function With Two Parameters</p>
				      </div>
				    </li>
				</ul>
			</div>
		</div>
	</li>
</ul>

#### PART 2. Statistics and Probability

In machine learning and data science, probability and statistics are used to deal with uncertainty. This uncertainty comes from various sources, from data collection to the process you're trying to model itself. This part will introduce you to descriptive statistics, probability distributions, bayesian statistics and information theory.

<ul class="collapsible">
    <li>
    	<div class="collapsible-header collapsible-h1">
	    	<i class="material-icons right more">expand_more</i>
	    	<i class="material-icons right less" style="display: none">expand_less</i>
	    	<span class="col-blue">03</span>. Statistics and Probability Theory
    	</div>
    	<div class="collapsible-body">
			<div class="collapse-out">
	    		<div class="collapse-left">
	    			<img src="../../assets/images/Essential-Math-for-Data-Science-Update/ch03_joint_distribution_10_slices_density.png" width="100%" alt="Joint and Marginal Probability" title="Joint and Marginal Probability">
					<em>Joint and Marginal Probability</em>
	    		</div>
				<ul class="collapsible collapse-in collapse-right">
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  3.1 Descriptive Statistics
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 3.1.1 Variance and Standard Deviation</p>
						<p class="subsection-toc"> 3.1.2 Covariance and Correlation</p>
						<p class="subsection-toc"> 3.1.3 Covariance Matrix</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  3.2 Random Variables
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 3.2.1 Definitions and Notation</p>
						<p class="subsection-toc"> 3.2.2 Discrete and Continuous Random Variables</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  3.3 Probability Distributions
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 3.3.1 Probability Mass Functions</p>
						<p class="subsection-toc"> 3.3.2 Probability Density Functions</p>
						<p class="subsection-toc"> 2.3.3 Implementing Gradient Descent</p>
						<p class="subsection-toc"> 2.3.4 MSE Cost Function With Two Parameters</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  3.4 Joint, Marginal, and Conditional Probability
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 3.4.1 Joint Probability</p>
						<p class="subsection-toc"> 3.4.2 Marginal Probability</p>
						<p class="subsection-toc"> 3.4.3 Conditional Probability</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header no-click">
				      	<i class="material-icons right more white-text">expand_more</i>
				      	<i class="material-icons right less white-text" style="display: none">expand_less</i>
		    			3.5 Cumulative Distribution Functions
	    			  </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  3.6 Expectation and Variance of Random Variables
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 3.6.1 Discrete Random Variables</p>
						<p class="subsection-toc"> 3.6.2 Continuous Random Variables</p>
						<p class="subsection-toc"> 3.6.3 Variance of Random Variables</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right col-gray">build</i>3.7 Hands-On Project: The Central Limit Theorem
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 3.7.1 Continuous Distribution</p>
						<p class="subsection-toc"> 3.7.2 Discrete Distribution</p>
				      </div>
				    </li>
				</ul>
			</div>
		</div>
    </li>
    <li>
    	<div class="collapsible-header collapsible-h1">
	    	<i class="material-icons right more">expand_more</i>
	    	<i class="material-icons right less" style="display: none">expand_less</i>
	    	<span class="col-blue">04</span>. Common Probability Distributions
	    	</div>
    	<div class="collapsible-body">
			<div class="collapse-out">
	    		<div class="collapse-left">
	    			<img src="../../assets/images/Essential-Math-for-Data-Science-Update/output_ch04_16_0.png" width="100%" alt="Gaussian Distributions" title="Gaussian Distributions">
					<em>Gaussian Distributions</em>
	    		</div>
				<ul class="collapsible collapse-in collapse-right">
				    <li>
						<div class="collapsible-header no-click">
							<i class="material-icons right more white-text">expand_more</i>
							<i class="material-icons right less white-text" style="display: none">expand_less</i>
							4.1 Uniform Distribution
						</div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  4.2 Gaussian distribution
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 4.2.1 Formula</p>
						<p class="subsection-toc"> 4.2.2 Parameters</p>
						<p class="subsection-toc"> 4.2.3 Requirements</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header no-click">
				      	<i class="material-icons right more white-text">expand_more</i>
				      	<i class="material-icons right less white-text" style="display: none">expand_less</i>
		    			4.3 Bernoulli Distribution
	    			  </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  4.4 Binomial Distribution
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 4.4.1 Description</p>
						<p class="subsection-toc"> 4.4.2 Graphical Representation</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  4.5 Poisson Distribution
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 4.5.1 Mathematical Definition</p>
						<p class="subsection-toc"> 4.5.2 Example</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  4.6 Exponential Distribution
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 4.6.1 Derivation from the Poisson Distribution</p>
						<p class="subsection-toc"> 4.6.2 Effect of λ</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header no-click">
		    			  <i class="material-icons right col-gray">build</i>4.7 Hands-on Project: Waiting for the Bus
	    			  </div>
				    </li>
				</ul>
			</div>
		</div>
    </li>
    <li>
    	<div class="collapsible-header collapsible-h1">
	    	<i class="material-icons right more">expand_more</i>
	    	<i class="material-icons right less" style="display: none">expand_less</i>
	    	<span class="col-blue">05</span>. Bayesian Statistics and Information Theory
    	</div>
    	<div class="collapsible-body">
			<div class="collapse-out">
	    		<div class="collapse-left">
	    			<img src="../../assets/images/Essential-Math-for-Data-Science-Update/output_ch05_92_0.png" width="100%" alt="Bayesian Inference" title="Bayesian Inference">
					<em>Bayesian Inference</em>
	    		</div>
				<ul class="collapsible collapse-in collapse-right">
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  5.1 Bayes’ Theorem
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 5.1.1 Mathematical Formulation</p>
						<p class="subsection-toc"> 5.1.2 Example</p>
						<p class="subsection-toc"> 5.1.3 Bayesian Interpretation</p>
						<p class="subsection-toc"> 5.1.4 Bayes’ Theorem with Distributions</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  5.2 Likelihood
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 5.2.1 Introduction and Notation</p>
						<p class="subsection-toc"> 5.2.2 Finding the Parameters of the Distribution</p>
						<p class="subsection-toc"> 5.2.3 Maximum Likelihood Estimation</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  5.3 Information Theory
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 5.3.1 Shannon Information</p>
						<p class="subsection-toc"> 5.3.2 Entropy</p>
						<p class="subsection-toc"> 5.3.3 Cross Entropy</p>
						<p class="subsection-toc"> 5.3.4 Kullback-Leibler Divergence (KL Divergence)</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
		    			  <i class="material-icons right col-gray">build</i>5.4 Hands-On Project: Bayesian Inference
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 5.4.1 Advantages of Bayesian Inference</p>
						<p class="subsection-toc"> 5.4.2 Project</p>
				      </div>
				    </li>
				</ul>
			</div>
		</div>
    </li>
</ul>

#### PART 3. Linear Algebra

Linear algebra is the core of many machine learning algorithms. The great news is that you don't need to be able to code these algorithms yourself. It is more likely that you'll use a great Python library instead. However, to be able to choose the right model for the right job, or to debug a broken machine learning pipeline, it is crucial to have enough understanding of what's under the hood. The goal of this part is to give you enough understanding and intuition about the major concepts of linear algebra used in machine and data science. It is designed to be accessible, even if you never studied linear algebra.

<ul class="collapsible">
    <li>
    	<div class="collapsible-header collapsible-h1">
	    	<i class="material-icons right more">expand_more</i>
	    	<i class="material-icons right less" style="display: none">expand_less</i>
	    	<span class="col-blue">06</span>. Scalars and Vectors
    	</div>
    	<div class="collapsible-body">
			<div class="collapse-out">
	    		<div class="collapse-left">
	    			<img src="../../assets/images/Essential-Math-for-Data-Science-Update/output_ch06_139_0.png" width="100%" alt="L1 Regularization. Effect of Lambda." title="L1 Regularization. Effect of alpha.">
					<em>L1 Regularization. Effect of Lambda.</em>
	    		</div>
				<ul class="collapsible collapse-in collapse-right">
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  6.1 What Vectors are?
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 6.1.1 Geometric and Coordinate Vectors</p>
						<p class="subsection-toc"> 6.1.2 Vector Spaces</p>
						<p class="subsection-toc"> 6.1.3 Special Vectors</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  6.2 Operations and Manipulations on Vectors
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 6.2.1 Scalar Multiplication</p>
						<p class="subsection-toc"> 6.2.2 Vector Addition</p>
						<p class="subsection-toc"> 6.2.3 Transposition</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  6.3 Norms
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 6.3.1 Definitions</p>
						<p class="subsection-toc"> 6.3.2 Common Vector Norms</p>
						<p class="subsection-toc"> 6.3.3 Norm Representations</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  6.4 The Dot Product
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 6.4.1 Definition</p>
						<p class="subsection-toc"> 6.4.2 Geometric interpretation: Projections</p>
						<p class="subsection-toc"> 6.4.3 Properties</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
		    			  <i class="material-icons right col-gray">build</i>6.5 Hands-on Project: Regularization
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 6.5.1 Introduction</p>
						<p class="subsection-toc"> 6.5.2 Effect of Regularization on Polynomial Regression</p>
						<p class="subsection-toc"> 6.5.3 Differences between $L^1$ and $L^2$ Regularization</p>
				      </div>
				    </li>
				</ul>
			</div>
		</div>
    </li>
    <li>
    	<div class="collapsible-header collapsible-h1">
	    	<i class="material-icons right more">expand_more</i>
	    	<i class="material-icons right less" style="display: none">expand_less</i>
	    	<span class="col-red">07</span>. Matrices and Tensors
    	</div>
    	<div class="collapsible-body">
			<div class="collapse-out">
	    		<div class="collapse-left">
	    			<img src="../../assets/images/Essential-Math-for-Data-Science-Update/ch07_scalars_to_tensors.png" width="100%" alt="Scalars, vectors, matrices and tensors" title="Scalars, vectors, matrices and tensors">
	    			<em>Scalars, vectors, matrices and tensors</em>
	    		</div>
				<ul class="collapsible collapse-in collapse-right">
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  7.1 Introduction
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 7.1.1 Matrix Notation</p>
						<p class="subsection-toc"> 7.1.2 Shapes</p>
						<p class="subsection-toc"> 7.1.3 Indexing</p>
						<p class="subsection-toc"> 7.1.4 Main Diagonal</p>
						<p class="subsection-toc"> 7.1.5 Tensors</p>
						<p class="subsection-toc"> 7.1.6 Frobenius Norm</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  7.2 Operations and Manipulations on Matrices
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 7.2.1 Addition and Scalar Multiplication</p>
						<p class="subsection-toc"> 7.2.2 Transposition</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  7.3 Matrix Product
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 7.3.1 Matrices with Vectors</p>
						<p class="subsection-toc"> 7.3.2 Matrices Product</p>
						<p class="subsection-toc"> 7.3.3 Transpose of a Matrix Product</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  7.4 Special Matrices
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 7.4.1 Square Matrices</p>
						<p class="subsection-toc"> 7.4.2 Diagonal Matrices</p>
						<p class="subsection-toc"> 7.4.3 Identity Matrices</p>
						<p class="subsection-toc"> 7.4.4 Inverse Matrices</p>
						<p class="subsection-toc"> 7.4.5 Orthogonal Matrices</p>
						<p class="subsection-toc"> 7.4.6 Symmetric Matrices</p>
						<p class="subsection-toc"> 7.4.7 Triangular Matrices</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
				      	<i class="material-icons right col-gray">build</i>7.5 Hands-on Project: Image Classifier
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 7.5.1 Images as Multi-dimensional Arrays</p>
						<p class="subsection-toc"> 7.5.2 Data Preparation</p>
				      </div>
				    </li>
				</ul>
			</div>
		</div>
    </li>
    <li>
    	<div class="collapsible-header collapsible-h1">
	    	<i class="material-icons right more">expand_more</i>
	    	<i class="material-icons right less" style="display: none">expand_less</i>
	    	<span class="col-red">08</span>. Span, Linear Dependency, and Space Transformation
    	</div>
    	<div class="collapsible-body">
			<div class="collapse-out">
	    		<div class="collapse-left">
	    			<img src="../../assets/images/Essential-Math-for-Data-Science/linear-combination-two-vectors.png" width="100%" alt="Linear Combination" title="Linear Combination">
					<em>All linear Combinations of two vectors</em>
	    		</div>
				<ul class="collapsible collapse-in collapse-right">
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  8.1 Linear Transformations
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 8.1.1 Intuition</p>
						<p class="subsection-toc"> 8.1.2 Linear Transformations as Vectors and Matrices</p>
						<p class="subsection-toc"> 8.1.3 Geometric Interpretation</p>
						<p class="subsection-toc"> 8.1.4 Special Cases</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  8.2 Linear combination
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 8.2.1 Intuition</p>
						<p class="subsection-toc"> 8.2.2 All combinations of vectors</p>
						<p class="subsection-toc"> 8.2.3 Span</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  8.3 Subspaces
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 8.3.1 Definitions</p>
						<p class="subsection-toc"> 8.3.2 Subspaces of a Matrix</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  8.4 Linear dependency
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 8.4.1 Geometric Interpretation</p>
						<p class="subsection-toc"> 8.4.2 Matrix View</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  8.5 Basis
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 8.5.1 Definitions</p>
						<p class="subsection-toc"> 8.5.2 Linear Combination of Basis Vectors</p>
						<p class="subsection-toc"> 8.5.3 Other Bases</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  8.6 Special Characteristics
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 8.6.1 Rank</p>
						<p class="subsection-toc"> 8.6.2 Trace</p>
						<p class="subsection-toc"> 8.6.3 Determinant</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header no-click">
				      	<i class="material-icons right col-gray">build</i>8.7 Hands-On Project: Span
	    			  </div>
				    </li>
				</ul>
			</div>
		</div>
    </li>
    <li>
    	<div class="collapsible-header collapsible-h1">
	    	<i class="material-icons right more">expand_more</i>
	    	<i class="material-icons right less" style="display: none">expand_less</i>
	    	<span class="col-red">09</span>. Systems of Linear Equations
    	</div>
    	<div class="collapsible-body">
			<div class="collapse-out">
	    		<div class="collapse-left">
	    			<img src="../../assets/images/Essential-Math-for-Data-Science/projection-in-column-space-with-b.png" width="100%" alt="Projection of a vector onto a plane" title="Projection of a vector onto a plane">
					<em>Projection of a vector onto a plane</em>
	    		</div>
				<ul class="collapsible collapse-in collapse-right">
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  9.1 System of linear equations
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 9.1.1 Row Picture</p>
						<p class="subsection-toc"> 9.1.2 Column Picture</p>
						<p class="subsection-toc"> 9.1.3 Number of Solutions</p>
						<p class="subsection-toc"> 9.1.4 Representation of Linear Equations With Matrices</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  9.2 System Shape
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 9.2.1 Overdetermined Systems of Equations</p>
						<p class="subsection-toc"> 9.2.2 Underdetermined Systems of Equations</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  9.3 Projections
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 9.3.1 Solving Systems of Equations</p>
						<p class="subsection-toc"> 9.3.2 Projections to Approximate Unsolvable Systems</p>
						<p class="subsection-toc"> 9.3.3 Projections Onto a Line</p>
						<p class="subsection-toc"> 9.3.4 Projections Onto a Plane</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
				      	<i class="material-icons right col-gray">build</i>9.4 Hands-on Project: Linear Regression Using Least Approximation
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 9.4.1 Linear Regression Using the Normal Equation</p>
						<p class="subsection-toc"> 9.4.2 Relationship Between Least Squares and the Normal Equation</p>
				      </div>
				    </li>
				</ul>
			</div>
		</div>
    </li>
    <li>
    	<div class="collapsible-header collapsible-h1">
	    	<i class="material-icons right more">expand_more</i>
	    	<i class="material-icons right less" style="display: none">expand_less</i>
	    	<span class="col-red">10</span>. Eigenvectors and Eigenvalues
    	</div>
    	<div class="collapsible-body">
			<div class="collapse-out">
	    		<div class="collapse-left">
	    			<img src="../../assets/images/Essential-Math-for-Data-Science-Update/output_ch10_113_0.png" width="100%" alt="Principal Component Analysis on audio samples." title="Principal Component Analysis on audio samples.">
					<em>Principal Component Analysis on audio samples.</em>
	    		</div>
				<ul class="collapsible collapse-in collapse-right">
				    <li>
				      <div class="collapsible-header no-click">
					      <i class="material-icons right more white-text">expand_more</i>
					      <i class="material-icons right less white-text" style="display: none">expand_less</i>
		    			  10.1 Eigenvectors and Linear Transformations
	    			  </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  10.2 Change of Basis
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 10.2.1 Linear Combinations of the Basis Vectors</p>
						<p class="subsection-toc"> 10.2.2 The Change of Basis Matrix</p>
						<p class="subsection-toc"> 10.2.3 Example: Changing the Basis of a Vector</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  10.3 Linear Transformations in Different Bases
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 10.3.1 Transformation Matrix</p>
						<p class="subsection-toc"> 10.3.2 Transformation Matrix in Another Basis</p>
						<p class="subsection-toc"> 10.3.3 Interpretation</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  10.4 Eigendecomposition
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 10.4.1 First Step: Change of Basis</p>
						<p class="subsection-toc"> 10.4.2 Eigenvectors and Eigenvalues</p>
				      	<p class="subsection-toc"> 10.4.3 Diagonalization</p>
						<p class="subsection-toc"> 10.4.4 Eigendecomposition of Symmetric Matrices</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
				      	<i class="material-icons right col-gray">build</i>10.5 Hands-On Project: Principal Component Analysis
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 10.5.1 Under the Hood</p>
						<p class="subsection-toc"> 10.5.2 Making Sense of Audio</p>
				      </div>
				    </li>
				</ul>
			</div>
		</div>
    </li>
    <li>
    	<div class="collapsible-header collapsible-h1">
	    	<i class="material-icons right more">expand_more</i>
	    	<i class="material-icons right less" style="display: none">expand_less</i>
	    	<span class="col-red">11</span>. Singular Value Decomposition
    	</div>
    	<div class="collapsible-body">
    		<div class="collapse-out">
	    		<div class="collapse-left">
	    			<img src="../../assets/images/Essential-Math-for-Data-Science/ch09_SVD_geometry.png" width="100%" alt="SVD Geometry" title="SVD Geometry">
					<em>SVD Geometry</em>
	    		</div>
				<ul class="collapsible collapse-in collapse-right">
					<li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  11.1 Nonsquare Matrices
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 11.1.1 Different Input and Output Spaces</p>
						<p class="subsection-toc"> 11.1.2 Specifying the Bases</p>
						<p class="subsection-toc"> 11.1.3 Eigendecomposition is Only for Square Matrices</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  11.2 Expression of the SVD
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 11.2.1 From Eigendecomposition to SVD</p>
						<p class="subsection-toc"> 11.2.2 Singular Vectors and Singular Values</p>
						<p class="subsection-toc"> 11.2.3 Finding the Singular Vectors and the Singular Values</p>
				      	<p class="subsection-toc"> 11.2.4 Summary</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  11.3 Geometry of the SVD
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 11.3.1 Two-Dimensional Example</p>
						<p class="subsection-toc"> 11.3.2 Comparison with Eigendecomposition</p>
				      	<p class="subsection-toc"> 11.3.3 Three-Dimensional Example</p>
						<p class="subsection-toc"> 11.3.4 Summary</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header">
					      <i class="material-icons right more">expand_more</i>
		    			  <i class="material-icons right less" style="display: none">expand_less</i>
		    			  11.4 Low-Rank Matrix Approximation
	    			  </div>
				      <div class="collapsible-body">
				      	<p class="subsection-toc"> 11.4.1 Full SVD, Thin SVD and Truncated SVD</p>
						<p class="subsection-toc"> 11.4.2 Decomposition into Rank One Matrices</p>
				      </div>
				    </li>
				    <li>
				      <div class="collapsible-header no-click">
				      	<i class="material-icons right more col-gray">build</i>
				      	<i class="material-icons right less col-gray" style="display: none">build</i>
				      	11.5 Hands-On Project: Image Compression
	    			  </div>
				    </li>
				</ul>
			</div>
		</div>
    </li>
</ul>

I hope that you'll find this content useful! Feel free to contact me if you have any question, request, or feedback!

<!--

#### Ch04. Scalars and Vectors

Chapter 04 is the first chapter in the central part of the book on linear algebra. It is about scalars and vector. You'll build the crucial intuition about the relation between geometric vectors and lists of numbers. Then, you'll start to think in terms of spaces and subspaces. We'll cover the dot product and the idea of norm, with an example on regularization.

<img src="../../assets/images/Essential-Math-for-Data-Science/ch04_unit_circle_all.png" width="100%" alt="Different kind of norms" title="Different kind of norms">
<em>Different kind of norms</em>


#### Ch07. Matrices and Tensors

In Chapter 05, you'll learn all you need about matrices. Along with Chapter 04, it makes the foundations of linear algebra, that we'll use in the next chapters.

<img src="../../assets/images/Essential-Math-for-Data-Science/matrix-vector-dot-product-weights.png" width="100%" alt="Illustration of the dot product between a matrix and a vector" title="Illustration of the dot product between a matrix and a vector">
<em>Illustration of the dot product between a matrix and a vector</em>


#### Ch08. Span, Linear Dependency, and Space Transformation

In Chapter 04 and 05, we considered vectors and matrices as lists of numbers and geometric representations of these numbers. The goal of Chapter 06 it to go one step ahead and develop the idea of matrices as linear transformations. We'll also cover the major notions of linear dependency, subspaces and span.

<img src="../../assets/images/Essential-Math-for-Data-Science/linear-combination-two-vectors.png" width="100%" alt="Projection" title="Projection">
<em>Projection</em>






#### Ch09. Systems of Linear Equations

In this Chapter, we'll see how you can use matrices and vectors to represent systems of equations and leverage what we learned so far to understand the geometry behind it. We'll also dive into the concept of projection and see how it relates to systems of equations.

<img src="../../assets/images/Essential-Math-for-Data-Science/projection-in-column-space-with-b.png" width="100%" alt="Projection" title="Projection">
<em>Projection</em>





#### Ch10. Eigenvectors and Eigenvalues

In Chapter 08, we'll use many linear algebra concepts from previous chapters to learn about a major topic: eigendecomposition. We'll develop intuition about change of basis to understand it, and see its implication in data science and machine learning.

<img src="../../assets/images/Essential-Math-for-Data-Science/ch08_change_of_basis.png" width="100%" alt="Change of Basis" title="Change of Basis">
<em>Change of Basis</em>




#### Ch11. Singular Value Decomposition

Chapter 09 is the last chapter of Part 2 on linear algebra. You'll see what is the Singular Value Decomposition or SVD, how it relates to eigendecomposition, and how it can be understood geometrically. We'll see that it is a great method to approximate a matrix with a sum of low rank matrices.

<img src="../../assets/images/Essential-Math-for-Data-Science/ch09_full_thin_truncated_svd.png" width="100%" alt="Full, Thin and Truncated SVD." title="Full, Thin and Truncated SVD.">
<em>Full, Thin and Truncated SVD.</em>



<img src="../../assets/images/Essential-Math-for-Data-Science/ch09_SVD_geometry.png" width="100%" alt="SVD Geometry" title="SVD Geometry">
<em>SVD Geometry</em>



Part 3 is still in progress and will be about Statistics and Probability. Stay tuned to get the last new about the book!

Feel free to send me your feedbacks/opinions/considerations on this topic, I'll be very happy to discuss about it!
 -->
