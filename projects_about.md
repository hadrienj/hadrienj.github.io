---
bg: "about1.jpg"
layout: page
title: "About"
crawlertitle: "About"
permalink: /about/
summary: "hadrienj. <br>PhD student . Cognitive Science <br>ENS - Paris, France"
active: about
---

<p class="intro">
    <div class="avatar">
      <div>
        <a href="/about">
        {% include avatar.html %}
        </a>
      </div>
      <div class="avatar-txt">
        Previously working as a Machine Learning Scientist at <a href="https://www.ava.me/">Ava Accessibility</a>.<br>

        I finished my PhD in Dec. 2018 in Cognitive Sciences at the <img class="inline-icon" src="../../assets/images/icons/ens_crop.png" width="30" alt="ens icon" title="ENS"> &Eacute;cole Normale Supérieure (ENS) in Paris, France on auditory perceptual learning (pitch perception and auditory selective attention) using psychophysics and electrophysiology (EEG).
      </div>
    </div>
  </p>

<h2 class='about-h2'>Data Science</h2>

I am currently managing a project for bird detection using deep learning with the non profit organization Wazo in Paris. This project has been selected in the season 06 of DataForGood Paris from September to December 2019.

I am also bloging here on mathematics for machine learning and deep learning. I think that computer science is a great way to learn theoretical knowledge with a practical approach.

I worked on creating and mainting machine learning pipelines for speaker diarization from multi microphone signals.

I used R to analyse behavioral data and create vizualisations and Python to analyse EEG data (see [my toolbox](https://github.com/hadrienj/EEG) for EEG processing) and elaborate offline/online signal processing workflow.

At the corner of data science and web developement, I created the skeleton of a neurofeedback app that streamed and transfered the data from the EEG system to a web server in Django and get the data in the browser with web sockets for final feedback display.


<div class='card-section'>
    <div class='skills'>
        <div class='skills-col'>
            <div class="skills-cat">Programming</div>
            <div class="skills-item">
                <img src="../../assets/images/icons/python.png" width="30" alt="python icon" title="Python">
                Python
            </div>
            <div class="skills-item">
                <img src="../../assets/images/icons/r.png" width="30" alt="r icon" title="R">
                R
            </div>
        </div>
        <div class='skills-col'>
            <div class="skills-cat">Visualization</div>
            <div class="skills-item">
                <img src="../../assets/images/icons/matplotlib.png" width="30" alt="Matplotlib icon" title="Matplotlib">
                Matplotlib
            </div>
            <div class="skills-item">
                <img src="../../assets/images/icons/d3.png" width="30" alt="D3 icon" title="D3">
                D3
            </div>
            <div class="skills-item">
                <img src="../../assets/images/icons/ggplot2.png" width="30" alt="GGPlot2 icon" title="GGPlot2">
                GGPlot2
            </div>
        </div>
        <div class='skills-col'>
            <div class="skills-cat">ML/DL</div>
            <div class="skills-item">
                <img src="../../assets/images/icons/sklearn.png" width="30" alt="sklearn icon" title="Scikit Learn">
                Scikit Learn
            </div>
            <div class="skills-item">
                <img src="../../assets/images/icons/keras.png" width="30" alt="keras icon" title="Keras">
                Keras
            </div>
        </div>
        <div class='skills-col'>
            <div class="skills-cat">Tools</div>
            <div class="skills-item">
                <img src="../../assets/images/icons/pandas.png" width="30" alt="Pandas icon" title="Pandas">
                Pandas
            </div>
            <div class="skills-item">
                <img src="../../assets/images/icons/np.png" width="30" alt="numpy icon" title="Numpy">
                Numpy
            </div>
            <div class="skills-item">
                <img src="../../assets/images/icons/scipy.png" width="30" alt="Scipy icon" title="Scipy">
                Scipy
            </div>
        </div>
    </div>
</div>

<h2 class='about-h2'>Web Development - Full-Stack</h2>

During my PhD, I created web apps with Django and Javascript for auditory experiments running on computers and tablets. I also worked with NoSQL databases (CouchDB) hosted on a DigitalOcean and PouchDB to build offline-first web app.

I also used D3 and React to build data vizualisation on the web.

The Web app I created used the Web Audio API to create sounds with controled features on the web (for instance [a demo](https://fm-am.auditory.fr/) of amplitude and frequency modulation with visualizations).

<div class='card-section'>
    <div class='skills'>
        <div class='skills-col'>
            <div class="skills-cat">Back-end</div>
            <div class="skills-item">
                <img src="../../assets/images/icons/django.png" width="30" alt="django icon" title="Django">
                Django
            </div>
            <div class="skills-item">
                <img src="../../assets/images/icons/digitalocean.png" width="30" alt="digitalocean icon" title="Digitalocean">
                Digitalocean
            </div>
            <div class="skills-item">
                <img src="../../assets/images/icons/heroku.png" width="30" alt="heroku icon" title="Heroku">
                Heroku
            </div>
        </div>

        <div class='skills-col'>
            <div class="skills-cat">Front-end</div>
            <div class="skills-item">
                <img src="../../assets/images/icons/js.png" width="30" alt="javascript icon" title="Javascript">
                Javascript
            </div>
            <div class="skills-item">
                <img src="../../assets/images/icons/html.png" width="30" alt="html icon" title="HTML">
                HTML
            </div>
            <div class="skills-item">
                <img src="../../assets/images/icons/css.png" width="30" alt="css icon" title="CSS">
                CSS
            </div>
            <div class="skills-item">
                <img src="../../assets/images/icons/react.png" width="30" alt="react icon" title="React">
                React
            </div>
        </div>

        <div class='skills-col'>
            <div class="skills-cat">Database</div>
            <div class="skills-item">
                <img src="../../assets/images/icons/couchdb.png" width="30" alt="couchdb icon" title="CouchDB">
                CouchDB
            </div>
            <div class="skills-item">
                <img src="../../assets/images/icons/pouchdb.png" width="30" alt="pouchdb icon" title="Pouchdb">
                Pouchdb
            </div>
        </div>
    </div>
</div>


<h2 class='about-h2'>Some Projects</h2>

<article class="index-page">


  <a class='noDeco' href="/deep-learning-book-series-home">
    <div class='card-section'>
      <div class="card-section-img">
        <img src="../../assets/images/2.12/gradient-descent.png" width="400" alt="Mechanism of the gradient descent algorithm" title="Mechanism of the gradient descent algorithm">
      </div>
      <div class="card-section-text">
        <div class="card-section-text-title">
          Linear Algebra Series
        </div>
        <div class='card-section-text-description'>
          Series on linear algebra chapters from the Deep Learning Book from Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016): 12 blog posts + Python Notebooks to get all you need to do great things in machine learning and deep learning.
        </div>
      </div>
    </div>
  </a>

  <a class='noDeco' href="https://fm-am.auditory.fr/">
    <div class='card-section'>
      <div class="card-section-img">
        <img src="../../assets/images/AM-FM/demo.png" width="400" alt="Screenshot
        of amplitude and frequency modulation demo" title="Amplitude and frequency
        modulations">
      </div>
      <div class="card-section-text">
        <div class="card-section-text-title">
          Amplitude and Frequency modulations
        </div>
        <div class='card-section-text-description'>
          Interactive demo illustrating the concept of amplitude and frequency
          modulation.
        </div>
      </div>
    </div>
  </a>

  <a class='noDeco' href="https://tonecloud.auditory.fr/">
    <div class='card-section'>
      <div class="card-section-img">
        <img src="../../assets/images/tonecloud/demo.jpg" width="400" alt="Screenshot
        of tone cloud demo" title="Tone cloud with React, D3 and WebAudio">
      </div>
      <div class="card-section-text">
        <div class="card-section-text-title">
          Tone clouds
        </div>
        <div class='card-section-text-description'>
          Audiovisual represention of tone clouds. Tone clouds are sets of pure
          tones with random spectrotemporal characteristics.
        </div>
      </div>
    </div>
  </a>
</article>


<h2 class='about-h2'>Contact</h2>

<div class='connect'>
    <div>
        <img src="../../assets/images/flat_web_icon_set/color/Email.png" width="30" alt="email icon" title="Email">
    </div>
    <a href="mailto:code.datascience@gmail.com">
        <div class='connect-text'>
            Drop me an email (code.datascience@gmail.com)
        </div>
    </a>
</div>

<div class='connect'>
    <div>
        <img src="../../assets/images/flat_web_icon_set/color/LinkedIn.png" width="30" alt="linkedin icon" title="Linkedin">
    </div>
    <a href="https://www.linkedin.com/in/hadrienj/">
        <div class='connect-text'>
            Say hello on Linkedin (hadrienj)
        </div>
    </a>
</div>

<div class='connect'>
    <div>
        <img src="../../assets/images/flat_web_icon_set/color/Twitter.png" width="30" alt="twitter icon" title="Twitter">
    </div>
    <a href="https://twitter.com/_hadrienj">
        <div class='connect-text'>
            Find me on Twitter (_hadrienj)
        </div>
    </a>
</div>
