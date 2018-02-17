---
bg: "tools.jpg"
layout: post
mathjax: true
title:  "Writing academic documents with Markdown + Pandoc + Sublime Text 3"
crawlertitle: "Writing academic documents with Markdown + Pandoc + Sublime Text 3"
summary: "Use Pandoc to convert Markdown to academic documents"
date:   2018-01-05 20:09:47 +0700
categories: posts
tags: ['markdown', 'sublime-text', 'pandoc']
author: hadrien
---

Workflow to write easily in Markdown and get the beauty of Latex.

# Installation

## Pandoc

## Sublime text packages

### Pandoc

### Markdown

I like to use the syntax Github Flavored Markdown (GFM) from [MarkdownEditing](https://github.com/SublimeText-Markdown/MarkdownEditing).

- Non breaking space: \ + space

# Including latex commands (and load packages when it is necessary)

- Create a template file with `pandoc -D latex > default_1.tex`
- Add the package to the template
- Put the template in the right directory (`/usr/local/Cellar/pandoc/2.0.6/share/x86_64-osx-ghc-8.2.2/pandoc-2.0.6/data/templates/default_1.tex`)
- Use the template for your document with the command `"--template=default_1.tex"`. For using it with the sublime text package, you can add it to the used transformation (for example `PDF TOC`) in the user setting file of Pandoc.
- Just add the latex command in the text (don't use `$$` that will try to use the package `amsmath` to compile it)

## Figures side by side with captions

As for now, Pandoc doesn't allow to put figures side by side and still using a caption. The only solution is to do this

```
![Masqueur](images/maskingV2/toneCloud.pdf){width=50%}\ ![Masqueur+cible](images/maskingV2/toneCloudWithTarget.pdf){width=50%}
```

but you won't have a caption because figures are considered inline.

However you can use \latex to do it. Here is the example using the package `subcaption` designed to do exactly this.

1. Load the package in the template
2. Insert the \latex command in your text

```latex
\begin{figure}
    \begin{subfigure}[b]{.5\linewidth}
        \centering
            \includegraphics{images/maskingV2/toneCloudWithTarget.pdf}
            \caption{Nuage de tons avec cible}
        \label{fig:toneCloudWithTarget}
    \end{subfigure}
    \begin{subfigure}[b]{.5\linewidth}
        \centering
            \includegraphics{images/maskingV2/toneCloud.pdf}
            \caption{Nuage de tons sans cible}
        \label{fig:toneCloudWithoutTarget}
    \end{subfigure}
    \caption{Exemple d'un nuage de tons avec et sans cible dans la condition 515~Hz. Dans les deux cas, deux bandes de fréquences au-dessus et au-dessous de la cible sont protégées.}
    \label{fig:toneCloud}
\end{figure}
```



