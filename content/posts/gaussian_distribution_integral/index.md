---
title: "Gaussian Distribution Integral"
date: 2020-05-14T16:22:04+05:30
description: "Integral solution for the Gaussian Distribution PDF."
math: true
plotly: true
---

## Introduction
The Gaussian Distribution is often represented as $\large{\mathcal{N}(x; \mu, \sigma)}$ where $\mu$ is the mean and $\sigma$ is the standard deviation. The Probability Density Function (PDF) for a Gaussian distribution is defined as
$$
\Large{p(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}} \tag{1}
$$
Since it's a PDF,
$$
\Large{\int_{-\infty}^{\infty}{p(x)dx} = 1} \tag{2}
$$

This post is about showing how the integral equals 1.

## Laplace's method
A convenient method for solving integrals of the form $\large\int{e^{Mf(x)}dx}$ was first shown by Pierre-Simon Laplace in his [Memoir on the Probability of the Cause of Events](https://projecteuclid.org/download/pdf_1/euclid.ss/1177013621).

Using Laplace's method, we're going to find the value of
$$
\large I = \int_{-\infty}^\infty{e^{-x^2}dx} \tag{3}
$$

First, let's see what the plot for $\large f(x) = e^{-x^2}$ looks like and come up with a rough estimate for the value.

&nbsp;

{{<plotly name="f_x" src="f_x.json" caption="Figure 1: Plot for $e^{-x^2}$">}}

The integral for $f(x)$ is just the orange shaded area for $x$ from $-\infty$ to $\infty$ (Note that this plot only shows $f(x)$ for $x$ from $-2.5$ to $2.5$, but $f(x) \rightarrow 0 \text{ for } x \ge 2.5 \text{ and } x \le -2.5$).

We can get a rough estimate for the area by drawing a triangle over the curve as shown with the Blue triangle in the above figure. The area of the triangle is $\frac{1}{2}\times{4}\times{1} = 2$. From the plot, it's fair to assume that the integral is less than 2. So, our first guess for a rough upper bound of $eq\ (3)$ is $\sim 2$.

It's hard to calculate the integral directly, so we'll go through the steps of Laplace's method. The first step of Laplace's method is to multiply $eq\ (3)$ by itself.

$$
\large
\begin{aligned}
I^2 &= \Bigg(\int_{-\infty}^\infty{e^{-x^2}dx}\Bigg)^2
\\\\ \\\\ &= \Bigg(\int_{-\infty}^\infty{e^{-x^2}dx}\Bigg)\Bigg(\int_{-\infty}^\infty{e^{-x^2}dx}\Bigg)
\\\\ \\\\ &=  \Bigg(\int_{-\infty}^\infty{e^{-x^2}dx}\Bigg)\Bigg(\int_{-\infty}^\infty{e^{-y^2}dy}\Bigg)
\\\\ \\\\ &= \int_{-\infty}^\infty{\int_{-\infty}^\infty{(e^{-x^2})(e^{-y^2})dxdy}}
\\\\ \\\\ &= \int_{-\infty}^\infty{\int_{-\infty}^\infty{e^{-(x^2+y^2)}dxdy}} \tag{4}
\end{aligned}
$$

&nbsp;

Basically, $I^2$ is the volume under the surface of $f(x,y) = e^{-(x^2+y^2)}$. Let's take a look at the surface plot $f(x, y)$ below:

&nbsp;

{{<plotly name="f_x_y" src="f_x_y.json" caption="Figure 2: Plot for $e^{-(x^2+y^2)}$">}}

From the plot, we can see that $I^2$ is the volume under the cone like surface. The surface is pretty much encompassed by a cone with the base centered at 0, radius $r = 2$ and height $h = 1$ (You can hover over the 3D plot to see circles emulating the base of the cone). The volume of the cone is then given by $\pi r^2\frac{h}{3} = \frac{4}{3}\pi$. So, our second guess for the upper bound of $eq\ (3)$ is $\sqrt{\frac{4\pi}{3}} \sim 2$, which is pretty close to the first guess.

It's hard to calculate this double integral as well, so let's move on to the second step of the method. Every point in the $x-y$ plane can also be expressed in polar co-ordinates $r,\theta$ as
$$
x = rcos\theta\qquad y = rsin\theta \tag{5}
$$
$$
r^2 = x^2 + y^2 \tag{6}
$$

The $dxdy$ in $eq\ (4)$ represents an infinitesimal area in the $x-y$ plane. The same area in the polar plane is given by $rdrd\theta$. Converting $eq\ (4)$ to polar co-ordinates, we get
$$
\large
\begin{aligned}
I^2 &= \int_0^\infty{\int_0^{2\pi}{e^{-r^2}rdrd\theta}}
\\\\ \\\\ &= \int_0^\infty{e^{-r^2}rdr\int_0^{2\pi}{d\theta}}
\\\\ \\\\ &= 2\pi\int_0^\infty{e^{-r^2}rdr}
\\\\ \\\\ &\text{Using the substitution }\ u = x^2 \text{, we get } du = 2xdx
\\\\ \\\\ &= \pi\int_0^\infty{e^{-u}du}
\\\\ \\\\ &= \pi(-e^{-u}{\Large|}_0^\infty)
\\\\ \\\\ &= \pi\tag{7}
\end{aligned}
$$

Taking the square root of $eq (7)$, we get the value for $eq (3)$:
$$
\large
I = \sqrt{\pi} \tag{8}
$$

So the area under $e^{-x^2}$ is $\sqrt{\pi}\approx1.772$. Our upper bound guesses of 2 were a little far off, but gave a good geometrical intuition. $Eq\ (8)$ can be generalized further in the following way:
$$
\large
\int_{-\infty}^\infty{e^{-mx^2}dx} = \sqrt{\frac{\pi}{m}} \tag{9}
$$

## Gaussian PDF Integral
We can now apply $eq(9)$ to $eq(2)$ as follows:
$$
\large
\begin{aligned}
I &= \int_{-\infty}^{\infty}{\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}dx}
\\\\ \\\\ &= \frac{1}{\sqrt{2\pi\sigma^2}}\int_{-\infty}^{\infty}{e^{-\frac{1}{2\sigma^2}(x-\mu)^2}dx}
\\\\ \\\\ &\text{Substituting } x - \mu = u \text{ and using } eq(9)
\\\\ \\\\ &= \frac{1}{\sqrt{2\pi\sigma^2}}{\sqrt{\frac{\pi}{\frac{1}{2\sigma^2}}}}
\\\\ \\\\ &= 1\tag{10}
\end{aligned}
$$

I hope the constant $\frac{1}{\sqrt{2\pi\sigma^2}}$ in the PDF for the Gaussian Distribution now makes sense. Laplace's method can also be used to show that $E[x] = \mu$ and $Var(x) = \sigma^2$. However, those integrals are a little bit more complicated than $eq(10)$.

## Further Learning
In this post, we showed that the Gaussian Distribution function is indeed a probability density function. The Gaussian Distribution function also has many other amazing properties which make it a popular choice for many Machine Learning modeling tasks. To learn more about these properties, I recommend watching [Probabilistic ML - Lecture 6 - Gaussian Distributions](https://www.youtube.com/watch?v=FIheKQ55l4c&list=PL05umP7R6ij1tHaOFY96m5uX3J21a6yNd&index=7).

## Remarks
I first learned about Laplace's method from [Gaussian Integrals](http://www.umich.edu/~chem461/Gaussian%20Integrals.pdf). I've been trying to sharpen the mathematical skills needed for Machine Learning, and this was good practice for integration. This post also helped me come up with an easy process to embed interactive plotly figures. See the Hugo shortcode for plotly here - https://github.com/hemildesai/dev/blob/master/layouts/shortcodes/plotly.html.