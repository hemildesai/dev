---
title: "My Interests - 2020"
date: 2021-08-12T19:39:52-07:00
draft: false
description: "My areas of interest during 2020."
---

This is a new series I'm experimenting with. This series is an effort to document the areas I was/am interested in during the timeframe mentioned (hopefully in a chronological order), the reasoning behind those interests, the efforts taken to learn and build upon those interests and the reasoning to pursue them further or diverge from them. I hope this journaling helps me to understand how my thoughts evolve and also reflect on the benefits of pursuing various interests (either diving deep into a few or just exploring them on the surface). This series will also let others know what I'm currently into and how I got there. Finally, I also plan to include links to resources I found helpful while learning about those interests so at the very least this series provides a collection of resources for various different topics.

## January - March
I had just finished my ML Platform Engineer role and moved back to India. So these months were not the most productive of the year. But I tried to keep up with Kubernetes, its cloud offerings by Amazon and Google and ML Platforms on top of K8s like Kubeflow.
### Links
- [Deconstructing Kubernetes](https://www.youtube.com/watch?v=90kZRyPcRZw)
- [Kubernetes Design Principles](https://www.youtube.com/watch?v=ZuIQurh_kDk)
- [Building and Managing a Centralized Kubeflow Platform at Spotify](Building and Managing a Centralized Kubeflow Platform at Spotify)
- [Towards Continuous Computer Vision Model Improvement with Kubeflow](https://www.youtube.com/watch?v=9UPnCo-LG04)

## April - July
By this time, there was a complete lockdown in my home city. I had also received an admit from UCLA to pursue my Master's in CS starting Fall 2020 (which got pushed to Winter 2021). My interest in ML had been growing till this point, and I wanted to try out research during my Masters program. So I started diving deep into the Math and foundations of ML and Deep Learning. I also got interested in Probabilistic and Bayesian Machine Learning and tried to understand the mathematical part of the topics. Another topic of major interest was unsupervised learning. I watched a few lecture series and talks pertaining to these topics and tried implementing some basic projects.

### Links
- [Deep Learning with PyTorch NYU](https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI)
- [Information Theory, Pattern Recognition, and Neural Network - David MacKay](https://www.youtube.com/playlist?list=PLruBu5BI5n4aFpG32iMbdWoRVAA-Vcso6)
- [Deep Unsupervised Learning](https://sites.google.com/view/berkeley-cs294-158-sp20/home)
- [Andrew Gordon Wilson's Talks](https://www.youtube.com/watch?v=E1qhGw8QxqY)
- [Yarin Gal's talks on Bayesian Deep Learning](https://www.youtube.com/watch?v=G6tUZRHnJYc)

## August - September
I took a slight detour during these months and dived into full-stack web development again due to a project I was working on with my friends. I am grateful that I took this detour and worked on the project because I got to learn a lot about frontend development (I was focused primarily on frontend). I finally worked on a real React project and got to a comfortable level with it. Also learned a bit about design (I wish to dive a little deeper into design. Design is very important to me!). I read up on the frontend ecosystem and learned a bunch of new things. On the backend side, I also learned about the latest in serverless technologies and got to experiment a bit with Aurora Serverless. I also got to refresh my cloud networking concepts while setting up the Lambda to Aurora connection.

### Links
I mostly used docs and articles on a per needed basis. Some of the technologies I used/learned were:
- https://reactjs.org/
- https://aws.amazon.com/rds/aurora/serverless/
- https://redwoodjs.com/
- https://tailwindcss.com/
- https://www.prisma.io/
- https://graphql.org/

## October - December

This was a busy period coordinating the logistics for my masters program. So essentially I only worked on a few blog posts and learned a bit about fourier transforms.

### Links
- [Fourier Analysis](https://www.youtube.com/playlist?list=PLMrJAkhIeNNT_Xh3Oy0Y4LTj0Oxo8GqsC)

## Summary
Over the course of the year, I learnt a lot about the foundations of the latest innovations in Deep Learning and Machine Learning. I got a chance to dive deep into Unsupervised learning, Supervised learning, Generative Modeling and Probabilistic ML. I went from the point of being completely lost in a research paper to being able to grok most of the new papers. I worked on some practical projects limited by the compute available to me. Paralelly, I continued exploring infrastructure projects and large scale systems - learning about new projects. I got to explore a bit of frontend development and reached a comfortable point with React (which I am quite happy about).

Since the majority of my time was spent around ML, I noticed some trends (these are my personal opinions) which might have started a pivot in my interests and what I want to pursue at this moment.
- Most of the new techniques and innovations in terms of algorithms have converged to a few neural network technologies - like CNNs and Transformers. More recently, transformers seem to be taking over Vision as well. NLP as such is largely dominated by transformers. As a result, progress is seen more in terms of the size of the models, the size of data and compute and the distributed systems techniques for large scale learning. Bigger the number of GPUs, better the results.
- There has been a lot of great progress in Unsupervised learning and Self-supervised learning. But these models are enormous and require large amounts of compute and data. As such, as an individual I found it very limiting to work on practical projects related to these areas. And this is seen in the research papers as well, most of them are from large organizations with access to enormous amounts of compute.
- Getting these larger models to train is in itself a very challenging problem involving distributed systems, networking, parallel programming, etc. And this challenge excited me a lot (more on this in the next part).
- As for Probabilistic ML, while I may not be pursuing it deeper at the moment, the tools and techniques I learned are very important for real world applications of ML and I now know where to find the right tools to incorporate the benefits of probabilistic and bayesian ML.
- While research continues at a staggering pace, the tools to transfer the research into practical applications are not advancing so fast. For NLP, we have seen how revolutinizing HuggingFace Transformers has been. We need more such tools that enables large scale adoption of these amazing Deep Learning techniques. For instance, there's tons of research on network pruning, quantization, distillation, etc to boost inference but practical tools for adopting these to real world models are not so ubiquitous. Developing such tools has also piqued my interest recently.

I am very grateful for the effort I invested in learning these topics and I know the skills I learned will help with whatever I do in the future. In the next post, I will write about my interests in 2021 and where I currently stand in terms of what I want to pursue further.