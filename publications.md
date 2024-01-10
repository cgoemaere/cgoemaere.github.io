---
layout: page
title: "Publications"
head-extra: mathjax.html
---

Here is a list of my current publications. Be sure to check out my [Google Scholar](https://scholar.google.be/citations?user=4BQ4DZsAAAAJ) page for a (probably) more updated version.

**Accelerating Hierarchical Associative Memory: A Deep Equilibrium Approach** (2023) \
**<font color='green'>NeurIPS 2023 - AMHN workshop</font>**  [**[ArXiv]**](https://arxiv.org/abs/2311.15673)  [**[OpenReview]**](https://openreview.net/forum?id=Vmndp6HnfR) \
**Cédric Goemaere**, Johannes Deleu, Thomas Demeester

We show that (Hierarchical) Associative Memory can be cast as a Deep Equilibrium Model. Moreover, we identify and resolve a redundancy in synchronous updates of HAMs, and show that our solution boils down to parallellizing asynchronous updates.

**Exploring the Temperature-Dependent Phase Transition in Modern Hopfield Networks** (2023) \
**<font color='green'>NeurIPS 2023 - AMHN workshop</font>**  [**[ArXiv]**](https://arxiv.org/abs/2311.18434)  [**[OpenReview]**](https://openreview.net/forum?id=AXiMq2k4cb) \
Felix Koulischer, **Cédric Goemaere**, Tom Van Der Meersch, Johannes Deleu, Thomas Demeester

We investigate the role of the temperature parameter $\beta$ in Modern Hopfield Networks, and identify two behavioral regimes, with a phase transition determined by a critical temperature $\beta_c$. \
*This work stems from Felix's Master's thesis, which I supervised together with Thomas.*

**Efficient Keyword Generation using Pretrained Language Models** (2022) \
**<font color='green'>Master's thesis</font>**  [**[University Library Ghent]**](https://lib.ugent.be/en/catalog/rug01:003063469) \
**Cédric Goemaere** \
**<font color='green'>BNAIC/BeNeLearn 2022 - Thesis abstract</font>**  [**[Proceedings]**](https://bnaic2022.uantwerpen.be/wp-content/uploads/BNAICBeNeLearn_2022_submission_4135.pdf) \
**Cédric Goemaere**, Thomas Demeester, Tim Verbelen, Bart Dhoedt, Cedric De Boom

My Master's thesis looks into the problem of keyword generation: given a context prompt, generate a fluent completion that contains a pre-specified keyword. In contrast to previous methods that looked into fine-tuning (parts of) a pretrained LM, I show that keyword generation can be achieved more efficiently by working directly on the logits of the unmodified LM. I propose 4 simple models: one based on a target-specific prior, another based on the FastText similarity between the target and (every token in) the vocabulary[^masterthesis1], and finally two combinations of these two. Training happens on a per-target basis, where the loss maintains the likelihood of the generated sentence if it contains the keyword, or drives it to zero otherwise[^masterthesis2]. After training, the generated sentences contain the target keyword in 10-50% of the cases (depending on the chosen model and target), with minimal degradation in fluency. Thanks to their simplicity, the trained models can easily be interpreted, providing more insight into their inner workings.

*An example of keyword generation: complete the sentence "As I was walking across the " such that it contains the keyword "analysis". While there are a huge number of possible solutions, this is still quite a hard problem to solve. One correct solution would be "As I was walking across the lab, I noticed the computer had already finished its analysis of the substrate."*

___
[^masterthesis1]: Fun fact: this model contains only 65 trainable parameters!
[^masterthesis2]: In hindsight, this is equivalent to using [DPO](https://arxiv.org/abs/2305.18290) and setting the reward $r$ to zero if the sentence $s$ contains the keyword $k$, and to $-\infty$ otherwise (i.e., $r=\log(I(k \ in s))$).
