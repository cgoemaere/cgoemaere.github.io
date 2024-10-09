---
layout: page
title: "Publications"
head-extra: mathjax.html
---

Here is a list of my current publications. Be sure to check out my [Google Scholar](https://scholar.google.be/citations?user=4BQ4DZsAAAAJ) page for a (probably) more updated version.

**Accelerating Hopfield Network Dynamics: Beyond Synchronous Updates and Forward Euler** (2024) \
**<font color='green'>ECAI 2024 - ML-DE workshop</font>**  [**[ArXiv]**](https://arxiv.org/abs/2311.15673v2)  [**[PMLR]**](https://proceedings.mlr.press/v255/goemaere24a.html)  [**[GitHub]**](https://github.com/cgoemaere/hopdeq) \
**Cédric Goemaere**, Johannes Deleu, Thomas Demeester

This work greatly expands the scope of the NeurIPS workshop paper below. It provides an accessible introduction to Hopfield networks and also considers the case of the Continuous Hopfield network (which we prove are just fancy HAMs). The theoretical analysis is much more substantial and rigorous, and we added a ton more experimental results (which required some tinkering with the learning rates).

**Accelerating Hierarchical Associative Memory: A Deep Equilibrium Approach** (2023) \
**<font color='green'>NeurIPS 2023 - AMHN workshop</font>**  [**[ArXiv]**](https://arxiv.org/abs/2311.15673v1)  [**[OpenReview]**](https://openreview.net/forum?id=Vmndp6HnfR)  [**[GitHub]**](https://github.com/cgoemaere/hopdeq/tree/NeurIPS23_AMHN_workshop_code) \
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

My Master's thesis looks into the problem of keyword generation: given the start of a sentence, generate a fluent completion that contains a pre-specified target keyword. \
In contrast to prior work on fine-tuning (parts of) a pretrained LM, I show that keyword generation can be achieved more efficiently by working directly on the logits of the unmodified LM. I propose 4 simple, interpretable models: one based on a target-specific prior, another based on the FastText similarity between the target and (every token in) the vocabulary, and finally two combinations of these two.

**Example**: complete the sentence "*As I was walking across the*" such that it contains the keyword "*analysis*". One correct solution would be "*As I was walking across the* lab, I noticed the computer had already finished its *analysis* of the substrate."

**Fun fact**: the second model contains only 65 trainable parameters and demonstrates zero-shot generalization

**Afterthought**: the simple training procedure I used is mathematically equivalent to using [DPO](https://arxiv.org/abs/2305.18290) where the reward $r$ is zero if the sentence $s$ contains the keyword $k$, and $-\infty$ otherwise (i.e., $r=\log(\mathbb{1}(k \in s))$ ).
