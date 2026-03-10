---
layout: post
title: What most people call Predictive Coding isn't actually Predictive Coding
subtitle: How biological plausibility is hurting PC at its core
---

For the last two years, I've been working on Predictive Coding (PC), a niche learning algorithm that serves as an alternative to backpropagation. From an ML perspective, [PC is really bad](https://arxiv.org/abs/2407.01163): it scales poorly, it's very slow, and it almost never beats backprop. But all of that doesn't matter to the niche, because PC has one cool property that backprop will never have: ✨*biological plausibility*✨

That means: "compatibility with (at least some of) the many[^brain_constraints] physical and biological constraints that the brain faces". In fact, PC is foremost a [popular](https://www.lesswrong.com/w/predictive-processing) [neuroscience](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/whatever-next-predictive-brains-situated-agents-and-the-future-of-cognitive-science/33542C736E17E3D1D44E8D03BE5F4CD9) [candidate](https://www.sciencedirect.com/science/article/pii/S0896627318308572) for "the algorithm of the brain", and was only later [repurposed](https://pubmed.ncbi.nlm.nih.gov/28333583/) as an ML algorithm.

[^brain_constraints]: Constraints of the brain include: decentralized communication, asynchronous operation, localized parameters (no copying), limited activation range, propagation delays, spatial packing of the neurons and synapses, etc.

In this blogpost, I'll explain why **biological plausibility is the root of all problems for PC**, at least in Machine Learning. Moreover, these problems, when not properly handled (as is commonly the case), give rise to a _different_ algorithm that is neither PC nor bio-plausible, nor useful for ML. What people call PC is often not PC. At the end, I'll provide some tips on how to detect and avoid this pseudo-PC, and how to do _actual_ PC on a GPU.

>_This post bundles the insights from our [ePC](https://arxiv.org/abs/2505.20137) paper in a non-technical way. If you're looking for a more technical, ML-based interpretation of PC, check out my [other blogpost](https://cgoemaere.github.io/2025-07-23-predictive-coding-local-losses/)._

## The curse of biological plausibility
Before digging into PC specifically, let us first look at the broader class of "biologically-plausible AI algorithms" to see why they almost always underperform their Deep Learning counterparts.

Often, the goal of such models is to suggest a candidate for brain function, sometimes coming from the naive idea that we can derive "the algorithm of the brain" from first principles only. So let's take it one step further: say we knew the _exact_ algorithm the brain uses. Could we simulate it with a PyTorch implementation?

Probably, but it'd be terribly slow. Definitely much slower than a brain -- which is essentially single-purpose hardware tailored to this specific algorithm.[^brain_HW_algo] In fact, the brain and a GPU have such vastly different architectures and constraints, that any algorithm well-suited to one substrate is almost certainly ill-suited to the other. This hardware-algorithm mismatch is fundamental, hurting basically all bio-plausible algorithms by construction.

[^brain_HW_algo]: Actually, it's the other way around! The brain's hardware is not just tailored to the algorithm, it _defines_ it.

This forms the essence of what I like to call "the curse of biological plausibility": **if it's bio-plausible, it'll likely run poorly on a GPU and [therefore](https://arxiv.org/abs/2009.06489) no one will use it.**

Of course, if we had specialized brain-like hardware, a bio-plausible algorithm might (easily) outperform standard Deep Learning on GPUs. But who wants to build expensive custom hardware for an algorithm no one uses? Ultimately, we end up at a hardware lottery deadlock.

![Venn diagram showing gap between brain-efficient and GPU-efficient algorithms](https://raw.githubusercontent.com/cgoemaere/cgoemaere.github.io/refs/heads/master/assets/img/what_most_call_pc_is_not_pc/hw_algo_venn.png)
_A silly figure to reiterate: good brain algorithm ⟹  bad GPU algorithm_

## The (cursed) brain-inspired design of Predictive Coding
Time for a case study of the curse. Seeking biological plausibility, PC was _specifically designed_ to operate under the following four constraints:
1. **Strictly local interactions**: neurons can only talk directly to their neighbors. They are not aware of what's happening outside of their small inner circle. Global signals are passed on from one neuron to the next, like in a [telephone game](https://www.wikihow.com/Play-the-Telephone-Game).
2. **Distributed neuron activations** _(spatial freedom)_: there's no external controller telling which neurons may (not) react. All neurons are actively reacting to incoming signals.
3. **Asynchronous neuron activations** _(temporal freedom)_: there's no clock signal telling neurons when to fire. Neurons are free to "fire at will". All dynamics are continuous-time.
4. **Bidirectional communication**: if neuron A can talk to neuron B, then neuron B can also talk to neuron A, using the same channel. Technically speaking, the brain does not have this constraint (axons are unidirectional), but most physical systems do (e.g., electricity can flow both ways in a wire).

Ironically, it's precisely the combination of these four constraints that causes trouble for PC when simulated on a GPU.

Any numerical simulation requires time discretization, where time advances in discrete steps instead of continuously. But in PC, if the time step is too large, a reverberation may occur where two neighboring neurons are repeating and amplifying each other's messages, leading to an exploding signal. Therefore, **a stable PC simulation requires a small time stepsize** _(for PC experts: this corresponds to the update learning rate for the states)_. As a result, instead of amplifying messages, the simulation is *actively weakening* every exchange between two neurons. However, this is highly problematic for our telephone game: the signal quickly fades as it tries to propagate through the network.

To ensure all signals have reached their destination at full strength, **we have no other choice but to use an immense number of simulation steps.** This is exactly the reason why PC is so slow and doesn't scale to deeper networks: with more layers in the telephone game, the required number of simulation steps increases *exponentially*.

## How (not) to simulate Predictive Coding

### Pseudo-PC: when fast PC is not PC anymore

"Surely, Predictive Coding can't be *that* cursed. I mean, just look at [this paper](https://iclr.cc/virtual/2025/poster/28118): they train VGG7 on TinyImageNet in under half an hour, using PC!"

Ah yes. While [the](https://direct.mit.edu/neco/article/29/5/1229/8261/An-Approximation-of-the-Error-Backpropagation) [vast](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011280) [majority](https://pmc.ncbi.nlm.nih.gov/articles/PMC7610561/) [of](https://arxiv.org/abs/2212.00720) [PC](https://direct.mit.edu/neco/article-abstract/34/6/1329/110646/Predictive-Coding-Approximates-Backprop-Along) [papers](https://openreview.net/forum?id=Ryy7tVvBUk) just sticks to shallow models (≤5 layers), there are some that aim ([much](https://openreview.net/forum?id=lSLSzYuyfX)) bigger. So, how do they do it? How do they survive the telephone game catastrophe?

They don't. Whereas PC demands a strict equilibrium, at which all signals have settled to their final values, these papers instead terminate the algorithm much much sooner, basically when *any* signal has traversed the network. Of course, this is much faster than waiting for the correct solution, but importantly, it's not the correct solution. **Yes, their code is fast, but it gives the wrong answer.** It's still a valid learning algorithm, but it's just not PC.

I guess everyone in the community would agree that this approach does not provide _exact_ solutions for PC (i.e., we're not exactly at equilibrium), but the question is always: how far off are we? How different is this fast pseudo-PC from regular PC?

Unfortunately, the answer depends on virtually *all* hyperparameters: architecture, learning rate, optimizer, data distribution, etc etc. Depending on the setup, pseudo-PC may be entirely different from actual PC, and there's no practical way to quantify the mismatch. **To simulate exact PC, you'd need over 1000× more iterations than typically provided.**[^lots_of_iters_PC] To put this in perspective, if pseudo-PC takes half an hour per run, exact PC would take over three months.

[^lots_of_iters_PC]: In Appendix D of our [ePC paper](https://arxiv.org/abs/2505.20137), we demonstrate how a linear 20-layer PC-MLP takes >10.000 iterations to reach equilibrium. In unpublished results, we find that this dramatically increases for non-linear models, reaching well over 100.000 iterations. A common heuristic in PC papers is _#iters ~ 1-2× #layers_, meaning that these networks would get only 20-40 iterations instead of the required 100.000.

Okay, so it's a shortcut, why is that so bad? Well, **pseudo-PC is no longer biologically plausible.** By artificially cutting time short, it tacitly imposes an external controller to globally and simultaneously disable all neurons after a predefined time window, thereby violating PC's second and third assumptions ('spatiotemporal freedom').

This is a logical consequence of the curse of biological plausibility. If it runs well on a GPU, it's probably not bio-plausible. And that's exactly what's happening here.

---

{: .box-note}
### Tip: how to detect and avoid pseudo-PC
* **When reading a paper**, immediately check on the number of inference steps (#iters) vs the depth of the network (#layers). A good heuristic is that _#iters ≥ #layers^3_ (yes, cubed!) to get somewhat exact PC. Otherwise, it's likely pseudo-PC, as is often the case.
* **For your own experiments**, don't rely on this handwavy heuristic. Always monitor the convergence / energy *per layer* throughout inference. Do not rely on _global_ convergence! This will deceive you into thinking everything has already equilibrated, when in reality, the signal simply hasn't reached the deepest layers yet. *(This is illustrated in Fig 1.left of the [ePC paper](https://arxiv.org/pdf/2505.20137))*
* **In general**, if something seems too good to be true, it probably is. Super speedy PC algorithm? Insane backprop-like results? Magic convergence after only 2 steps? There's always a tradeoff, whether the paper mentions it or not.

---

### Breaking the curse for PC: ditch the brain argument, embrace the GPU
If PC is cursed, perhaps we should abandon it?

Not quite. In an amazing coincidence, PC turns out to be quite an extraordinary algorithm -- perhaps even unique. Aside from the well-established bio-plausible formulation, we discovered a _second_, fully equivalent PC formulation that is _extremely_ well-suited to GPUs. One algorithm, two formulations -- each adapted to a different kind of hardware.

This GPU-enabled PC formulation, called [ePC](https://arxiv.org/abs/2505.20137), bypasses the curse of biological plausibility and enables digital PC simulations at 100-1000x the speed. I'll spare you the technical details, but the gist of it is that backprop (which breaks all four constraints) is an excellent signal carrier, which we can exploit to speed up PC simulations and stop playing the telephone game entirely. This also means that ePC can train *much* deeper PC networks than previously possible.

![Same Venn diagram, showing how PC can be both brain-efficient (sPC) and GPU-efficient (ePC)](https://raw.githubusercontent.com/cgoemaere/cgoemaere.github.io/refs/heads/master/assets/img/what_most_call_pc_is_not_pc/sPC_ePC_venn.png)
_PC has two equivalent formulations: one for brains, one for GPUs_

The price we pay is that ePC is *not* biologically plausible anymore, breaking all four of PC's original constraints. This is a feature: if you want decent speed on GPU, you need to make this tradeoff. Yet notably, ePC still emulates the _outcome_ of a bio-plausible algorithm. That is what makes PC unique: we can use ePC to study PC's learning dynamics in simulation, gaining insights that are transferable across formulations and hardware settings. I realize it's counterintuitive to use a backprop-based algorithm for research in computational neuroscience and neuromorphic hardware. But with the curse of biological plausibility, you really have no other choice.

## Conclusion
### Brief recap
If you're working on Predictive Coding, you know how slow it is. You might be considering using fewer iterations, a common shortcut in the literature. But this is a false solution: it secretly changes the algorithm and its properties.

The reason PC is so slow, is because of a mismatch with the underlying hardware: PC was designed for brains, not GPUs. Luckily, we can also formulate PC in a way that does align well with GPUs, necessarily forgoing compatibility with the brain. This new formulation, ePC, finally enables large-scale simulations of exact PC, allowing you to explore PC's true properties without taking shortcuts.

### Biological plausibility or algorithmic correctness? Choose one.
So then, how should one simulate PC? Well, it depends on what you care about:

1. **Biological plausibility**: the original motivation of PC and what most works focus on.
Unfortunately, this form of PC is inherently cursed on GPU, so you'll either have to stick to tiny models or accept long compute times. Resist the temptation to use fewer iterations: this pseudo-PC is likely not biologically plausible and differs from exact PC, thereby distorting your conclusions.

2. **Algorithmic correctness**: exact PC simulations on a GPU, made practical with ePC.
If you care about the properties of PC's learning dynamics at a larger scale, absolutely use ePC. It will make your life so much easier. There's no loss here: any findings for ePC also apply to the biologically plausible version of PC.

### Outro
For me, this has been quite a research journey, from discovering that a common practice is actually wrong and harmful, to understanding why that is and coming up with a solution. Although I'm proud of the outcome so far, it's not over yet.

Surprisingly, this has been a hard story to sell, even though it's true and in my opinion profound. Many people dogmatically dismiss the idea of a non-bio-plausible PC algorithm, despite recognizing the issues that come with the original version. And so, without fully realizing it, they put themselves in an awkward position of having to choose: stick to what they *know* is wrong or go with what they *feel* is wrong. And while the latter is uncomfortable, the former is just absurd.