---
layout: post
title: "Why I stopped working on Hopfield networks"
---

The first year of my PhD eventually became a full deep dive into Hopfield networks. These neural networks, inspired by biological memory, promise a new direction for AI that is more biologically plausible than traditional Deep Learning — and significantly more energy-efficient. And yet, after that first year, I shifted my focus to Predictive Coding networks and haven't looked back. This post is a reflection on my journey, the allure of Hopfield networks, and why I ultimately decided to move on.

## Hopfield networks: a new hope beyond digital backprop
When I started my PhD in late 2022, a gloomy trend was revealing itself in the field of Deep Learning: the push for improved performance inevitably leads to ever greater demands in compute. The release of ChatGPT and the subsequent LLM hype took that prediction to the next level, boosting the NVIDIA stock price by over 800%.

Hopfield networks offer an energy-efficient alternative to traditional backprop-trained neural networks. Their architecture allows for an implementation in _fully analog hardware_, which is, in theory, up to [10,000x more efficient and 100x faster than GPUs](https://www.nature.com/articles/s41928-022-00869-w.epdf?sharing_token=obdo1wnSsAp-Wvr2XfRhAdRgN0jAjWel9jnR3ZoTv0PUinovJ2yNZM_TDxWphRDVaDB3OPWfk8lx-GM_9_uKl0eE6M4mksdZW7w2GvgMegP6Ch04urlmss6SmvDxO93n2JOL9_UElB7jkI8Y4OWNhoMdBwClznMAhtvKHYNnl70=).
Specifically, the following model characteristics are crucial for an analog implementation:
1. **Bidirectional operation:** A wire allows electricity to flow both ways, so the model must natively support the coexistence of feedforward and feedback signals.
2. **Asynchronous state updates:** In analog hardware, there is no clock. All states must be allowed to change at any time — unlike an MLP, where there is a clear hierarchy of consecutive state updates.
3. **Local weight updates:** It's rather easy to perform inference on analog hardware; it has already been done for most Deep Learning architectures, actually. The real gold lies in analog _training_. But backprop requires 1) sequential weight updates; 2) memorization of state dynamics; and 3) global information flow; which is all really tricky to do in analog hardware. A backprop alternative called [Equilibrium Propagation](https://arxiv.org/abs/1602.05179) (EqProp) solves this problem for Hopfield networks specifically.

Now, consider the brain. It, too, is a piece of analog hardware and must possess the above characteristics. Therefore, it is not surprising that neuroscience models of how the brain might work, come with great potential for analog implementations. The Hopfield network is a great example of this: it was originally suggested as a model for associative memory and has been linked to the CA3-region of the hippocampus.

For those interested in how Hopfield networks work exactly, I can highly recommend [this video](https://www.youtube.com/watch?v=1WPJdAW-sFo).

## Where have all the Hopfield networks gone?
At this point, you might be thinking: if Hopfield networks are so great, why aren't we using them anywhere?

The short answer is: because they suck. For now.

The long answer:
Developing analog hardware is expensive. _Very expensive!_ So you'd only want to do it for an established model that works really well. But here's the catch: without analog hardware, Hopfield networks are very slow to train. So progress is slow too and their performance remains substandard. In essence, the field is stuck in a stalemate: no one wants to develop analog hardware for a lousy model, and no one wants to try to improve an unproven model that takes ages to train.

Clearly, there are two avenues for progress: develop analog hardware or improve the model on GPU. I went for the latter, which seemed like the more accessible approach.

## My journey into the (Hop)field
Excited about their game-changing potential, I went all in on Hopfield networks. Over three intensive months, I immersed myself in the literature, reading every recent paper I could find. About six months in, I found an insightful connection to [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) (DEQs), which resulted in [a NeurIPS workshop paper](https://openreview.net/forum?id=Vmndp6HnfR) and eventually [my first full paper](https://proceedings.mlr.press/v255/goemaere24a). The gist is that a Hopfield network can (and should!) be seen as a DEQ. This is useful because the field of DEQs is already quite mature and their limitations are well studied.

This realization, together with the insights of [this eye-opening paper](https://arxiv.org/abs/2202.04557), suggested that something wasn't quite right with some of the claims being made in the field. The mathematics often seemed to obscure rather than reveal the underlying simplicity of the model. The more I learned, the more I realized that Hopfield networks might have more limitations than I originally believed.

I took my doubts with me to the very first [Hopfield networks workshop](https://amhn.vizhub.ai/) at NeurIPS 2023, hoping for some new perspectives from the community. Unfortunately, I left with most of my questions still unanswered.

## Understanding the limitations
In my effort to gain intuition around Hopfield networks, I bumped into several limitations and complications that deserve more attention in the literature. Let me walk you through them.
 
### Hopfield networks are literal databases
By definition, Hopfield networks are memory models. They store patterns as vectors in their weights and retrieve them when given a query. Want to store a new pattern? No problem! Just add it to the weights! _(I'm not kidding, this is actually how it's done)_

In that sense, a Hopfield network is no different from a database that literally stores all training patterns. Retrieval is simple: compare the given query to all memories in the database using a similarity metric (e.g., dot product is typically used) and return the most similar item (i.e., an argmax operation).

{: .box-note}
**Note:** this perfect argmax Hopfield network already exists and is very successful. It's called Vector Quantization and is used in [VQ-VAEs](https://arxiv.org/abs/1711.00937) and [VQ-GANs](https://arxiv.org/abs/2012.09841). _(though it is typically not referred to as a Hopfield network)_

For reasons that are not entirely clear to me, people have strictly avoided using the argmax operator — perhaps because it cannot be implemented in analog hardware or in the brain. In the '80s and '90s, most people used signum or sigmoid as selection operator instead of argmax. Of course, this can result in fuzzy retrievals, up to the point where adding more patterns is pointless, because there are always other spurious patterns 'close enough' to interfere. This resulted in the concept of [storage capacity](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2016.00144/full), which naturally depends on the dimensionality of the stored patterns (higher dimensionality = more space between two random patterns = less spurious patterns).

Many notable advances have dramatically increased the storage capacity, from [Dense Associative Memory (2016)](https://arxiv.org/abs/1606.01164) to [Modern Hopfield networks (2017)](https://arxiv.org/abs/1702.01929) and the widely-cited [Hopfield Networks is All You Need (2020)](https://arxiv.org/abs/2008.02217). I particularly appreciate this last paper for its excellent [blogpost](https://ml-jku.github.io/hopfield-layers/) and insights into attention mechanisms.

However, these advances are more limited than they may appear. The Modern Hopfield network uses _softmax_ instead of argmax, but essentially still works like a database: you get one-step retrieval and storage capacity that scales exponentially with dimensionality. This is not as impressive as it sounds: it just means that there are less spurious patterns (with argmax, there would be exactly 0 spurious patterns). The storage capacity still scales _linearly_ with the number of stored patterns — just like a regular database!

[Recent work](https://arxiv.org/abs/2410.24153) has started addressing this limitation by exploring memory superposition. While I didn't have the time to read it in detail, I'm glad that the problem is being acknowledged and worked on.

As an excellent case study of taking this limitation too far, let's consider the NeurIPS 2023 paper [Long Sequence Hopfield Memory](https://openreview.net/forum?id=Tz2uONpgpy). It presents a biologically plausible model of sequence memorization, which seems promising given how many of our own memories involve sequences. However, its implementation is very direct: it stores video frames in the weight matrix and, given the current frame, finds and returns the next matching frame — just like a video player. To me, it seems extremely unlikely that the brain memorizes sequences in such a literal manner: there's simply no room for that.

### Hopfield networks use _pixelwise_ comparison for image similarity
Okay, so Hopfield networks store patterns as memories to be retrieved. For retrieval, it checks which memory is the closest to the given query and returns that one. Clear and simple.

But how do you check the 'closeness'? Specifically, for image data, which is the popular choice for Hopfield networks research. As a long [history](https://dl.acm.org/doi/10.5555/197765.197784) [of](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94) [Computer](https://ieeexplore.ieee.org/document/1284395) [Vision](https://ieeexplore.ieee.org/document/4775883) [papers](https://arxiv.org/abs/2103.00020) will tell you, checking image similarity on a per-pixel basis is a recipe for disaster. Instead, you must compare their features — for instance, the output embedding of a vision model.

Yet somehow, this lesson hasn't made its way into the Hopfield networks community. Almost everyone is using pixel-wise dot product similarity and expects it to work well. To my knowledge, only [a single paper](https://ebooks.iospress.nl/doi/10.3233/FAIA230500) has explored using feature similarity, and strangely, it doesn't seem to perform that well.

I'm not sure what to think of this, but it seems like a bad idea to use pixelwise image comparison as the basis of your model.

### Hopfield networks are probably _not_ biologically plausible
While Hopfield networks were inspired by the brain, some aspects make me question their biological plausibility.

Humans are very much capable of constantly creating new memories — much to our regret, sometimes. It seems unlikely that we create entirely new independent synaptic connections for every new memory, and yet, that is exactly what Hopfield networks would do.

Another thing: when you remember something, how often do you get two memories mixed up into some weird hybrid? Not very often, right? Well, classical Hopfield networks do. A lot.

What about Modern Hopfield networks? Maybe our brain uses softmax? Probably not. Softmax is not a local operator: it needs to look at _every single memory_ at once to be able to pick the best one. Supposedly, this could be solved with astrocyte cells (which would even make [Transformers biologically plausible](https://www.pnas.org/doi/10.1073/pnas.2219150120)), because they go beyond simple two-neuron interactions. To me, that feels like a quite a stretch. 

My only possible conclusion: Hopfield networks are likely not what the brain does.

### Memory networks don't generalize

Once again, repeat after me: "Hopfield networks are memory models."
Their job is to store memories and retrieve them exactly. Any retrieval of memories not present in the training data (those "spurious patterns" I mentioned earlier) is seen as a problem to solve. How can you hope to generalize with such models?

At the Hopfield networks workshop in 2023, there were [two](https://openreview.net/forum?id=hkV9CvCOjH) [papers](https://openreview.net/forum?id=B1BL9go65H) linking Hopfield networks to Diffusion Models. It seemed ridiculous at first: Diffusion Models are amazing at generalization, while Hopfield networks explicitly try to avoid it.

But maybe I was too quick to judge. [A recent paper](https://openreview.net/forum?id=zVMMaVy2BY) suggests that these spurious patterns might actually be useful, as they mark the onset of generalization. I haven't read it in detail yet, but it's an exciting direction.

### Hopfield networks are Deep Equilibrium Models, but nobody cares
Aha, a shameless self-plug! But important, nonetheless.

[My paper](https://proceedings.mlr.press/v255/goemaere24a) shows that Hopfield networks can and should be seen as [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) (DEQs). A rather straightforward result, even if I say so myself, but with major implications. You see, the DEQ community has figured out a lot about how these models should work: how to check if they're stable, what to measure, which algorithms to use.

But the Hopfield network community seems to work in isolation. Take the [Energy Transformer](https://arxiv.org/abs/2302.07253) for example: it's similar to the DEQ-Transformer from the original DEQ paper, but doesn't use any of the DEQ field's insights to improve performance.

It is absolutely crucial to track the necessary DEQ metrics. After all, it's too easy to train a lousy model, if you're not checking for training stability or model convergence. Unfortunately, most Hopfield networks papers don't report these metrics, making it difficult to validate their baselines and conclusions.

### Equilibrium Propagation is just a really good approximation of backprop, nothing more
Most Hopfield network researchers use standard backpropagation for training. But for analog hardware, there's this alternative called [Equilibrium Propagation](https://arxiv.org/abs/1602.05179) (EqProp), remember?

[Some](https://openreview.net/forum?id=jl5a3t78Uh) compare EqProp directly with backprop, as if they're completely different things. But here's the thing: EqProp is actually just a clever way to approximate backprop that works on analog hardware. That's not a criticism — it's actually great! It means we can get backprop-like training even on devices where traditional backprop wouldn't work.

But it's important to understand what EqProp really is and that it shares the same limitations as backprop. You still need labeled data, you still need to go through all layers and you still need two separate phases (though you can [parallellize them in space](https://arxiv.org/abs/2108.00275)).

It's unreasonable to expect EqProp to perform differently from backprop — at least, not in a positive way.

### Academic obscurantism: sounds complicated
One thing that really stands out in Hopfield network papers: they often look [incredibly](https://arxiv.org/abs/1606.01164) [complex](https://arxiv.org/pdf/2305.05179). The math is hard to follow, the notation is dense, and when the results aren't great, there's usually a long section about neuroscience interpretations.

The funny thing is, the model itself isn't that complicated — it's actually one of the simpler models in AI. So, if that's the case, then why all the hassle?

My suspicion: to confuse or impress reviewers and get papers accepted. Imagine the look on your reviewer's face when he reads it's just a database! ;)

## So… Predictive Coding it is then
Ugh, Hopfield networks are _so_ 2023!

Now, I'm focussing on Predictive Coding networks. The goal is still the same: creating energy-efficient AI models that can run on analog hardware.

Predictive Coding has some advantages that Hopfield networks don't have:

1. **Room for generalization:** Instead of storing exact patterns, these networks try to match features at different levels, allowing for some flexibility [(though they can do memory too)](https://openreview.net/forum?id=VuzPO_TZHPc).
3. **More biologically plausible:** The theory comes from [‪Karl Friston‬‬](https://scholar.google.co.uk/citations?user=q_4u0aoAAAAJ&hl=en), the most cited neuroscientist in the world.
4. **Connections to Reinforcement Learning:** That same theory is at the basis of [Active Inference](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0006421), an alternative to RL. It's nice to have a foot in the door to such a massive field.
5. **Proven performance:** Though there is still much to work on, [a recent paper](https://arxiv.org/abs/2407.01163) shows the model working well at larger scales.
6. **A training algorithm that's surely not backprop (I hope):** While I'm still getting my head around how training works exactly, it seems to be [quite different](https://www.nature.com/articles/s41593-023-01514-1) from backprop.

Of course, there are some challenges too:

1. **Academic obscurantism:** Oh no, not again! I'll try to change this.
2. **No analog implementations yet:** In theory, Predictive Coding networks can be easily mapped to analog hardware, but in practice? No one really knows.
3. **Small field:** The research community is quite small, with the same groups publishing most papers. Interestingly, one of its principle authors has written that he's [not so optimistic](https://www.beren.io/2023-03-30-Thoughts-on-future-of-PC/) about Predictive Coding. But he probably doesn't mean it... Right?

## Closing words
So there you have it, why I transitioned from Hopfield networks to Predictive Coding. I felt like the Hopfield network was being portrayed as something it is not, advertising capabilities it did not have, with a questionable biological plausibility and an unnecessary complexity. Predictive Coding, on the other hand, already has some proven results and aspects that definitely make it unique.

I'm curious to see where my research will lead me next. Perhaps, Predictive Coding is the same story wrapped differently; I have no way of knowing. You need to be deep into the field to be able to identify issues like these. And once you're deep in, it may be difficult to overcome your internal [sunk cost fallacy](https://thedecisionlab.com/biases/the-sunk-cost-fallacy). But remember: the time we invest in learning new concepts and developping a critical mindset, is always time well spent.

_Throughout this post, I may have been overly harsh to the Hopfield networks community and perhaps short-sighted. If you see things differently or have insights to add, I'd love to hear from you! The field is evolving quickly, and what seems like a limitation today might lead to a breakthrough tomorrow._

---

_Note to self: I should have written this blogpost in January 2024, when I decided to stop pursuing Hopfield networks. At that point, I was still an expert in the field, and writing this post would have been good therapy in dealing with the change of topic. A lesson to keep in mind for the future!_
