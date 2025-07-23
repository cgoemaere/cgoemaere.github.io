# Predictive Coding as backprop with local losses
Deep Learning has been massively successful in recent years, in part because of the efficiency of its learning algorithm: backpropagation. However, the brain is also pretty good at learning<sup>_(citation needed)_</sup>, and it's [definitely](https://pubmed.ncbi.nlm.nih.gov/30704969/) _not_ doing backprop. One theory for how the brain learns, is Predictive Coding (PC), which has recently been repurposed as a biologically-plausible alternative to backpropagation. 

Coming from theoretical neuroscience, PC is often described in overly complicated and confusing ways, with [free energies](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/) and [variational inference](https://arxiv.org/abs/2107.12979). While technically correct, it's not very helpful to gain intuition on how PC works, and it might scare off newcomers looking for entry into the field.

So, to maintain my sanity, I developed an interpretation of PC that better fit my Deep Learning background: ***Predictive Coding is just backpropagation with local losses.***

Let me explain.

## Predictive Coding: inserting local losses into the network
Consider the following feedforward architecture:
![Feedforward network architecture](https://raw.githubusercontent.com/cgoemaere/cgoemaere.github.io/refs/heads/master/assets/img/pc_local_losses/feedforward_network.png)

Typically, with backprop, we'd train this with a single global loss $\mathcal{L}_{class}$ at the output:
![Backpropagation with its global loss](https://raw.githubusercontent.com/cgoemaere/cgoemaere.github.io/refs/heads/master/assets/img/pc_local_losses/backprop.png)

However, in order to be biologically plausible, every neuron must have a learning signal available right at its doorstep, instead of a few layers more upstream. PC solves this by adding an intermediate loss $\mathcal{L}_i$ to every hidden layer:
![PC with intermediate local losses](https://raw.githubusercontent.com/cgoemaere/cgoemaere.github.io/refs/heads/master/assets/img/pc_local_losses/pc_local_losses.png)

Now, instead of perfectly forwarding a layer's prediction $\bm{\hat{s}_i}$, we allow some slack. We introduce a free variable $\bm{s_i}$, representing the input of the next layer. The local loss $\mathcal{L}_i$ tries to tie $\bm{\hat{s}_i}$ and $\bm{s_i}$ together as closely as possible, but they don't necessarily have to be equal to one another.

During training, PC tries to find the set of states $\{\bm{s_i}\}$ that minimizes _the sum_ of all of these local losses (typically referred to as the energy $E$):
$$\mathcal{L}_{PC} = \sum_i \mathcal{L}_i + \mathcal{L}_{class} $$
_(note that the classification loss has simply become the local loss for the top layer)_

With that set of optimal states (typically found via gradient descent), you can perform purely local weight updates which are 100% biologically plausible.

## Implications
Okay, so PC is just inserting local losses and finding the states that minimize the sum of the local losses. How is this interpretation more useful than variational inference?

Well, glad you asked. Let's clear up some confusion and list up some direct consequences and ideas:

- **Where's the squared difference I always see in PC?** To get standard PC, just use MSELoss for $\mathcal{L}_i$. If you want to add precision weights, go for Pytorch's [GaussianNLLLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html).
- **Can I really use any loss I'd like?** Hmmm, not quite. Many losses are secretly negative log likelihoods of some distribution (e.g., MSELoss belongs to a Gaussian), [and you can really use any distribution you'd like for PC.](https://arxiv.org/abs/2211.03481) But still, there are a few caveats:
	1.  The global loss should be bounded from below. Otherwise, gradient descent will go all the way down to negative infinity! Now, this doesn't necessarily mean each individual local loss should be bounded, but still, be careful.
	2. Keep in mind that your target is now $\bm{s_i}$, which is variable. This can be a problem for losses that expect the target to have a certain form (e.g., CrossEntropy expects a one-hot).
	3. You need to account for the distribution's normalization constant to get [correct weight regularization](https://arxiv.org/abs/1907.06845).
- **Do other losses lead to other models?** Yes! For example, the loss $\mathcal{L}_i  = \frac{1}{2} ||\bm{s_i}||^2 - \rho(\bm{s_i})^T \cdot \bm{\hat{s}_i}$ corresponds to the Continuous Hopfield network from [Bengio](https://arxiv.org/pdf/1510.02777).
- **Can I use a _trainable_ loss?** If you take into account the caveats above, yes you can! For example, you could use a 2D normalizing flow as a componentwise loss. However, the caveats are not always easy to enforce, so you'd have to engineer your way around them.
- **Why are local losses enough for global learning?** Turns out there's no need for backprop's global loss, just some time to spread the information across the local losses. Once that's done, every layer has its own loss to minimize, as if we had a bunch of parallel 1-layer networks trying to predict a fixed target. I find this pretty remarkable, to be honest!
- **Which one is greater: the PC energy or the backprop loss?** With standard initialization (which amounts to setting all $\mathcal{L}_i = 0$), the PC energy at the start will be exactly equal to the backprop loss. Next, we do energy minimization, so we can say that the energy will always be lower than the loss. Probably, the ratio $E/\mathcal{L}_{class}$ can be interpreted as something interesting, but I'm not sure what exactly.
- **How should I implement PC?** If you're not planning on doing anything too fancy, inserting local losses is enough! Most DL frameworks were designed to minimize losses, so it shouldn't be too much of a hassle.  The core of my [implementation](https://github.com/cgoemaere/pc_error_optimization) consists of only around 200 lines.

## Going beyond local losses: minimal-norm perturbations
The view of PC as local losses corresponds to the established form of PC doing state updates on free variables $\bm{s_i}$. However, in [my latest paper](https://arxiv.org/abs/2505.20137), I show that you can go _much_ faster (100-1000x) by directly modelling the errors $\bm{e_i}$ instead of the states.

Technical details aside, how can you interpret this in the local loss framework?

Instead of separating $\bm{\hat{s_i}}$ and $\bm{s_i}$, we simply add a skip connection and define $\bm{s_i} := \bm{\hat{s_i}} + \bm{e_i}$. The standard MSELoss now simplifies to an L2 norm on the errors $||\bm{e_i}||^2$. But feel free to choose any other norm!

Instead of local losses, we now have local perturbations that we add to the network predictions. 
![PC with minimal-norm perturbations](https://raw.githubusercontent.com/cgoemaere/cgoemaere.github.io/refs/heads/master/assets/img/pc_local_losses/pc_minimal_perturbations.png)

The goal is to minimize the classification loss while keeping the errors as small as possible (minimal norm):
$$\mathcal{L}_{PC} = \sum_i ||\bm{e_i}||^2 + \mathcal{L}_{class} $$
It's a bit similar to an adversarial attack, except that you're now trying to _improve_ the loss instead of degrade it.

You still prefer the local losses? That's alright. The errors can also be used to speed up the local loss formulation of PC. Just use the definition $\bm{s_i} := \bm{\hat{s_i}} + \bm{e_i}$ and plug it into your favorite loss function.

## Outro
No more need for fancy maths: Predictive Coding is just backprop with local losses. I hope this view can help you forward in getting some intuition on PC. Of course, once you get further into the field, it can be very useful to explore the mathematical structure of PC's graphical networks and its relation to ELBO. But for just a quick introduction? Local losses all the way.
