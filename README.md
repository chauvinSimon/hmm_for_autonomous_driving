# hmm_for_autonomous_driving

WORK IN PROGRESS!!

todo:
- python implementation of Baum-Welch
	- pytest
- squash commits
- generation based on the model

# Introduction
Disclaimer:

- The goal of this repository is to get more familiar with the concepts of **`Hidden Markov Models`** (**= HMM**).
- The scope of ambition is therefore limited and the examples are very simple, only serving educational purposes.

Addressed topics:
- Problem setting using a basic example to introduce the **HMM terminology**.
- Illustration of the **Bayes Rule** to determine the most likely state given an observation sequence.
- Implementation of the **Forward Algorithm** and the **Backward Algorithm** to compute the probability of a particular observation sequence.
- Implementation of the **Viterbi Decoding Algorithm** to find the most likely sequence of hidden states which could have generated a given observation sequence.
- Implementation of the **Baum-Welch Algorithm** to find the most likely parameters (state transition and emission models) of the HMM given an observation sequence.

[Bonus](#Bonus):
- **Literature review** of **HMM implementations** for **Autonomous Driving**.

# Problem motivation
For left-hand-drive countries such as the UK, just invert the reasoning :smiley:

- Your car is driving on a **2-lane highway**.
- Imagine that you can **remotely monitor the velocity of the car** (e.g. I communicate it to you).
- But you do have **no direct access to the lateral position** (`right lane` of `left lane`).
	- Formally, you **cannot access the underlying stochastic walk between `lane` states**.
- How could you **infer the `lane`** based on the single information you receive (the `speed`)?

### Emission probability

If I am telling you that I am driving with a `low speed`, you **may guess** that I am on the right lane.
- For instance, because I am just driving alone at a reasonable pace.
- Or because I am blocked by a slow vehicle while not able to take it over.
- But I could also drive fast on this `right lane`:
	- Have you ever been driving alone on a non-limited German highway?

Similarly, if you get informed of a `high speed`, you could say that I am **more likely** to be driving on the left lane.
- Probably overtaking another vehicle.
- Nevertheless, this is **not always true**:
	- Think of the situation where you are waiting on the left lane behind a truck trying to overtake another truck.

We get a **first intuition**:
- The variable `lane` seems to have an impact on the variable `speed`.
- In  other words: **you do not drive at the same pace depending if you are one the `left lane` or the `right lane`**.
- But the relation is **not deterministic**, rather **stochastic**.

This **causality finding** will be modelled using **`emission probabilities`** in the following.

### Transition probability

You could have a second intuition about the **sequential process**:
- Human drivers usually **stay on their lanes**.
- Hence if you are on `right lane` at time `t`, you are likely to still be on `right lane` at time `t+1`.
- Again, this **does not always hold** and you can find **exception**.
- But here comes a second intuition: **the `lane` at time `t` is influenced by the `lane` at time `t-1`**.

The concept of **`transition probability`** will be used to model this second remark.

In the next sections we will see how can we **mathematically support our intuition**.

## Terminology

| ![The speed is the `observation` while the lane constitutes the `hidden state`. Some examples show that all `emissions` are possible](docs/terminology.PNG "The speed is the `observation` while the lane constitutes the `hidden state`. Some examples show that all `emissions` are possible")  | 
|:--:| 
| *The speed is the `observation` while the lane constitutes the `hidden state`. Some examples show that all `emissions` are possible* |


## Objectives

We can now define three problems which can be solved by an HMM:

- 1- **Learning**
	- The first is **learning the parameters** of its **the latent structure** (emission, transition and initial state models).
	- In the case of a known structure and some fully observable sampling, on can apply the concept of **Maximum Likelihood Estimation** (MLE):
		- Some observation sequences (`speed`) and their associated states (`lane`) have been collected. The **samples** form the **training dataset**.
		- The parameter can be selected so as to maximise the likelihood for the model to have produced the data from the given dataset.
	- In case it is not possible to sample from hidden states (partially observable), one can use **Expectation Maximization** (EM) or **Markov Chain Monte Carlo** (MCMC)
	- Learning is covered in questions [Q1](#q1) and [Q5](#q5)

- 2- **Evaluation**
	- The second task is to find, given a HMM, how likely is it to get a observation sequence.
	- It is covered in question [Q3](#q3)

- 3- **Inference**
	- The third one is applying the trained HMM to an observation sequence. We want to find the sequence of states **that best explains** the observation sequence.
	- Concretly we want to **infer the sequence of lanes** driven by the car ((`right` or `left`) = **hidden state**) based on a **sequence of speed measurements** (= **observation**).
	- In this case, you can thing of HMM as a **sequence classifier** (if we make the hidden state of HMM fixed, it would have been a **Naive Bayes classifier**).
	- Inference is covered in questions [Q2](#q2) and [Q4](#q4)
	- Three types of inference can be distinguished:
		- **Filtering**: determine the latest belief state, i.e. the posterior distribution P(`lane(t)` given [`speed(1)`, ..., `speed(t)`])
		- **Decoding**: determine the hidden state sequence, that gives the best explanation for the observation sequence
			- find [`lane(1)`, ..., `lane(t)`] that maximizes P([`lane(1)`, ..., `lane(t)`] given [`speed(1)`, ..., `speed(t)`])
			- it is **equivalent to maximizing the Joint Probability** P([`lane(1)`, ..., `lane(t)`] given [`speed(1)`, ..., `speed(t)`]) since the marginal over observation is independant of the variables we are maximizing over.
		- **Prediction**: determine the probability for the future hidden state in K steps, i.e. the posterior conditional distribution P(`lane(t+K)` given [`speed(1)`, ..., `speed(t)`] )
		- **Smooting**: determine the probability for the past hidden state K steps ago, i.e. the posterior conditional distribution P(`lane(t-K)` given [`speed(1)`, ..., `speed(t)`] )
		
## Assumptions
To keep the problem as simple as possible:
- Let's **discretize the speed** into `low speed` and `high speed`.
- Time steps are discretized.
- Lane transitions are ignored: either you are on `left lane` or you are on `right lane`.

### Stationary Process

- We assume that the HMM models (transition and emission matrices and initial distribution) are **constant over time**
- `P[speed(t)` given `lane(t)]`, `P[lane(t+1)` given `lane(t)]` and `P[lane(t)]` are independant of `t`.

### Observation Independence

- We talked about emission probability, explaining that state `lane (t)` impacts observation `speed (t)`.
- One could other sources of influence: `speed (t-1)` and `lane (t-1)` for instance. 
- Here we assume that the **probability of an observation depends only on the state that produced the observation** and not on any other states or any other observations
- Each observation variable `speed` depends only on the current state `lane`.
- This is a **strong assumption** since we decide not to capture dependencies between each input element in the observation sequence
- But it will **relax computation** during decoding.
- In supervised learning terminology: **"each feature (observation) is conditional independent of every other feature, given the class (hidden state)"**
- As for Naive Bayes, probabilities are independent given the class `y` and hence can be **"naively"** multiplied: p(`x1,x2,…,x1|y`)`=`p(`x1|y`) `*` p(`x2|y`) `*` ... `*` p(`xm|y`)

### First-order Markov Property

- We just said that it is useful to know the present `lane` (at time `t`) to infer the future `lane` (at time `t+1`).
- What about the previous `lane` at `t-1`? It probably also hold relevant information?
- Here is a strong assumption about inferring in this stochastic process:
	- the conditional probability distribution of **future states** of the process (conditional on both past and present states) **depends only upon the present state**, not on the sequence of events that preceded it.
- In other words, **"the future is independant of the past given the present"**.
- This **strong assumption** is known as the first-order **Markov Property** (also named **"memoryless property"**) and will make computations easier in the following.

Based on these assumptions, the problem can be modelled using a **graphical model**:
- The **Markov Property** makes sure state connection occurs only for consecutive states.
- The **Output Independence** is responsible for each observation to receive only a single edge (coming from the associated state).
- HMM are **directed models** (hence arrows) since we can distinguish what is the reason (`lane` state) and what is the result (`speed` observation).
- HMM is a special case of **Dynamic Bayesian Network** (Dynamic belief network) since it forms a **probabilistic directed acyclic graphical model**.

| ![HMM Graphical Representation](docs/hmm_graphical_model.PNG "HMM Graphical Representation")  | 
|:--:| 
| *HMM Graphical Representation* |

## Relations with other machine learning techniques

For better understanding, I find convenient to compare HMMs with the other algorithms and methods I know.

- HMM is a special case of **Finite State Machine** (FSM).
- **Kalman Filters** can be conceived of as continuous valued HMMs:
	- HMM uses *discrete* state (**Markov chain**). KF uses *continuous* state (**Markov process**).
	- HMM uses *arbitrary transition*, KF uses *(Linear-)Gaussian transitions*.
	- HMM uses *observation matrices*. KF uses *Gaussian observations*.
- HMM is a special case of **Dynamic Bayes Networks** (DBNs) in which the entire state of the world is represented by a single hidden state variable.
- HMM and **Partially Observable Markov Decision Processes** (POMDPs) have many similarities in their formulations. But in POMDPs, the user has some control (not just a spectator) over the process transitions between hidden states (introduction of the concept of `action`).
- Contrary to HMM, in **Naive Bayes model**, hidden state are fixed (no state sequence since there is no transition happening).
- States are hidden in HMM. They are fully observable in **Markov Chains**.
- HMM is a special case of **Generative Directed Graphs** (GDG), where the graphical structure is a first-order-chain structure (one could impose dependencies to arbitrary elements, not just on the previous element).
- **Linear-Chain CRF** can be seed as the discriminative version of HMM (like Logistic Regression and more generally  Maximum Entropy Models are the discriminate version of Naive Bayes) i.e. the consideration is on the conditional probability p(y|x) instead of the joint probability p(y,x).
	- It is a discriminative undirected probabilistic graphical model.
	- It is used to encode known relationships between observations and construct consistent interpretations. 
	
- In **Conditional Random Fields** (CRF), the two strong (unreasonable) HMM hypotheses are dropped (it better addresses the so-called "labeling bias issue" but also become more complicated for inference).
- **Maximum Entropy Markov Models** combine features of HMMs (Markov chain connections) and maximum entropy (MaxEnt) models: it is a discriminative (not generative) model that allows the user to specify lots of correlated, but informative features.
	- MEMMs focus on p（`state|observation`), while HMMs focus on p（`observation|state`）
	- btw, CRF can be seen as a more advanced MEMM with global variance normalization and with undirected connections, to address the issue of "label bias problem"
	- HMM models "state decides observation" and this is why it is called "generative". MEMMs model "observation decides state".

| ![HMMs share concepts with other methods and algorithms](docs/hmm_neighbours.PNG "HMMs share concepts with other methods and algorithms")  | 
|:--:| 
| *HMMs share concepts with other methods and algorithms* |

# Problem formulation

## Definitions:

A Hidden Markov Model (HMM) is a **5-tuple** composed of:

- A set of **Hidden States**: discrete random variable `lane` in {`Right Lane`, `Left Lane`}
- A set of possible **Observations**: discrete random variable `speed` in {`Low Speed`, `High Speed`}
- A stochastic matrix which gives **Emission probabilities**: `P[speed(t)` given `lane(t)]`
- A stochastic matrix which gives **Transition probabilities**: `P[lane(t+1)` given `lane(t)]`
- A **Initial State Probability** distribution: `P[lane(t=t0)]`

## Questions:
- [Q1](#q1) - How to **derive our probability models**?
- [Q2](#q2) - If you receive a **single `speed` observation**, what is the probability for the car to be in each lane?
- [Q3](#q3) - What is the probability to observe a particular **sequence of `speed` measurements**? For instance [`low speed`, `high speed`, `low speed`].
- [Q32](#q32) - Given a **sequence of `speed` measurements**, what is the most likely **latest `lane`**?
- [Q4](#q4) - What is the **most likely `lane` sequence** if the **observation sequence** is [`low speed`, `high speed`, `low speed`]?
- [Q5](#q5) - How to estimate the HMM parameters with **no state annotations** in the training data?
- [Q6](#q6) - After a long time, what is the probability for the car to be in each lane?

# Answers

## Q1 - How to derive our probability models?

Given some observation `speed` sequence and the associated `lane` states
- Here are only a few for simplicity but you could imagine longer recordings.
- You can think of it as some **samples** of the underlying **joint distributions**.
- How can we estimate the HMM parameters (for the emission, transition and intial state models)?
- We can count how many times each event occurs in the data and normalize the counts to form proper probability distributions.
- In can be formalize with **Maximum Likelihood Estimation** in the context of **Supervised Learning**, i.e. finding parameters `θ` that maximize `P(x|θ)`.

The idea is to **approximate the model parameters by counting the occurrences**, similar to Naive Baye method where training is mainly done by counting features and classes.

Note: 
- If a rare transition or emission is **not encountered** in the training dataset, its **probability** would be **set to `0`**.
- This may have be caused by over-fitting or a small sample size.
- To avoid `0` probabilities, an alternative is to use **pseudocounts** instead of **absolute counts**.

### Transition probability

**Counting the number** of transitions, we can derive a **transition probability** model.
- For instance, among the `15` transitions starting from `right lane`, `3` ended in `left lane`.
	- Hence P[`right lane` -> `left lane`] = `0.2`
- Since probabilities must sum to one (normalization), or just by counting for the other case (`right lane` is followed `12` times by a `right lane`)
	- Hence P[`right lane` -> `right lane`] = `0.8`

| ![Derivation of the transition probability model](docs/deriving_transition_model.PNG "Derivation of the transition probability model")  | 
|:--:| 
| *Derivation of the transition probability model* |

### Emission probability

Counting can also be used to determine the **emission probability** model.
- Similar to Naive Bayes in Supervised Learning where we count occurrences of features for all classes in the training data.
- For instance, how many times has the hidden state `left lane` caused a `high speed` observation?

| ![Derivation of the emission probability model](docs/deriving_emission_model.PNG "Derivation of the emission probability model")  | 
|:--:| 
| *Derivation of the emission probability model* |

## Initial state probability

This can be seen as a special case of transition probability as the first state, by virtue of being first, doesn't have any state preceding it and hence transition probabilities cannot be applied

At any time `t`, what is your guess on the **distribution of the hidden state if no observation is given**?
- Two options are available:
	- Either you use the transition model (1) and the fact that probabilities sum to `1` (2):
		- (1) P[`left lane`, `t`] = P[`left lane` -> `left lane`] `*` P[`left lane`, `t-1`] + P[`right lane` -> `left lane`] `*` P[`right lane`, `t-1`]
		- (2) P[`right lane`, `t`] = `1` - P[`left lane`, `t`]
	- Either you simply **count occurrences** in the supplied data. We simple count how many samples in the training data fall into each class (here state instance) divided by the total number of samples:
		- P[`left lane`, `t`] = `1/3`
		- P[`right lane`, `t`] = `2/3`

## Note: incomplete or missing state/observation

How to cope with **absence of one state or one observation instance in the sampled data**?

- Obviously we would need to collect much more pairs to refine the parameter estimation of our models.
- Clearly, assigning `0` may cause issues in the inference process.
- Every state could have a small emission probability of producing an unseen observation
	- P(`obs` | `state`) = `espilon_for_unseen_case` if `obs` has not been seen.
- Look for **"Laplace Smoothing"** or **"Additive Smoothing"** if you are interested.

### Summary

| ![Hidden Markov Model with the `initial state probability` model (up), `transition probability` model (middle), and the `emission probability` model (below)](docs/hmm.PNG "Hidden Markov Model with the `initial state probability` model (up), `transition probability` model (middle), and the `emission probability` model (below)")  | 
|:--:| 
| *Hidden Markov Model with the `initial state probability` model (up), `transition probability` model (middle), and the `emission probability` model (below)* |

## Q2 - If you receive a **single `speed` observation**, what is the probability for the car to be in each lane?

### Prior

Before any observation we know that `right lane` appears `2/3` of the time and `left lane` `1/3`.
- These probabilities would have been the answers if we were to **ignore the observation**.
- This distribution (called `prior`) must be **updated** when considering the observation.
- When considering the observation, the `prior` is converted to a `posterior` using `likelihood` terms.

### Likelihood

Based on supplied data, we found that on the left lane it is more likely to drive fast. And slow on the right lane.
In other words, you will be rather "suprised" if a left lane causes.
I like the concept of **"Surprise"** to introduce the concept of **"Likelihood"**
- The likelihood is, in this case, equivalent to the `emission probability`:
	- **given a state**, what is the probability for each observation.
- Actually, the question is interested by the exact opposite:
	- **given an observation**, what is the probability for each state?
	- this is called the `posterior` and is linked to `prior` via the `likelihood`.

### Marginal
The Bayes Rule states that `Posterior` = `Normalized (prior * likelihood)`
- The normalization (the posterior terms must sum to `1`) is achieved using the `Marginal Probabilities`.
- Under all possible hypotheses, how probable is each `speed`?
- Using the **law of total probability**:
	- P(`high speed`) = P(`high speed` given `left lane`) `*` P(`left lane`) + P(`high speed` given `right lane`) `*` P(`right lane`)
	- P(`low speed`) = P(`low speed` given `left lane`) `*` P(`left lane`) + P(`low speed` given `right lane`) `*` P(`right lane`)
- Eventually:
	- P(`high speed`) = `1/3`
	- P(`low speed`) = `2/3`

### Bayes Rule
- Let's use Bayesian Statistics to recap:
	- P(`lane` given `speed`) = P(`lane`) `*` P(`speed` given `lane`) / P(`speed`)
	- `Posterior` = `Prior` `*` `Likelihood` / `Marginal`
- For instance
	- P(`left lane` given `high speed`) = P(`left lane`) `*` P(`high speed` given `left lane`) / P(`high speed`)

Priors:
- P(`right lane`) = `2/3`
- P(`left lane`) = `1/3`

Marginals:
- P(`high speed`) = `1/3`
- P(`low speed`) = `2/3`

Likelihoods:
- P(`high speed` given `left lane`) = `0.6`
- P(`low speed` given `left lane`) = `0.4`
- P(`high speed` given `right lane`) = `0.2`
- P(`low speed` given `right lane`) = `0.8`

Posteriors:
- P(`left lane` given `high speed`) = `1/3` `*` `0.6` / `1/3` = `0.6`
- P(`left lane` given `low speed`) = `1/3` `*` `0.4` / `2/3` = `0.2`
- P(`right lane` given `high speed`) = `2/3` `*` `0.2` / `1/3` = `0.4`
- P(`right lane` given `low speed`) = `2/3` `*` `0.8` / `2/3` = `0.8`

### Summary

| ![Derivation of the posterior probabilities for a single observation](docs/posteriors.PNG "Derivation of the posterior probabilities for a single observation")  | 
|:--:| 
| *Derivation of the posterior probabilities for a single observation* |

The question was *given an observation, what is the most likely hidden state?*.
Well, just looking at the numbers on the figure below and taking the `max()`, the answer is:

- Given `high speed` is observed, the most likely state is `left lane`
- Given `low speed` is observed, the most likely state is `right lane`

It is close to our intuition.

## Q3 - What is the probability to observe a particular **sequence of `speed` measurements**?

Had it been a first-order Markov Chain (no hidden state), one could have marginalized over all `speed(t)` observations and simplify the large expression using the **conditional independance** offered by the Markov Property.

The final expression would be:
- P([`speed(1)`, ..., `speed(t)`]) = P[`speed(1)`] * SUM ( P([`speed(t)` given `speed(t-1)`]) )

In the HMM case, we need to add some modifications:
- We use the **transition model** to navigate between two states.
- We use the **emission model** to generate each observation.

| ![Derivation of the marginal probability of an observation sequence](docs/marginal_proba_obs_sequence.PNG "Derivation of the probability of an observation sequence")  | 
|:--:| 
| *Derivation of the probability of an observation sequence* |

This requires to develop all the possible state sequences of size T.
- In a **brute force search**, it will **look at all paths**, try all possibilities, and calculate their joint probability.
- In our case state space has size `2` (`left_lane`, `right_lane`). Hence the **sum will contain `2^T` terms**.
- This **naive** (meaning we list all the possibilities based on the **definition of marginal probabilities**) approach can nevertheless **become impractible** and will not scale for larger state spaces and/or large sequence sizes.

An alternative is to use **Dynamic Programming**.
- the idea is to compute cells in a table `alpha`(`i`, `t`)
	- think of rows (index `i`) as state instances (`left_lane`, `right_lane`)
	- think of columns (index `t`) as time step (`t`=`1`, `t`=`2`, ... , `t`=`T`)
- three notable relations:
	- 1st relation: **`alpha`(`i`, `t+1`) can be computed from `alpha`(`i`, `t`)**.
		- Hence we can the table can be filled efficiently (only `2T` terms in the table)
		- Idea of the derivation:
			- marginalize `alpha`(`i`, `t+1`) over `state(t)`
			- split `observation[1 ... t+1]` = `observation[1 ... t]` + `observation(t+1)`
			- decompose the joint probability into a product of conditional probabilities (isolate `observation(t+1)` and `state(t+1)`)
			- use **conditional independance** to simplify two terms ([given `state(t)`, `state(t+1)` is independant of all possible `observation`] and [given `state(t+1)`, `observation(t+1)` is independant of all other terms])
			- three terms are left:  [emission at `t+1`], [transition from `t` to `t+1`] and [`alpha`(`i`, `t`)]
			- `alpha`(`i`, `t+1`) = [emission at `t+1`] * SUM[over `state(t)`] [transition`t`to`t+1` * `alpha`(`i`, `t`)]
	- 2nd relation: the table initialisation [`alpha`(`i`, `t=1`)] is easily computed with the initial state distribution and the emission model.
	- 3rd relation: the marginal probability (what we are really looking for) can be expressed with the `alpha`(`i`, `T`) (just by **summing the terms of the last column**).
	- Therefore it is **computationally-efficient** to fill the `alpha` table to **derive the marginal distribution of observation sequence**.

| ![Derivation of construction rules for the `alpha table`](docs/alpha_derivation.PNG "Derivation of construction rules for the `alpha table`")  | 
|:--:| 
| *Derivation of construction rules for the `alpha table`* |

Each element in the column `t+1` is a weighted sum of the elements at `t`:
- The weights are the transition probabilities
- The obtained sum is finally scaled by the emission probability (for the state `i` to emit the observation present at `t+1` in our observation sequence)

The **`alpha` values** are the **joint probabilities** of
- observing the **first `t` observations**
- and being in `state k` at `time t`. 

**Path**
The partial probability  𝞪t(k) is the probability of reaching state ‘k’ at a stage ‘t’, going through all possible paths over the ‘t’ stages
If the model is in state Sj in at time t − 1 and is to move to the state Si at time t for the observation Ot , then it first has to transition from state Sj to Si , given by the transition probability T(Si|Sj), and then needs to emit the observation Ot , given by the emission probability E(Ot|Si).  
Sum for P(O). The above and trivially true because 𝞪N(x)  gives us the probability of all paths ending in state ‘x’ after ‘N’ stages. Hence any final state can occur, we sum 𝞪N(x) over all possible states ‘x’.

| ![Construction of the `alpha table` using Dynamic Programming](docs/alpha_table.gif "Construction of the `alpha table` using Dynamic Programming")  | 
|:--:| 
| *Construction of the `alpha table` using Dynamic Programming* |

Because there are **`S\*T` entries** and each entry examines a total of `S` other entries, this leads to
- **`O(S\*S\*T)` time complexity**,
- and **O(S\*T) space complexity**.
- where `S` denotes the number of hidden states and `T` the length of the observation sequence.

Once the `alpha table` is constructed, it is straight forward to get the the **marginal probability** for the associated **observation sequence**:
- summing the `alpha` values at time `t` gives the probabily of the observation sequence up to time `t`
- for instance, among the `8` possible 3-observable sequences, [`low speed`, `high speed`, `low speed`] has a probability = `0.13184`

| ![Use of the `alpha table` for the **marginal probability** of an **observation sequence**](docs/alpha_table_marginal.PNG "Use of the `alpha table` for the **marginal probability** of an **observation sequence**")  | 
|:--:| 
| *Use of the `alpha table` for the **marginal probability** of an **observation sequence*** |

#### Note

The **sum of the elements in the last column** of the dynamic programming table provides the **total probability of an observed sequence**.
- In practice, given a sufficiently **long sequence of observations**, the forward probabilities decrease very rapidly.
- To circumvent issues associated with **storing small floating point numbers**, **logs-probabilities** are used in the calculations instead of the probabilities themselves.

## Q32 - Given a **sequence of `speed` measurements**, what is the most likely **latest `lane`**?

Filtering is one important application for robotics and **autonomous driving**:
- We often want to **estimate the current state** of some objects (e.g. _position_, _speed_, or a belief over the _route intention_).
- In order to **reduce the variance of the estimate**, you may want not to base your estimation only on the latest measurement.
- Hence, similar to what is done with Bayesian Filters (BFs) (such as **Kalman Filters**, **Particles Filters**), two ingredients are used to update your latest **State Belief**:
	- a **sequence of measurements** (= **observation**)
	- some evolution **models**
		- for instance odometry measurements and `constant acceleration` or `constant velocity` models in the context of **sensor fusion** for localization
		- here we have an emission model and a transition model)

The form of the `alpha table` turns out to be very appropriate for **filtering**:
- Let's focus on P(`lane(t=3)` == `right` given [`low speed`, `high speed`, `low speed`])
- Express condition probability with the joint probability.
- Note that we find the **marginal probability of the observation sequence** at the denominator.
- Marginalize it over last the hidden state `lane(t=3)`.
- All terms left are `alpha` values.

| ![Use of the `alpha table` for **filtering**](docs/alpha_table_filtering.PNG "Use of the `alpha table` for **filtering**")  | 
|:--:| 
| *Use of the `alpha table` for filtering* |


### Note: Markov Property

> Why do you consider **all the three observations**? Does not the **Markov Property** state that you **only need the latest one**?

Have a look at the graphical representation of the HMM.
- The Markov Property only applies for the transitions between hidden states.
- The rule of **D-separation** (commonly used in **Bayesian Networks**) states that the knowledge of the realisation of one previous hidden could have blocked the path from some observations.
- That would have made our result **conditionally independant** of these observations.
- For instance, in P(`lane(t=3)` == `right` given [`low speed`, `high speed`, `low speed`] and `lane(t=1)`==`right`), the first observation (at `t=1`) is useless since all its paths to `lane(t=3)` (there is only one path here) are blocked by `lane(t=1)` (which realisation is known).
- Since we only consider realisations of observation, no simplification can be done.

Actually, when filtering over the last observation only, we get a different result:
- P(`lane(t=3)` == `right` given [`low speed`, `high speed`, `low speed`]) = 0.73786
- P(`lane(t=1)` == `right` given [`low speed`]) = 0.8 (also readable in the `alpha table`: `8/10` `/` (`8/10` + `2/10`) )

### From `alpha table` to `beta table`

The `alpha table` can be used:
- To determine the belief state (**filtering**).
- To compute **marginal probability** of an **observation sequence**.

Note that the `alpha table` was completed starting **from left and moving to right**
- This is the reason why this inference method is called **"Forward Algorithm"**
- One could have the idea of going the other way round.
- This would lead to the **`beta table`**

| ![Derivation of construction rules for the `beta table`](docs/beta_derivation.PNG "Derivation of construction rules for the `beta table`")  | 
|:--:| 
| *Derivation of construction rules for the `beta table`* |


| ![Construction of the `beta table` using Dynamic Programming](docs/beta_table.gif "Construction of the `beta table` using Dynamic Programming")  | 
|:--:| 
| *Construction of the `beta table` using Dynamic Programming* |

The `beta table` can actually be used to compute the **marginal probability of an observation sequence**:
- Let's focus on P([`low speed`, `high speed`, `low speed`])
- Marginalize over the first hidden state `lane(t=1)` (insert it in the joint distribution and sum over all its possible realisations)
- Write the decomposition specific to the HMM structure (an **first-order Markov Chain**).
- The term P(`speed[2 ... t]` given `lane(1)`==`j`) is by definition `beta`(`j`, `t=1`)
- In other words, the **marginal probability of an observation sequence** can be obtained from the terms in the **first columns** of the associated `beta table`.

| ![Use of the `beta table` for the **marginal probability** of an **observation sequence**](docs/beta_table_marginal.PNG "Use of the `beta table` for the **marginal probability** of an **observation sequence**")  | 
|:--:| 
| *Use of the `beta table` for the **marginal probability** of an **observation sequence*** |


### Smoothing

**Smooting** is made easy when `alpha table` and `beta table` have been computed for an observation sequence.
- Let's consider the observation sequence [`low speed` (`t=1`), `high speed` (`t=2`), `low speed` (`t=3`)]
- What is the **probability distribution for the `lane` at `t=2`?**
- Note that we can answer the question for `t=3`: cf. **filtering** of [Q32](#q32).

Ideas for the derivation:
	- the posterior probability is turned to a **joint probability** over all possible hidden states, introducing a `normalization constant`
	- P(`lane(t=2)` == `left` given [`low speed` (`t=1`), `high speed` (`t=2`), `low speed` (`t=3`)])
	- the observation sequence is **split at `t=2`**
	- = P(`lane(t=2)` == `left` and [`low speed` (`t=1`)] and [`high speed` (`t=2`), `low speed` (`t=3`)]) `/` `Normalization_constant`
	- = P(`lane(t=2)` == `left` and [`low speed` (`t=1`)]) `*` P(`lane(t=2)` == `left` given [`high speed` (`t=2`), `low speed` (`t=3`)]) `/` `Normalization_constant`
	- Note that [`low speed` (`t=1`)] does not appear in the second term (conditional probability), since given the realisation of `lane(t=2)`, all its paths to [`high speed` (`t=2`), `low speed` (`t=3`)] are "blocked" (**conditional independance**).
	- = **`alpha`** (`left`, `t=2`) `*` **`beta`**(`left`, `t=2`) `/` **`Normalization_constant`**
- Since probabilities must sum to one,
	- `Normalization_constant` = `alpha`(`left`, `t=2`) `*` `beta`(`left`, `t=2`) `+` `alpha`(`right`, `t=2`) `*` `beta`(`right`, `t=2`)
	- `Normalization_constant` = `14/125` `*` `0.56` `+` `12/125` `*` `0.72` = `0.13184`
	- This result should be familiar to you ;-)
- For each time `t`, [SUM over `lane i` of (`alpha`(`i`, `t`) `*` `beta`(`i`, `t`))] represents the probability of observing [`low speed`, `high speed`, `low speed`] among the among the `8` possible 3-observable sequences
	- For `t=1`: (`2/15`) `*` `0.2592` `+` (`8/15`) `*` `0.1824` = `0.13184`
	- For `t=2`: (`14/125`) `*` `0.56` `+` (`12/125`) `*` `0.72` = `0.13184`
	- For `t=3`: (`108/3125`) `*` `1` `+` (`304/3125`) `*` `1` = `0.13184`
- Answers:
	- P(`lane(t=2)` == `left` given [`low speed` (`t=1`), `high speed` (`t=2`), `low speed` (`t=3`)]) = (`14/125`) `*` `0.56` `/` `0.13184` = `0.475728`
	- P(`lane(t=2)` == `right` given [`low speed` (`t=1`), `high speed` (`t=2`), `low speed` (`t=3`)]) = (`12/125`) `*` `0.72` `/` `0.13184` = `0.524272`
	- ??? Very close result between the two instances of `lane`.

### Prediction

Given an observation sequence, distributions over **future hidden states can be inferred** using a variant of the `alpha table`.
- Let's consider the observation sequence [`low speed` (`t=1`), `high speed` (`t=2`), `low speed` (`t=3`)]
- What is the **probability distribution for the `lane` at `t=5`**, i.e. in **two time steps in the future**?
- Note that we can answer the question for `t=3`: cf. **filtering** of [Q32](#q32).
- Using Dynamic Programming, we define a new quantity `pi` such as
	- `pi`(`lane i`, `time t+k`) = P(`lane` **`(t+k)`** = `i` given [`speed(1)` ... `speed(t)`])

A recursion rule can be derived:
- `pi`(`lane i`, `time t+k+1`) = SUM over `state` `j` of [P(`lane(t+k+1)=i` given `lane(t+k)=j`) `*` `pi`(`lane i`, `time k+1`)]
	- For the derivation, insert `lane(t+k)=j` in the definition of `pi`(`lane i`, `time t+k+1`).
	- Since its realisation is not known, we marginalize over `lane(t+k)=j`.
	- Break the joint part [`lane(t+k+1)=i` and `lane(t+k)=j`] in a conditional.
	- The expression can be simplified since `lane(t+k+1)=i` is conditionally independant of the observation sequence (`lane(t+k)=j` is blocking )

The initialisation has `k=0`, i.e. it is a filtering problem (inference for the current time):
- similar to the computation in [Q32](#q32),
- `pi`(`i`,`0`) = `pi`(`lane i, time t+0`) = P(`lane(t) = i` given [`speed(1)` ... `speed(t)`]) = `alpha(i, t)` / [SUM over j of `alpha(j, t)`]

In other words,
- Each element of the last columm of the `alpha table` is used to initialize the first column in the `pi table`.
- Then each element in the `pi table` is a **weighted sum of the elements in the previous column**.
- Weights are the **transition probabilities**.

| ![Derivation of construction rules for the `pi table` using Dynamic Programming](docs/pi_table_derivation.PNG "Derivation of construction rules for the `pi table` using Dynamic Programming")  | 
|:--:| 
| *Derivation of construction rules for the `pi table` using Dynamic Programming* |

| ![Construction of the `pi table` using Dynamic Programming](docs/pi_table.PNG "Construction of the `pi table` using Dynamic Programming")  | 
|:--:| 
| *Construction of the `pi table` using Dynamic Programming* |

| ![Use of the `pi table` for **prediction**](docs/pi_inference.PNG "Use of the `pi table` for **prediction**")  | 
|:--:| 
| *Use of the `pi table` for **prediction*** |

Answer:
- P(`lane` = `right` at `t=5` given [`low speed` (`t=1`), `high speed` (`t=2`), `low speed` (`t=3`)]) = `pi`(`right lane`, `k=2`) = `0.688`
- P(`lane` = `left` at `t=5` given [`low speed` (`t=1`), `high speed` (`t=2`), `low speed` (`t=3`)]) = `pi`(`left lane`, `k=2`) = `0.312`

Is there a convergence of the `pi` values as `k` grows?
- Intuitively, it is asking the question: _what will be the hidden state in an infinite number of steps?_
- It converges to the **initial state distribution**
- It **forgets about the observation** and **conditional probability becomes a prior probability**

[`pi_table.ipynb`](pi_table.ipynb) computes the `pi(k)` values associated to this observation sequence for any time step `t+k`.

```python
import numpy as np
# 0 = left
# 1 = right
pi_zero = np.array([0.8, 0.2])  # deduced from the `alpha_table`
transition = np.array([[0.8, 0.4], [0.2, 0.6]])

def pi(k):
    if k == 0:
        return pi_zero
    return np.dot(transition, pi(k-1))
```

| `k`   | `pi`(`right lane`, `k`) | `pi`(`left lane`, `k`) |
| :---: | :---:                   |     :---:              |
| `0`   | `0.8`                   | `0.2`                  |
| `1`   | `0.72`                  | `0.28`                 |
| `2`   | `0.688`                 | `0.312`                |
| `3`   | `0.6752`                | `0.3248`               |
| `4`   | `0.67008`               | `0.32992`              |
| `5`   | `668032`                | `0.331968`             |
| `10`  | `66668065`              | `0.33331935`           |
| `inf` | `2/3`                   | `1/3`                  |

| ![Change in the **state distribution** as the **prediction horizon** increases](docs/pi_of_k.svg "Change in the **state distribution** as the **prediction horizon** increases")  | 
|:--:| 
| *Change in the **state distribution** as the **prediction horizon** increases* |

Remark:
- The `pi table` can also be used to **make predictions about the observation**.


## Q4 - What is the **most likely `lane` sequence** if the **observation sequence** is [`low speed`, `high speed`, `low speed`]?

### Question interpretation

When trying to **reformulate the question**, I was puzzled since I ended up with two possible answers.

It all depends on what we mean with _"What is the **most likely state sequence** given an observation sequence?"_

- it can be the state sequence that has the **highest conditional probability?** This is was we have done in [Q2](#q2) with `#BayesRule`. `#PosteriorDecoding`.
- it can be the state sequence that **makes the observation sequence the most likely to happen?** `#MLE`. `#Viterbi`. In this case, we compute the **Joint Probabilities** i.e. the probabilities for the **Intersection** [`State Sequence` `+` `Observation Sequence`].
- ??(can the MLE be on the conditional probability of observation given state? what differs with the upper joint event is the normalization)??
- ?? `argmax(A)`(`P`[`A, B`]) = `argmax(A)`(`P`[`A| B`])

In other words, the answer could be maximizing two kind of probability

- either the **conditional probability**: (state **given** observation)
- or the **joint probability**: (state **and** observation).

> For this question, we will be looking for the **most likely sequence of hidden states** which could have generated a **given observation sequence**.

#### Maximum Likelihood Estimation (MLE)
We will to **pick the sequence (in [Q2](#q2) it was of size `1`) of hidden states that makes the observations the most likely to happen**.

This method is called **Maximum Likelihood Estimation** (MLE).

| ![Derivation of the probability of the event [`RL-LL-RL`; `LS-HS-LS`]](docs/compute_three.PNG "Derivation of the probability of the event [`RL-LL-RL`; `LS-HS-LS`]")  | 
|:--:| 
| *Derivation of the probability of the event [`RL-LL-RL`; `LS-HS-LS`]* |

Here are the different steps performed for the observation sequence [`low speed`, `high speed`, `low speed`].
- First enumerate the `2^3 = 8` possible sequences of hidden states.
- For each candidate, **compute the probability for the state sequence candidate to generate the observation sequence**.
	- Start by the probability of the first state element to happen (**initial state probability**).
	- List the **emission** and **transition probabilities**.
	- The probability of the observation sequence (?likelihood) is the product of all listed probabilities (thank @MarkovProperty).
- Apply `max()` to get the **Maximum Likelihood Estimate**:
	- In this case, the state sequence [`right lane`, `right lane`, `right lane`] makes the observation sequence the most likely to happen.

| ![Our three assumptions (`First-order Markov property`, `Observation Independance` and `Stationary Process`) simplify the computation of the joint distribution of sequences](docs/joint_proba.PNG "Our three assumptions (`First-order Markov property`, `Observation Independance` and `Stationary Process`) simplify the computation of the joint distribution of sequences")  | 
|:--:| 
| *Our three assumptions (`First-order Markov property`, `Observation Independance` and `Stationary Process`) simplify the computation of the joint distribution of sequences* |


For instance with the state sequence candidate [`low speed`, `high speed`, `low speed`]
- The **joint probability** is the product of all the probabilities listed on the figure below.
- P([`low speed`, `high speed`, `low speed`] **and** [`right lane`, `left lane`, `right lane`]) = `0.02048`

| ![Derivation of the MLE for a particular observation sequence](docs/results_three.PNG "Derivation of the MLE for a particular observation sequence")  | 
|:--:| 
| *Derivation of the MLE for a particular observation sequence* |

#### Note

`0.1318394` `=` `0.01152` `+` `0.01536` `+` `0.000853` `+` `0.01536` `+` `0.054613` `+` `0.0068267` `+` `0.02048` `+` `0.0068267`

If you sum all the probabilities of the eight cases depicted the figure above, you do not end up to `1`, but to `0.1318394`. Why?
- Well, `0.1318394` represents the **probability for the observation sequence** [`low speed`, `high speed`, `low speed`] to happen.
	- P[`low speed`, `high speed`, `low speed`] = P[`obs`] = P[`obs` `&&` `state seq1`] + P[`obs` `&&` `state seq2`] + ... + P[`obs` `&&` `state seq8`]
	- ?? total probability is with conditional. What is it with intersections?
	- We had already find this result in [Q3](#q3)
- What would sum to `1` is the sum over all possible 3-element observation sequences:
	- `1` = P[`LLL`] + P[`LLH`] + P[`LHL`] + P[`LHH`] + P[`HLL`] + P[`HLH`] + P[`HHL`] + P[`HHH`]

The presented approach could be used for **larger observation sequences**.
By the way you notice that HMM can **handle inputs of variable length**.
But for longer observation sequences, an issue appears:

| Size of the `observation` sequence | Number of probabilities to compute before applying `max()` (for **MLE**) |
| :---:        |     :---:      |
| `1`   | `2`     |
| `2`   | `4`     |
| `3`   | `8`     |
| `i`     | `2^i`       |
| `10`     | `1024`       |

The above method was a **brute force approach**:
- we calculate the **joint probabilities** of a given observation sequence and **ALL possible paths** (state sequences)
- and then **pick THE path** with the **maximum joint probability**.
- Problem: there are exponential number of paths, hence this **brute force search approach** is very **time consuming** and **impractical**.
- Solution: **Dynamic Programming** can be used.

### Dynamic Programming: an alternative to the naive enumeration

Assume that after the second observation, the sub-sequence (`left lane`, `right lane`) is found to be more likely that the sub-sequence (`right lane`, `right lane`).
- Is it **worth carry on some investigation** in the branch (`left lane`, `right lane`)?
- Do not forget that the only goal is to **find the most likely sequence (and nothing else)!**
- Whatever you append to the sub-sequence (`left lane`, `right lane`), the resulting sequence will be less likely than appending the same to (`right lane`, `right lane`).

This intuition will be implemented in the so called Viterbi Algorithm:
- Similar to `alpha` table (from left to right), it calculates the best sequence **storing partial paths**, the paths that are winning so far (and **dropping the ones that have lower probability** so far).
- When it gets to the end, it **goes back using pointers** to get the **most likely path**.

This example show the intuition of `Dynamic Programming`:

> Compute local blocks and consider only the most promising ones to build the next ones.

#### Dynamic Programming is nothing but cached recursion

To better understand the concept of **Dynamic Programming and its benefits**, [`cached_vs_vanilla_recursion.ipynb`](cached_vs_vanilla_recursion.ipynb) compares the performance of two recursive approches on the famous Fibonacci computation.

The first one implements **"classic recursion"**:

```python
def fibo_vanilla_recursive(i):
    if i <= 0:
        return 0
    elif i == 1:
        return 1
    else:
        return fibo_vanilla_recursive(i-1) + fibo_vanilla_recursive(i-2)

%timeit for _ in range(10): fibo_vanilla_recursive(10)
```

```237 µs ± 14.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)```

In the second one reuses previous results in the computation, i.e. a **"cached recursion"**:

```python
cache = {0:0, 1:1}
def fibo_cached_recursive(i):
    if cache.get(i) is not None:
        return cache[i]
    else:
        res = fibo_cached_recursive(i-1) + fibo_cached_recursive(i-2)
        cache[i] = res
        return res

%timeit for _ in range(10): fibo_cached_recursive(10)
```

```2.33 µs ± 140 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)```

The **difference** in the **computational time** is substantial (all the more since it only goes to `10`!).

Let's count how many times each `fibo_vanilla_recursive(i)` is called when computing all `fibo_vanilla_recursive(k)` for `k` in `range(10)`.

| ![Repetition in computation with the vanilla Fibonacci recursion](docs/number_calls_fibo_recursive.svg "Repetition in computation with the vanilla Fibonacci recursion")  | 
|:--:| 
| *Repetition in computation with the vanilla Fibonacci recursion* |

### Viterbi algorithm: similar to the (`alpha`) Forward Algorithm, with `MAX()` instead of `SUM()`

This is a maximization problem:
- The goal is to **find the hidden state sequence** [`lane_t=1`, `lane_t=2`, `lane_t=3`] that **maximizes the joint probability** `P`([`lane_t=1`, `lane_t=2`, `lane_t=3`] and [`low speed` (`t=1`), `high speed` (`t=2`), `low speed` (`t=3`)])
- It can be thought as a **search for the single most likely path**.

Let's call **`L\*`** the **optimal hidden state sequence**, and let's note `L1` = `lane_t=1`, `L2` = `lane_t=2` and `L3` = `lane_t=3`:
- `L*` = `argmax_over_L1_L2_L3`(`P`([`L1`, `L2`, `L3`] and [`low speed` (`t=1`), `high speed` (`t=2`), `low speed` (`t=3`)]))
- This **joint probability** can be turned to a **sum of conditionnal probabilities**:
- `L*` = `argmax_over_L1_L2_L3`(`term_1` `*` `term_2` `*` `term_3`)
- `L*` = `argmax_over_L3`(`argmax_over_L2`(`argmax_over_L1`(`term_1` `*` `term_2` `*` `term_3`)))
- With (simplified using the conditional independance of the HMM structure):
- `term_1` = `P`(`L1`) `*` `P`(`low speed` (`t=1`) given `L1`) `*` (`P`(`L2` given `L1`)
- `term_2` = `P`(`high speed` (`t=2`) given `L2`]) `*` (`P`(`L3` given `L2`])
- `term_3` = `P`(`low speed` (`t=3`) given `L3`])
- Maximizing over [`L1`, `L2`, `L3`] is equivalent to maximizing over [`L1`] over [`L2`] over [`L3`]
- Therefore let's group the terms that depend on what is maximized:
- `L*` = `argmax_over_L3`(`term_3` `*` `argmax_over_L2`(`term_2` `*` `argmax_over_L1`(`term_1`)))
- Now, we can **start from the right side**, solving the **maximization over `L1`** (it does not depend of `L2`)
- Then **store this result in memory and use it** to solve the **maximization over `L2`**
- Finally **use the retult** to solve the **maximization over `L3`**
- This is the idea of the **Dynamic Programming** approach.

Similar to the `alpha`, `beta` and `pi` variables, let's **introduce `alpha\*(i, t)`**:
- `alpha*(i, t)` = `P`([`observed speed` (`t=1`), ..., `observed speed` (`t=t`)] **and** [`L1`, ..., `Lt-1`] being optimal **and** `lane_t` `=` `i`)
- It can be noted that **`alpha\*(i, t)` is very similar to `alpha(i, t)`**:
	- Except that `alpha(i, t)` does not have the term _"and_ [`L1`, ... `Lt-1`] _being_ _optimal"_ 
	- Instead, **`alpha(i, t)` was marginalizing over [`L1`, ... `Lt-1`]**
	- Here, **`alpha\*(i, t)`**, their value are **fixed to the optimal sub-sequence for the time [`1`, `t-1`]**
- A **recursion rule** can be establish:
	- "[`L1`, ..., `Lt`] being optimal" can be written
		- `max_over_j`("[`L1`, ..., `Lt-1`] being optimal **and** `Lt` `=` `j`")
		- And the value that maximizes this quantity is precisely the **optimal value of `Lt\*`** (i.e. ** the `t-th` element in `L\*`**)
	- Then decompose the **joint probability** into some **conditional probability**
	- Simplify the expression using the **conditional independance**:
	- It yields to:
		- `alpha*(i, t+1)` = `P`(`speed_t+1` given `Lt+1=i`]) `*` `P`(`Lt+1` given `Lt`]) * `max_over_j`[`alpha*(j, t)`]
	- This is **very similar to the `alpha` construction**, except that the elements at `t+1` are constructed **using the `max()`** over elements at `t` **instead of summing** all elements at `t`.
	- **It is important, when solving the `max()`, to store the `argmax()`, i.e. the `lane` that has the higest `alpha\*`**.
		- This information will be used to **derive the best sequence `L*`**.
- Initialisation:
	- For `t=1`, there is no `max()` operation
	- Hence `alpha*(i, t=1)` `=` `P`( `speed_t=1` given `L1` `=` `i`]) `*` `P`( `L1` `=` `i`)
	- In other words, **`alpha\*(i, t=1)` == `alpha(i, t=1)`**
- Inferrence: how to **find the elements** of the **optimal sequence `L\*`**?
	- Start by applying `argmax()` in the **last column**. It gives the optimal value of `Lt` (**last state in `L\*`**)
	- Then, for each timestep `t`, starting by the end, **query the memory** and **find `argmax()` at `t-1`** that has been used to compute this `alpha*`.
	- This is the reason why it is important to **store the transitions resulting of `max()` operations** when builing the `alpha*` table.
	
| ![Construction of the `alpha* table` using Dynamic Programming](docs/alpha_star_table.PNG "Construction of the `alpha* table` using Dynamic Programming")  | 
|:--:| 
| *Construction of the `alpha\* table` using Dynamic Programming* |

Essentially, this algorithm defines `alpha\*(i, t)` to be the probability of the most likely path through state `state_t` = `i`, and it recursively computes  `alpha\*(i, t)` using the emission model and the `alpha\*(j, t-1)` weighted by transition probabilities.

Applying **`max()` in the last column** gives the **joint probability for the most probable state sequence**:
- `0.0546` = P([`low speed`, `high speed`, `low speed`] **and** [`right lane`, `right lane`, `right lane`])
- This result had already been found in the "naive approach" above.

| ![Use of the `alpha* table` for the **decoding** of an observation sequence](docs/alpha_star_table_decoding.gif "Use of the `alpha* table` for the **decoding** of an observation sequence")  | 
|:--:| 
| *Use of the `alpha\* table` for the **marginal probability** of an **observation sequence*** |

Similar to the construction of the `alpha` table (Forward algorithm), the Viterbi decoding algorithm has:
- **`O`(`S^2`\*`T`) time complexity**,
	- It has reduced from **exponential** (the brute force had: `O`(`2\*T`\*`S^T`)) to **polynomial**:
	- Linear in the length of the sequence and quadratic in the size of the state space.
- And **`O`(`S`\*`T`) space complexity**, to remember the **pointers**.
- Where `S` denotes the number of hidden states and `T` the length of the observation sequence.

**Decoding** only requires only two types of information:
- The `alpha*` values in the **last column**.
- The **pointers**, i.e. the transitions followed between columns when building the table (to recover the `argmax`).

In other words, **`alpha\*` values in the non-last columns are useless for decoding**:
- No `argmax()` operation is performed to construct the optimal sequence from the table, except for the last column.
- It **could be the case** that `alpha\*`(`left_lane`, `t=2`) could be larger than `alpha\*`(`right_lane`, `t=2`).
- But that due to **the transition probabilities**, it is **not chosen with `argmax()`** in the next column.
	- Remember that the `max()` operation is on the **product `alpha\*` `\*` `transition`**. Not just on `alpha*`.
- Nevertheless in this case **`alpha\*`(`right_lane`, `t=2`) would still be chosen when building L\*** since it is located on the optimal path.

Answer:
- [`right_lane`, `right_lane`, `right_lane`] is the **most likely `lane` sequence** if the **observation sequence** is [`low speed`, `high speed`, `low speed`]
- This had been found using the brute force approach.

Difference between the **Viterbi algorithm** and **Posterior decoding**:
- Posterior decoding provides the sequence of the most likely state at any point in time.
- Although the **Viterbi decoding** algorithm provides one means of estimating the hidden states underlying a sequence of observed characters, another valid means of inference is provided by **posterior decoding**.
- instead of identifying a single path of maximum likelihood, posterior decoding considers the probability of **ANY path lying in state `i`** at **time `t`** given **the whole observation sequence**.
	- i.e. P(`state_t` = `i` |[`obs_1` ... `obs_t` ... `obs_T`]).
	- The state that maximizes this probability for **a given time `t`** is then considered as **the most likely state at THAT point**.
- Why calling it *Posterior* Decoding?
	- Without seeing any observation, we have the prior that the vehicle is more likely to be driving on the `right_lane` (p=`2/3`).
	- Now appears the first observation. We update our prior using this information via Bayes rule:
		- p(`right_lane` | `obs_1`) = p(`obs_1` | `right_lane`) \* p(`right_lane`) / p(`obs_1`)
		- In other words: `Posterior` = normalized(`Prior` \* `Likelihood`)
		- If `obs_1` is `low_speed`, our **belief** that `state_1` is `right_lane` is **reinforced**.
		- If `obs_2` is also `low_speed`, information flow backwards from the second observation and **reinforces** our **belief** about `state_1` even more.
	- This example shows the way information flows **backward** and **forward** to affect our belief about the states in Posterior Decoding.
		- The computation uses both the forward algorithm and the backward algorithm.
		
- Can Posterior Decoding provide a path like the Viterbi algorithm would?
	- Yes, it can.
	- since Posterior Decoding can provide the most likely state for every point, assembling these states provides the path.
- Can Viterbi algorithm and Posterior decoding disagree on the path?
	- Yes, they can.
	- When trying to classify each hidden state, the Posterior decoding method is more informative because it takes into account all possible paths when determining the most likely state.
		- a sequence of **point-wise** most likely states
		- it may give an invalid sequence of states!
		- For example, the states identified at time points t and t + 1 might have zero transition probability between them.		
	- The Viterbi method only takes into account one path, which may end up representing a minimal fraction of the total probability

## Q5 - How to estimate the HMM parameters with no state annotations in the training data?

### Unsupervised estimation of parameters of an unlabelled dataset

Supervised learning:
- In [Q1](#q1), we were given a **training data with labels**.
- I.e. each `speed` measurement was associated to a `lane` state.

Unsupervised learning:
- Here, were are only supplied with some sequences of `speed` observation.
- Since the training dataset **contains no `lane` annotation**, we needed to **both** estimate **model parameters** and identify the **`lane` states**.

An **iterative approache** can be used for this **unsupervised learning** problem.
- Suppose we have some **prior believes** about the **HMM models**.
	- We could use decoding methods to **infer the hidden states** underlying the provided observation sequence.
	- These could **constitue our annotations** and we are back to **supervised learning**: we can estimate the HMM models (by counting as in [Q1](#q1)).
- We can repeat this procedure until the improvement in the data’s likelihood remains relatively stable.

The unsupervised estimation of parameters of an unlabelled dataset can be implemented using the concept of **Expectation Maximization** (**EM**).

### EM algorithm

To get familiar with the concept of EM Algorithm, I recommend having a look at this [short series of video](https://www.youtube.com/watch?v=REypj2sy_5U&list=PLBv09BD7ez_4e9LtmK626Evn1ion6ynrt) by Victor Lavrenko about Gaussians Mixture Models.

He justify the commonly used image of **"chicken and egg problem"**:
- Given true parameters, it would be easy to assign a class distribution (generative posterior probabilities) for each point.
- Given true assignements, it would be easy to estimate the parameters (e.g. `mu` and `sigma` if Gaussian as well as the priors), weighting posteriors similar to K-means.

And he draw parallels and stresses differences of **EM for Gaussian Mixtures Models** and **K-means**
- concept of **soft vs hard clustering**,
- use of Bayesian probabilities,
- non-uniformity in priors,
- and **recomputation of covariances** at each iteration.

Convergence:
- The power of **EM** lies in the fact that P([`obs_sequence`]|`HMM_parameters`) is **guaranteed to increase** with each iteration of the algorithm.
- When this probability converges, a **local maximum** has been reached.
- Since the EM algorithm is a gradient-based optimization method, it will generally get stuck in local optimal.
- To find the **global maximum**, one could run this algorithm with various initializations and select the highest scored model.

Der Baum-Welch-Algorithmus ist ein erwartungsmaximierender Algorithmus. Er berechnet die Maximalwahrscheinlichkeitsschätzungen (Maximum-Likelihood-Schätzwerte) und die sogenannten Posterior-Mode-Schätzungen für die gegebenen Parameter (Übergangs- und Emissionswahrscheinlichkeit) eines HMMs, wenn nur die Emissionen als Trainingsdaten gegeben sind. 

Der Algorithmus arbeitet in zwei Schritten:
- Erstens berechnet er die Vorwärtswahrscheinlichkeit (forward probability) und die Rückwärtswahrscheinlichkeit (backward probability) für jeden Zustand des HMMs.
- Zweitens, auf der Basis dieses ersten Schrittes, berechnet er die Frequenz der Übergangs-Emissions-Paar-Werte und dividiert diese durch die Wahrscheinlichkeit des gesamten Strings (sog. posterior decoding).
- Dies führt zu der Berechnung der erwarteten Zählung des einzelnen Übergangs-Emissions-Paares.
- Jedes Mal, wenn ein einzelner Übergang gefunden wird, erhöht sich der Wert des Quotienten des Übergangs und der Wahrscheinlichkeit der gesamten Zeichenkette.
- Dieser Wert kann dann zum neuen Wert des Übergangs gemacht werden.

#### Implementation
Both methods take an **observation sequence** as **input** and start with some prior on the HMM parameters (possibly a trivial one with **completely uniform probabilities**).

`1/2`: Expectation Maximization using Viterbi training.
- **`E`**-step: Viterbi decoding is performed to find THE best sequence of hidden states. The **E**stimate forms the annotation.
- **`M`**-step: the new parameters are computed using the simple counting formalism in supervised learning (**M**LE).
- Termination: Repeat the E and M steps until the likelihood P([`obs_sequence`]|`HMM_parameters`) converges.

`2/2`: Expectation Maximization: The Baum-Welch Algorithm
- Initialisation: HMM parameters are initialized to some best-guess parameters.
- **`E`**-step: considering the observation sequence, the **`E`XPECTED probability** is estimated for hidden states.
- **`M`**-step: based on the **probability distribution** of hidden states, new HMM parameters are estimated using **`M`AXIMUM likelihood Estimate** techniques.
- Termination: Repeat the E and M steps until the likelihood P([`obs_sequence`]|`HMM_parameters`) converges.

A parallel can be drawn with the difference between **point-based estimates** and **distributions** for estimating parameters (for instance the position of an object):
- Baum-Welch uses **distribution over states** to estimate the HMM parameters in the M-step.
	- Hence the Baum-Welch algorithm computes **exact state occupancies**.
	
- In Viterbi-EM, **annotations** are used.
	- Viterbi algorithm only finds the single most likely path, and its corresponding probability.
	- Hence annotations are the **best path elements** (e.g. [`right_lane`, `left_lane`, `left_lane` ...]) that have the **highest probability** (here the problem is discrete) in the distribution.

Remember that the Viterbi decoding **only considers THE most probable hidden path** instead of the collection of all possible hidden paths.
- This approximation causes the training to converge rapidly.
- The resulting parameter estimations are usually inferior to those of the Baum-Welch Algorithm.
- But Viterbi-EM can be used to **quickly get an estimate** of the model parameters, before **fine tuning them with Baum-Welch**.

### Baum-Welch

We know how to compute the likelihood P([`obs_sequence`]|`HMM_parameters`) using either the **forward** or **backward** algorithm’s final results:
- Sum over **last column** in the `alpha` table.
- Sum over **first column** in the `beta` table.

How to perform the M-step, i.e. to update the paramters using the expectation over state annotations?
- Let's call `θ` the HMM parameters (emission and transition probabilities).
- In our simple example, we are interested in learning contains `2^3` + `2` = `10` parameters:
	- P(`state_1` = `right_lane`)
	- P(`state_1` = `left_lane`)
	- P(`state_t+1` = `right_lane` | `state_t` = `right_lane`)
	- P(`state_t+1` = `right_lane` | `state_t` = `left_lane`)
	- P(`state_t+1` =  `left_lane` | `state_t` = `right_lane`)
	- P(`state_t+1` =  `left_lane` | `state_t` = `left_lane`)
	- P(`obs_t` =  `low_speed` | `state_t` = `right_lane`)
	- P(`obs_t` =  `low_speed` | `state_t` = `left_lane`)
	- P(`obs_t` = `high_speed` | `state_t` = `right_lane`)
	- P(`obs_t` = `high_speed` | `state_t` = `left_lane`)
- Let's note the likelihood P([`obs_sequence`]|`HMM_parameters`) = P(`x`|`θ`).
- We are looking for `θ*` = `argmax`[P(`x`|`θ`)]
	- i.e. the parameters that gives the highest probability for the observations to have been emitted by the HMM.
	- i.e. the parameters that explains the best the observations
 
We know how to do smoothing:
- We introduced `gamma`(`lane_k`, `time_t`):
- `gamma`(`lane_k`, `time_t`) = `alpha(k, t)` \* `beta(k, t)` / P(`x`|`θ`).
- It is the probability that the system is at `lane` `k` at the `t`-th time step, given the full observation sequence `x` and the model `θ`.
- Summing the last column of the `alpha` table gives P(`x`|`θ`) = Sum over `k` of `alpha(k, T)`.

In supervised learning (when working with single annotations), we processed by counting:
- For instance P(`state_t+1` = `left_lane` | `state_t` = `right_lane`) = `#`[`right`->`left`] / (`#`[`right`->`left`] + `#`[`right`->`right`])
- Here, we need to derive **Expectation** over these counts, for instance with:
	- the probability of being at state `k` at time `t`.
	- the probability of being at state `k` at time `t`, and `l` at time `j+1`.

Ingredient `1/2`: Expectation of `state` counts.
- At the **denominator**: how many times is the state trajectory **expected** to **transition from state `right`**?
- We can **use the `gamma` variable** introduced for **smoothing**:
- EXPECTED(`#`[transitions from `right_lane`]) = Sum for time `t=1` to time `t=T` of `gamma`(`right_lane`, `time_t`)

- This means that the probability of being in state ‘Si ’ at time ‘t’ is α t(i)* β t(i)
	- The summation of α t(i)* β t(i) over all time ‘t’ gives the probability of ever being in state Si at any time.
- So the summation of α T(i)* β T(i) over all states ‘Si ’ gives the probability of  observing the observation sequence ‘O’. 
- Therefore, the probability of being in state Si given the entire observation sequence ‘O’ is =  α t(i)* β t(i)/ Σi α T(i) 
- For the sake of convinience, we define γ i(t) = P(Q t = S i| O) 

Ingredient `2/2`: Expectation of `transition` counts.
- Similar to `gamma`, we introduce `xi`:
- `xi`(`lane_k`,`lane_l`, `time_t`) is the probability of being at state `k` at time `t`, and at state `l` at time `t+1`, given the full observation sequence and the current HMM model.
- `xi` can be computed using `alpha` values, `beta` values and the current HMM model.
- EXPECTED(`#`[`lane_k`->`lane_l`]) = Sum for time `t=1` to time `t=T-1` of `xi`(`lane_k`, `lane_l`, `time_t`)
- This is calculating the probabilities of being in a state at the times that the particular observation happened divided by the probabilities that we are in that state at any time.

Now we can derive update rules for the HMM parameters:
- `new` P(`lane_k`->`lane_l`) = `E`[# of transitions from state `k` to state `l`] / `E`[# of transitions from state `k`]
	- The numerator is: Sum for time `t=1` to time `t=T-1` of `gamma`(`right_lane`, `time_t`)
	- The denominator is: Sum for time `t=1` to time `t=T-1` of `xi`(`lane_k`, `lane_l`, `time_t`)
- `new` P(`lane_k` emits `speed_s`) = `E`[# of times in state `k`, when the observation was `s`] / `E`[# of times in state `k`]
	- The numerator is: Sum for time `t=1` to time `t=T` of `gamma`(`lane_k`, `time_t`) \* 1[`obs_t`==`s`]
	- The denominator is: Sum for time `t=1` to time `t=T` of `gamma`(`lane_k`, `time_t`)

To sumarize, working with distributions, an expression can be derived using the `alpha` and `beta` values:
- P(`state_t+1`=`l`|`state_t`=`k`) = sum over `t` of `alpha(k, t)` \* P(`state_t+1`=`l`|`state_t`=`k`) \* P(`obs_t+1`|`state_t+1`=`l`) \* `beta(l, t+1)` / P(`x`|`θ`)
- P(`obs_t`=`b`|`state_t`=`k`) = sum over `t` of `alpha(k, t)` \* `beta(k, t)` / P(`x`|`θ`)

The Baum-Welch **iteration loop** repeats the following steps until P([`obs_sequence`]|`HMM_parameters`) converges:
- Run the **forward** algorithm to derive the `alpha` table.
- Run the backward algorithm to derive the `beta` table.
- Calculate the new **log-likelihood P([`obs_sequence`]|`HMM_parameters`)**.
- **Update the Transition and Emission** models (potentially using pseudocounts).
	- based on the Estimate of the frequency of transition and emission for each pairs.

Complexity:
- The time complexity of the **forward** and **backward** algorithms was `O(S\*S\*T)`.
- When running them, we have all of the information necessary to calculate the likelihood and to update the emission and transition probabilities during each iteration.
- Updates are **constant time operations** once P(x|θ), fk(t) and bk(t) have been computed,
- Hence the **total time complexity** for this Baum-Welch algorithm is **`O(S\*S\*T\*N)`**, where
	- `S` denotes the number of hidden states.
	- `T` the length of the observation sequence.
	- `N` is the total number of iterations.

How to **encode your prior beliefs** when learning with Baum-Welch?
- Those prior beliefs are encoded in the **initializations** of the **forward and backward algorithms**

Convergence and overfitting.
- First, it is possible that P(`x`|`θ`) `>` P(`x`|`θ_true`).
	- This is due to overfitting over one particular data set.
- Then, the prior (or initial guess) on the transition and emission models play an important role
	- In this simple discrete problem, starting with uniform prior may result in the transition matrix for the states to remain uniform and estimates to be based on simple counts, as for supervised learning.
	- In other words, the Baum Welch algorithm ignores the fact that we have an HMM and goes for the maximum likelihood estimate of the transition and emission probabilities and reports that this is the observation probability for both states.
	- This is not wrong.
	- One way to prevent this it is to give more informed priors. For instance initializing with the first approximation obtained with the Viterbi-EM.
	- Another possibility is to use multivariate Gaussian mixtures for the observation probabilities instead of discrete values
		- Concretely, speed measurement could be continuous (e.g. with a emission model).
 
### Baum-Welch notebook

pass

## Q6 - After a long time, what is the probability for the car to be in each lane?

In an HMM:
- An **observation** is influenced by its **underlying state**.
- A **state** is only **function** of the **previous state**:
	- Its evolution is entirely controlled by the **HMM transition matrix**.
	- The **stochastic variable `state` **follows a **Markov chain**.

Let's focus on the **random process** followed by the `state` variable.

What is its **stationary distribution**, i.e. the probability distribution that **remains unchanged** in the Markov chain as time progresses?
- Let's call the stationary distribution `π` = [p_stat(`left_lane`), p_stat(`right_lane`)].
- Let's call `T_M` the **transition matrix**.
- If **convergence** is reached, the distribution `π` is **invariant** by the matrix `T_M`.
	- `π` = `π``T_M`
	- All entries must be **non-negative** and **sum to `1`**.
	- The **sum to `1`** can be obtained by **normalization**.
- Transposing: `π`.T = `T_M`.T `π`.T
	- Hence `π`.T is an **eigenvector of `T_M`.T** with **eigenvalue `1`**.
	- In other words, the **transposed transition matrix** has **eigenvectors with eigenvalue** that are **stationary distributions expressed as column vectors**.
	- In short, the **stationary distribution** is a left eigenvector (as opposed to the usual right eigenvectors) of the transition matrix.
- This equation can have **zero**, **one** or **several solutions**.
	- **Ergodic Markov chains** have a **unique** stationary distribution.
	- **Absorbing Markov** chains have stationary distributions with nonzero elements only in so-called **"absorbing states"**.
- This gives information about the **stability** of a **random process**.

## Summary

### Scoring
How likely is some observation sequence to be emitted (potentially in association with a state sequence)?

- One-path Scoring
	- The single path calculation is the **likelihood** of observing the **given observation sequence** over **ONE particular state sequence** (also called **path**)
	- It computes the joint probability using the decomposition **P(`obs`, `state`) = P(`obs` | `state`) \* P(`state`)
	- *??what is the name of this decomposition??*
- All-path Scoring
	- It computes given sequence of observations or emissions regardless of the 
	- It computes the probability of the observation sequence using the marginalization: P(`obs`) = Sum over all `state` of P(`obs`, `state`) where P(`obs`, `state`) can be seen as a one-path score.
	- The **forward algorithm** calculates the exact sum iteratively by using dynamic programming
	- It can be computed by summing the last column in the `alpha` table. Or the first column in the `beta` table.

Instead of computing the probability of a **single path** of hidden states emitting the observation sequence (Viterbi), the **forward algorithm** calculates the probability of the **observation sequence being produced by ALL possible paths**.
- The forward algorithm introduces the **`alpha` values**: the joint probability of **observing the first `t` observations** and being in **state `k` at time `t`**.
- The backward algorithm introduces the **`beta` values**: the conditional probability of **observing the observations from time `t` to the end** given the **state at time `t`**.
- Given that the number of paths is exponential in t, **dynamic programming** must be employed to solve this problem.
- Viterbi and Forward algorithm share the same recursion. But Viterbi algorithm uses the maximum function whereas the forward algorithm uses a sum

### Decoding
Given some observed sequence, what path gives us the maximum likelihood of observing this sequence?
- **Decoding** looks for a **path** (sequence of states).
- **Filtering** and **Smoothing** look for most likely state for **ONE single time**.

- One-path Decoding:
	- **Viterbi decoding algorithm** finds the most probable state path, i.e. **THE hidden state sequence** (a path) that **maximizes the joint probability** of the observation sequence [`obs_t1` ... `obs_tn`] and hidden state sequence [`state_t1` ... `state_tn`], i.e. P([`obs_t1` ... `obs_tn`], [`state_t1` ... `state_tn`]).
	- It is a **dynamic programming** algorithm: the best path can be obtained based on the best path of the previous states.
	- The `alpha*(i, t)` variable represents the probability of the most likely **path ending at state `i`** at time `t` in the path.
	- By keeping pointers backwards, the actual hidden state sequence can be found by backtracking.
	- Viterbi can be used to give a first approximation of the all-path scoring:
		- But it is just a small fraction of the probability mass of all possible paths.
		- Hence the approximation is valid only if this particular path has high probability density.
	
- All-path Decoding:
	- **Posterior Decoding** returns the sequence of hidden states that contains the **most likely states at any time point**.
		- It uses both the forward and the backward algorithm.

## Learning
Given some observation sequence (and potentially the associated state sequence), what are the most likely HMM parameters?

Let's call `θ` the HMM parameters (emission and transition probabilities), and `π` a **path** (i.e. a sequence of hidden states).

- One-path Learning:
	- In supervised learning, the **counting method** (MLE) looks for `argmax_over_θ` of [P(`x`, `π`|`θ`)], given the annotation true `π` (annotation).
	- In unsupervised learning, **Viterbi-EM** looks for `argmax_over_θ` of `MAX_over_π` of [P(`x`, `π`|`θ`)].
	
- All-path Learning:
	- In unsupervised learning, **Baum-Welch-EM** looks for `argmax_over_θ` of `SUM_over_π` of [P(`x`, `π`|`θ`)].
	
# Acknowledgement and references
I learnt and took some inspiration of:
- A [video series](https://www.youtube.com/playlist?list=PL6Xpj9I5qXYGhsvMWM53ZLfwUInzvYWsm) (in French) by Hugo Larochelle.
- A [video](https://www.youtube.com/watch?v=kqSzLo9fenk) by Luis Serrano.
- A [course](https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-410-principles-of-autonomy-and-decision-making-fall-2010/lecture-notes/) by Williams and Frazzoli, based on their experiences in the DARPA Urban Challenge.
- A [lecture](http://web.mit.edu/6.047/book-2012/Lecture08_HMMSII/Lecture08_HMMSII_standalone.pdf) from Mavroforakis and Ezeozue.
- A series of three [blog posts](http://www.davidsbatista.net/blog/2017/11/11/HHM_and_Naive_Bayes/) by David Soares Batista.
- Some useful [Q&A](http://www.speech.zone/forums/topic/viterbi-vs-backward-forward/) in Simon King's [speech.zone](http://www.speech.zone/) forum. 
- An [article](https://brilliant.org/wiki/stationary-distributions/) from [brilliant.org](https://brilliant.org/) about stationary distributions in Markov chains and HMMs.

# Bonus

To go further, are some **Research Papers** implementing HMMs for **Autonomous Driving**. The list is not exhaustive.

HMMs are been used for **manoeuvre recognition** and **driving behaviour estimation**, both serving **prediction purposes**.

[1]	S. Liu, K. Zheng, S. Member, L. Zhao, and P. Fan, **"A Driving Intention Prediction Method Based on Hidden Markov Model for Autonomous Driving,"** 2019.
[[pdf]](https://arxiv.org/pdf/1902.09068.pdf)

[2]	M. Zhao, **"Modeling Driving Behavior at Single-Lane Roundabouts,"** 2019.
[[pdf]](https://publikationsserver.tu-braunschweig.de/receive/dbbs_mods_00066445)

[3]	P. Vasishta, D. Vaufreydaz, and A. Spalanzani, **"Building Prior Knowledge: A Markov Based Pedestrian Prediction Model Using Urban Environmental Data."** 2018.
[[pdf]](https://arxiv.org/pdf/1809.06045.pdf)

[4]	S. B. Nashed, D. M. Ilstrup, and J. Biswas, **"Localization under Topological Uncertainty for Lane Identification of Autonomous Vehicles,"** 2018.
[[pdf]](https://arxiv.org/pdf/1803.01378.pdf)

[5]	T. Ganzow, **"Real-time detection of traffic behavior using traffic loops,"** 2018.
[[pdf]](https://staff.fnwi.uva.nl/a.visser/education/masterProjects/vanderHamThesis.pdf)

[6]	Y. Zhang, Q. Lin, J. Wang, S. Verwer, and J. M. Dolan, **"Lane-change Intention Estimation for Car-following Control in Autonomous Driving,"** 2018.
[[pdf]](https://www.researchgate.net/publication/324174189_Lane-change_Intention_Estimation_for_Car-following_Control_in_Autonomous_Driving)

[7]	W. Yuan, Z. Li, and C. Wang, **"Lane-change prediction method for adaptive cruise control system with hidden Markov model,"** 2018.
[[pdf]](https://www.researchgate.net/publication/327888086_Lane-change_prediction_method_for_adaptive_cruise_control_system_with_hidden_Markov_model)

[8]	E. Yurtsever et al., **"Integrating Driving Behavior and Traffic Context through Signal Symbolization for Data Reduction and Risky Lane Change Detection,"** 2018.
[[html]](https://ieeexplore.ieee.org/document/8370754)

[9]	G. Xie, H. Gao, B. Huang, L. Qian, and J. Wang, **"A Driving Behavior Awareness Model based on a Dynamic Bayesian Network and Distributed Genetic Algorithm,"** 2018.
[[pdf]](http://hive-hnu.org/uploads/soft/20190127/1548561157.pdf)

[10]	N. Deo, A. Rangesh, and M. M. Trivedi, **"How would surround vehicles move? A Unified Framework for Maneuver Classification and Motion Prediction,"** 2018.
[[pdf]](https://arxiv.org/pdf/1801.06523.pdf)

[11]	X. Geng, H. Liang, B. Yu, P. Zhao, L. He, and R. Huang, **"A Scenario-Adaptive Driving Behavior Prediction Approach to Urban Autonomous Driving,"** 2017.
[[pdf]](https://pdfs.semanticscholar.org/0b63/2048208c9c6b48b636f9f7ef8a5466325488.pdf)

[12]	D. Lee, A. Hansen, and J. Karl Hedrick, **"Probabilistic inference of traffic participants lane change intention for enhancing adaptive cruise control,"** 2017.
[[html]](https://ieeexplore.ieee.org/document/7995823)

[13]	W. Song, G. Xiong, and H. Chen, **"Intention-Aware Autonomous Driving Decision-Making in an Uncontrolled Intersection,"** 2016.
[[pdf]](https://www.researchgate.net/publication/301718813_Intention-Aware_Autonomous_Driving_Decision-Making_in_an_Uncontrolled_Intersection)

[14]	S. Lefévre, A. Carvalho and F. Borrelli, **"A Learning-Based Framework for Velocity Control in Autonomous Driving,"** 2015.
[[pdf]](https://borrelli.me.berkeley.edu/pdfpub/Stephanie_2016_TASE.pdf)

[15]	M. Schreier, **"Bayesian environment representation, prediction, and criticality assessment for driver assistance systems,"** 2015.
[[pdf]](https://core.ac.uk/download/pdf/76650732.pdf)

[16]	A. Carvalho, S. Lefévre, G. Schildbach, J. Kong, and F. Borrelli, **"Automated driving: The role of forecasts and uncertainty - A control perspective,"** 2015.
[[pdf]](https://scinapse.io/papers/2014414177)

[17]	B. Tang, S. Khokhar, and R. Gupta, **"Turn prediction at generalized intersections,"** 2015.
[[pdf]](https://www.researchgate.net/publication/283214809_Turn_Prediction_at_Generalized_Intersections)

[18]	T. Streubel and K. H. Hoffmann, **"Prediction of driver intended path at intersections,"** 2014.
[[pdf]](https://www.researchgate.net/publication/269294116_Prediction_of_driver_intended_path_at_intersections)

[19]	C. Laugier et al., **"Probabilistic analysis of dynamic scenes and collision risks assessment to improve driving safety,"** 2011.
[[pdf]](https://www.researchgate.net/publication/229034149_Probabilistic_Analysis_of_Dynamic_Scenes_and_Collision_Risk_Assessment_to_Improve_Driving_Safety)

[20]	G. S. Aoude, V. R. Desaraju, L. H. Stephens, and J. P. How, **"Behavior classification algorithms at intersections and validation using naturalistic data,"** 2011.
[[pdf]](http://acl.mit.edu/papers/IV11AoudeDesarajuLaurensHow.pdf)

[21]	D. Meyer-delius, C. Plagemann, and W. Burgard, **"Probabilistic Situation Recognition for Vehicular Traffic Scenarios,"** 2009.
[[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.421.2906&rep=rep1&type=pdf)

[22]	H. Berndt and K. Dietmayer, **"Driver intention inference with vehicle onboard sensors,"** 2009.
[[html]](https://ieeexplore.ieee.org/document/5400203)

[23]	D. Meyer-Delius, C. Plagemann, G. von Wichert, W. Feiten, G. Lawitzky, and W. Burgard, **"A Probabilistic Relational Model for Characterizing Situations in Dynamic Multi-Agent Systems,"** 2008.
[[pdf]](http://ais.informatik.uni-freiburg.de/publications/papers/meyerdelius07gfkl.pdf)

[24]	N. Oliver and A. P. Pentland, **"Driver behavior recognition and prediction in a SmartCar,"** 2000.
[[pdf]](http://www.nuriaoliver.com/driverbehavior/spie2000.pdf)