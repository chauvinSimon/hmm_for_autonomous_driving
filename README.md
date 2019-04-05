# hmm_for_autonomous_driving

WORK IN PROGRESS!!

# Introduction
Disclaimer:

- The goal of this repository is to get more familiar with the concepts of **`Hidden Markov Models`** (**= HMM**).
- The scope of ambition is therefore limited and the examples are very simple, only serving educational purposes.

Addressed topics:
- Problem setting using a basic example to introduce the **HMM terminology**.
- Illustration of the **Bayes Rule** to determine the most likely sequence of state given a observation sequence.
- Implementation of the **Forward and Backward Algorithm** to compute the probability of a particular observation sequence.
- Implementation of the **Viterbi Algorithm** to find the most likely sequence of hidden states which could have generated a given observation sequence (**posterior decoding**).
- Implementation of the **Baum-Welch Algorithm** to find the most likely parameters (state transition and emmission models) of the HMM given a observation sequence (**parameter learning** task).

Given the model parameters, : is Solved by the Viterbi algorithm and **Posterior decoding**.

# Problem motivation
For left-hand-drive countries such as the UK, just invert the reasoning :smiley:

- Your car is driving on a **2-lane highway**.
- Imagine that you can **remotely monitor the velocity of the car (I communicate it to you)**
- But you do have **no direct access to the lateral position** (`right lane` of `left lane`).
- How could you deduce the `lane` based on the single information we receive (the `speed`)?

#### Emission probability

If I am telling you that I am driving with a `low speed`, you **may deduce** that I am on the right lane.
- For instance, because I am just driving alone at a reasonable pace
- Or because I am blocked by a slow vehicle.
- But I could also drive fast on this `right lane` *(have you ever been driving alone on a German highway?)*

Similarly, if you get informed of a `high speed`, you could say that I am **more likely** to be driving on the left lane.
- Probably overtaking another vehicle.
- Nevertheless, this is **not always true**: think of the situation where you are waiting on the left lane behind a truck trying to overtake another truck.

We get a **first intuition**:
- The variable `lane` seems to have an impact on the variable `speed`.
- In  other words: **you do not drive at the same pace depending if you are one the `left lane` or the `right lane`**.
- But the relation is **not deterministic**, rather **stochastic**.

This finding will be modelled using **`emission probabilities`** in the following.

#### Transition probability

You could have another intuition:
- Human drivers usually **stay on their lanes**.
- Hence if you are on `right lane` at time `t`, you are likely to still be on `right lane` at time `t+1`.
- Again, this **does not always hold** and you can find **exception**.
- But here comes a second intuition: **the `lane` at time `t` is influenced by the `lane` at time `t-1`**.

The concept of **`transition probability`** will be used to model this second remark.

### Terminology

| ![The speed is the `observation` while the lane constitutes the `hidden state`. Some examples show that all `emissions` are possible](docs/terminology.PNG "The speed is the `observation` while the lane constitutes the `hidden state`. Some examples show that all `emissions` are possible")  | 
|:--:| 
| *The speed is the `observation` while the lane constitutes the `hidden state`. Some examples show that all `emissions` are possible* |


### Objective

> We want to **infer the lane** (`right` of `left`) of the car (= **hidden state**) based on a **sequence of speed measurements** (= **observation**).

### Assumptions
To keep the problem as simple as possible:
- Let's **discretize the speed** into `low speed` and `high speed`.
- Time steps are discretized.
- Lane transitions are ignored: either you are on `left lane` or you are on `right lane`.

A word about the **Markov Property**:
- We just said that it is useful to know the present `lane` (at time `t`) to infer the future `lane` (at time `t+1`).
- What about the previous `lane` at `t-1`? It probably also hold relevant information?
- Here is a strong assumption about inferring in this stochastic process:
	- the conditional probability distribution of **future states** of the process (conditional on both past and present states) **depends only upon the present state**, not on the sequence of events that preceded it.
- In other words, **"the future is independant of the past given the present"**.
- This **strong assumption** is known as the **Markov Property** (also named **"memoryless property"**) and will make computations easier in the following.

# Problem formulation

For HMM:

- **Hidden state**: discrete random variable `lane` in {`Right Lane`, `Left Lane`}
- **Observation**: discrete random variable `speed` in {`Low Speed`, `High Speed`}
- **Emission probability**: `P[speed(t)` given `lane(t)]`
- **Transition probability**: `P[lane(t+1)` given `lane(t)]`
- **Initial state probability**: `P[lane(t)]`

# Questions:
- [Q1](#q1) - How to **derive the probability models**?
- [Q2](#q2) - If you receive a **single speed observation**, what are the probability for the car to be in each lane?
- [Q3](#q3) - What is the **probability of an observation sequence**? For instance [`low speed`, `high speed`, `low speed`].
- [Q4](#q4) - What is the **most likely `lane` sequence** if the **observation sequence** is [`low speed`, `high speed`, `low speed`]?
- [Q5](#q5) - What if you are **not directly given the probability models**?

# Answers

## Q1 - How to derive the probability models?

Let assume we are given some data: a observation sequence and states.
- Here are only a few for simplicity but you could imagine longer recordings.
- You can think of it as some **samples** of the underlying **joint distributions**.

The idea is to **approximate the model parameters by counting the occurrences**.

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
- For instance, how many times has the hidden state `left lane` caused a `high speed` observation?

| ![Derivation of the emission probability model](docs/deriving_emission_model.PNG "Derivation of the emission probability model")  | 
|:--:| 
| *Derivation of the emission probability model* |

## Initial state probability

At any time `t`, what is your guess on the **distribution of the hidden state if no observation is given**?
- Two options are available:
	- Either you use the transition model (1) and the fact that probabilities sum to `1` (2):
		- (1) P[`left lane`, `t`] = P[`left lane` -> `left lane`] `*` P[`left lane`, `t-1`] + P[`right lane` -> `left lane`] `*` P[`right lane`, `t-1`]
		- (2) P[`right lane`, `t`] = `1` - P[`left lane`, `t`]
	- Either you simply **count occurrences** in the supplied data:
		- P[`left lane`, `t`] = `1/3`
		- P[`right lane`, `t`] = `2/3`

### Summary

| ![Hidden Markov Model with the `initial state probability` model (up), `transition probability` model (middle), and the `emission probability` model (below)](docs/hmm.PNG "Hidden Markov Model with the `initial state probability` model (up), `transition probability` model (middle), and the `emission probability` model (below)")  | 
|:--:| 
| *Hidden Markov Model with the `initial state probability` model (up), `transition probability` model (middle), and the `emission probability` model (below)* |

## Q2 - If you receive a single observation, what are the probability for the car to be in each lane?

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

This method is sometimes named **"Posterior Decoding"**.

## Q3 - What is the probability of an observation sequence?

pass

## Q4 - What is the most likely `lane` sequence if the observation sequence is [`low speed`, `high speed`, `low speed`]?

### Question interpretation

When trying to **reformulate the question**, I was puzzled since I ended up with two possible answers.

It all depends on what we mean with _"What is the **most likely state sequence** given an observation sequence?"_

- it can be the state sequence that has the **highest conditional probability?** This is was we have done in [Q2](#q2) with `#BayesRule`. `#PosteriorDecoding`.
- it can be the state sequence that **makes the observation sequence the most likely to happen?** `#MLE`. `#Viterbi`. In this case, we compute the **Joint Probabilities** i.e. the probabilities for the **Intersection** [`State Sequence` `+` `Observation Sequence`].
- ??(can the MLE be on the conditional probability of observation given state? what differs with the upper joint event is the normalization)??

In other words, the answer could be maximizing two kind of probability

- either the **conditional probability**: (state **given** observation)
- or the **joint probability**: (state **and** observation).

> For this question, we will be looking for the **most likely sequence of hidden states** which could have generated a **given observation sequence**.

#### Maximum Likelihood Estimation (MLE)
We will to **pick the sequence (in [Q5](#q5) it was of size `1`) of hidden states that makes the observations the most likely to happen**.

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

For instance with the state sequence candidate [`low speed`, `high speed`, `low speed`]
- The **joint probability** is the product of all the probabilities listed on the figure below.
- P([`low speed`, `high speed`, `low speed`] `&&` [`right lane`, `left lane`, `right lane`]) = `0.02048`

| ![Derivation of the MLE for a particular observation sequence](docs/results_three.PNG "Derivation of the MLE for a particular observation sequence")  | 
|:--:| 
| *Derivation of the MLE for a particular observation sequence* |

#### Note

`0.1318394` `=` `0.01152` `+` `0.01536` `+` `0.000853` `+` `0.01536` `+` `0.054613` `+` `0.0068267` `+` `0.02048` `+` `0.0068267`

If you sum all the probabilities of the eight cases depicted the figure above, you do not end up to `1`, but to `0.1318394`. Why?
- Well, `0.1318394` represents the **probability for the observation sequence** [`low speed`, `high speed`, `low speed`] to happen.
	- P[`low speed`, `high speed`, `low speed`] = P[`obs`] = P[`obs` `&&` `state seq1`] + P[`obs` `&&` `state seq2`] + ... + P[`obs` `&&` `state seq8`]
	- ?? total probability is with conditional. What is it with intersections?
	- ?? This confirms the result of [Q5](#q5) ??
- What would sum to `1` is the sum over all possible 3-element observation sequences:
	- `1` = P[`LLL`] + P[`LLH`] + P[`LHL`] + P[`LHH`] + P[`HLL`] + P[`HLH`] + P[`HHL`] + P[`HHH`]

The presented approach could be used for **larger observation sequences**. But quickly, an issue appears:

| Size of the `observation` sequence | Number of probabilities to compute before applying `max()` (for **MLE**) |
| :---:        |     :---:      |
| `1`   | `2`     |
| `2`   | `4`     |
| `3`   | `8`     |
| `i`     | `2^i`       |
| `10`     | `1024`       |

### Dynamic Programming

Assume that after the second observation, the sub-sequence (`left lane`, `right lane`) is found to be more likely that the sub-sequence (`right lane`, `right lane`).
- Is it **worth carry on some investigation** in the branch (`left lane`, `right lane`)?
- Do not forget that the only goal is to **find the most likely sequence (and nothing else)!**
- Whatever you append to the sub-sequence (`left lane`, `right lane`), the resulting sequence will be less likely than appending the same to (`right lane`, `right lane`).

This example show the intuition of `Dynamic Programming`:

> Compute local blocks and consider only the most promising ones to build the next ones.

pass  # notebook

## Q5 - What if you are **not directly given the probability models**?

pass

### EM algorithm

# Further work
Ideas to go further:
- at a crossing, make prediction of the route of other vehicles
	- `route` in {`left`, `straight`, `right`} is the hidden variable

# Acknowledgement and references
I took some inspiration of this [video](https://www.youtube.com/watch?v=kqSzLo9fenk) by Luis Serrano.