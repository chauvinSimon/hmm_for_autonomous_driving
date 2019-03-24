# hmm_for_autonomous_driving

| ![dd](docs/dd.PNG "dd")  | 
|:--:| 
| *dd* |

# Introduction
Disclaimer:
The goal of this repository is to get more familiar to the concept of **`Hidden Markov Models`** (**= HMM**) to through an **very simple** example.
The scope of ambition is therefore limited, only serving educational purposes.

Addressed topics:
- Problem setting: use a very basic example to introduce the **terminology**.
- implementation of the **Viterbi Algorithm** for training Hidden Markov Models.
- implementation of the **Baum–Welch Algorithm** to find the maximum likelihood estimate of the parameters of the HMM

# Problem motivation
(For left-hand-drive countries such as the UK, just invert the reasoning)

Your car is driving on a **2-lane highway**.
Imagine that you can **remotely monitor the velocity of the car (I communicate it to you)**, but you do have **no direct access to the lateral position** (`right lane` of `left lane`).

#### emission probability

If I am telling you that I am driving `slowly`, you **may deduce** that I am on the right lane.
For instance because I am just driving alone at a reasonable pace or because I am blocked by a slow vehicle.
But I could also drive fast on this `right lane` (have you ever been alone on a German highway??)

Similarly, if you get informed of an `high speed`, you could say that I am **more likely** to be driving on the left lane.
Probably overtaking another vehicle.
Nevertheless, this is **not always true**: think of the situation where you are waiting on the left lane behind a truck trying to overtake another truck

We get a first intuition: the variable `lane` seems to have an impact on the variable `speed`.
In  other words: *a priori* **you do not drive at the same pace depending if you are one the `left lane` or the `right lane`**.
But the relation is not deterministic (rather stochastic).
This will be modeled using **`emission probabilities`** in the following.

#### emission probability

You could have another intuition: human drivers usually **stay on their lanes**.
Hence if you are on `right lane` at time `t`, you might likely to still be on `right lane` at time `t+1`.
Again, this **does not always hold** and you can find **exception**, but here comes a second intuition:
*a priori* **the `lane` at time `t` is influenced by the `lane` at time `t-1`**.
The concept of **`transition probability`** will be used to model this second remark.

| ![The speed is the `observation` while the lane constitutes the `hidden state`. Some examples show that all `emissions` are possible](docs/terminology.PNG "The speed is the `observation` while the lane constitutes the `hidden state`. Some examples show that all `emissions` are possible")  | 
|:--:| 
| *The speed is the `observation` while the lane constitutes the `hidden state`. Some examples show that all `emissions` are possible* |


### objective
In other words, we want to **infer the lane** (`right` of `left`) of the car (= **hidden state**) based on a **series of speed measurements** (= **observation**)

### assumptions
To keep the problem as simple as possible:
- let's **discretize the speed** into `low speed` and `high speed`.
- time steps are discretized.
- lane transitions are ignored: either you are on `left lane` or you are on `right lane`

A word about the Markov Property:

# Problem formulation

Hidden state: discrete random variable `lane` € {`Right Lane`, `Left Lane`}
Observation: discrete random variable `speed` € {`Low Speed`, `High Speed`}
Emission probability: P[lane(t)->speed(t)]
Transition probability: P[lane(t)->lane(t+1)]
Marginal probability: P[lane(t)]

Bernoulli distribution

# Questions:
- Q1 - how to derive the three probability models?
- Q - if you receive a single observation, what are the probability for the car to be in each lane?
- Q2 - if you receive the sequence of observations [`high speed`, `high speed`, `low speed`, `high speed`], what is the most probable sequence of lane states?
- Q3 - ok, what if you are not directly given the probability models?

# Answers

## Q1
Let assume we are given some data: a series of observations and states.
Here are only a few for simplicity but you could imagine longer recordings.
You can think of it as some **samples** of the underlying **joint distributions**.

The idea is to **approximate the model parameters by counting the occurences**.

### transition probability

**Counting the number** of transitions gives a likelihoods we can used for our **transition probability** model.
For instance we made 15 transitions starting from `right lane` and among them 2 ended in `left lane`.
Hence the P[`right lane` -> `left lane`] = 
Since probabilities must sum to one (normalized), or just by counting for the other case (`right lane` is followed 10 times by a `right lane`), P[`right lane` -> `right lane`] = 

| ![deriving the transition probability model](docs/deriving_transition_model.PNG "deriving the transition probability model")  | 
|:--:| 
| *deriving the transition probability model* |

### emission probability

Using, we use counting to determine the **emission probability** model.
For instance how many times a `left lane` has caused a `high speed`?

In the **Bayesian framework**, it will correspond to the **likelihood**.

| ![deriving the emission probability model](docs/deriving_emission_model.PNG "deriving the emission probability model")  | 
|:--:| 
| *deriving the emission probability model* |

## marginal probability

At any time `t`, what is your guess on the distribution of the hidden state if no information is given (nothing about previous state and nothing about the observation)
Two options are available:
- either you count occurences
- or you use the transition model and the fact that probabilities sum to `1`:
	p[`left lane`, t] = 0.8 * p[`left lane`, t-1] + 0.2 * p[`right lane`, t-1]
	p[`right lane`, t] = 1 - p[`left lane`, t]
	
In the **Bayesian framework**, it will correspond to the **prior**.

## summary

| ![Hidden Markov Model with the `transition probability` model (up), and the `transition probability` model (below)](docs/hmm.PNG "Hidden Markov Model with the `transition probability` model (up), and the `transition probability` model (below)")  | 
|:--:| 
| *Hidden Markov Model with the `transition probability` model (up), and the `transition probability` model (below)* |

## Q2


#### Prior
Before any observation we know that `right lane` appears 2/3 of the time and `left lane` 1/3.
These probabilities would have been the answers if we were to ignore the observation.
This distribution (called `prior`) must be updated (via a term named `likelihood`) when considering the observation.
The `prior` is converted to a `posterior` as `observations` are considered.

#### Likelihood
based on data, we found that on the left lane it is more likely to drive fast. And slow on the right lane.
This led to emission probability, also understood as `likelihood`: **given a state**, what is the probability for each observation.
In the `posterior`, it is the **opposite** that apply: **given an observation**, what is the probability for each state?

#### Marginal
The Bayes Rule states that `Posterior` = `Normalized (prior * likelihood)`
The normalization is done using the `Marginal Probabilities`:
Under all possible hypothesese, how probable is each `lane`? Well we already know that from Q1.

#### Bayes Rule
Let's use Bayesian Statistics to recap.

The prior
P(`left lane` given `high speed`) = P(`high speed` given `left lane`) * P(`high speed`) / P(`left lane`)

## Summary

| ![derivation of the posterior probabilities for a single observation](docs/posteriors.PNG "derivation of the posterior probabilities for a single observation")  | 
|:--:| 
| *derivation of the posterior probabilities for a single observation* |

## Further work
Ideas to go further:
- route is the hidden variable

## Acknowledgement and references
I took some inspiration of this [video](https://www.youtube.com/watch?v=kqSzLo9fenk) by Luis Serrano.
