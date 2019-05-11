# HMMs: `stochastic` `generative` models

## Objectives
Among the different courses I took and the articles I read, some were describing HMMs as **"Generative models"**.

I wanted to learn more about this concept and present here some **notes and reflexions**.

## Machine learning techniques are used to model "relations"
Let's introduce two random variables:
- `X`: an **observation** (*outcome* or in particular contexts *features*).
- `Y`: a **label** (*class*).

Machine learning usually works in two stages:
- `1/2` Learning
	- This is about finding **some "relation"** (usually using a parametrized function) between the `X` and `Y` data from the available samples (*training dataset*).
- `2/2` Inference
	- Given a new observation `x`, use the derived model to deduce the **"related"** label `y` (sometimes called *target*).
		- Eventually, all methods are **predicting the conditional probability** p(`Y`|`X`).
	- But to perform that, **different probabilities are learnt** (i.e. different **"relations"** are modelled):
		- [`Y` **and** `X`], i.e. modelling **joint probabilities**.
		- [`Y` **given** `X`], i.e. modelling **conditional probabilities**.

Therefore, machine learning methods can be categorized into two families, depending on **which "relation" between `X` and `Y` they try to model**:
- **Generative models** (modeling **joint probabilities**).
	- What is modelled is the **actual distribution** of each class in `Y`.
- **Discriminative models** (modeling **conditional probabilities**)
	- What is modelled can be think of as the **decision boundary** between the available classes in `Y`.

## HMMs are generative models
**HMMs are generative models**, in that:
- They **model "generative relations"**.
- They are interested in the **joint distribution**.
- They can be used to **generate observations**.

### HMMs model "generative relations"
HMMs make the following assumption, called *generation assumption*:
- Some hidden samples from `Y` have been **generating (emitting) observations**.
- The observed data (`X`) are therefore assumed to be **truly sampled from the generative model**.

The two mentioned stages for machine learning are also present in HMMs:
- `1/2` Training
	- We have seen two cases of learning in an HMM, depending if the states are observable [Q1](https://github.com/chauvinSimon/hmm_for_autonomous_driving/blob/master/README.md#q1) or not [Q5](https://github.com/chauvinSimon/hmm_for_autonomous_driving/blob/master/README.md#q5).
	- In both cases:
		- A **generative model is assumed**: some functional form for **p(`Y`)** and **p(`X`|`Y`)**.
			- Note that our model explicitly describes the prior distribution on states **p(`Y`)**. Not just the conditional distribution of the observation given the current state.
			- In other words, since p(`X`|`Y`)\*p(`Y`)=p(`X`, `Y`), the HMM model actually gives a **joint distribution** on states and outputs.
			- Also remember that the final output of inference will be the **prediction of p(`Y`|`X`)**. This quantity is **not directly modelled** here but will be derived using Bayes' rule.
		- The parameters of this generative model are **fit so as to maximize the data likelihood**.
		- We are trying to find the parameters for multiple probabilities:
			- **Initial state probability**: `P[lane(t)]`. It can be seen as the prior p(`Y`).
			- **Transition probability**: `P[lane(t+1)` given `lane(t)]`. It can be seen as p(`X`|`Y`) with the sequence has siez `1`.
			- **Emission probability**: `P[speed(t)` given `lane(t)]`

- `2/2` Inference
	- When decoding in an HMM, the goal is to **infer a sequence of states** (`Y`) given the received observation (`X`).
	- HMMs ask the question: *Based on my generation assumptions, which category is most likely to generate this signal?*
	- During inference,
		- First, we listed all possible instances `y` of `Y`. 
		- Then, we compute the **joint distributions**.
		- Eventually the decision is made: select the `y` that maximizes the **joint probability distribution p(`x`, `y`)**.
	- Therefore, and similar to Naive Bayes classifier, HMMs try to predict **which class has generated the new observed example**.

### HMM are interested in the joint distribution
The learnt terms (**prior** p(`Y`) and **conditional** p(`X`|`Y`)) can be used to form two terms.
- `1/2` The **posterior probability** p(`Y`|`X`).
	- During inference, **p(`Y`|`X`) is derived from the learnt p(`X`|`Y`) and p(`Y`)** probabilities using **Bayes' rule**.
	- This is used to **make decision**: it returns the class that as the **maximum posterior probability** given the features.
- `2/2` The **joint distribution** p(`X`, `Y`)
	- One can also use it to **generate new data similar to existing data**.

### HMMs can generate observations

Given an HMM (*transition model*, *emission model* and *initial state distribution*), one can **generate an observation sequence** of length `T`.
- First, **sample** the state `state_1` from the **initial state distribution**.
- Then, loop until emitting the `T`-th observation:
	- **Emit** (generate) an observation using the emission model `obs_t`|`state_t`.
	- **Go to next state** according to the transition probability `state_t+1`|`state_t`.
- There is another option:
	- First, generate a **state sequence** (based on the transition model).
	- Then, **sample for each state sample** its observation (using the emission model).

Therefore, before seeing any observation, we **already have an idea on the `Y` distribution**.
- This is possible thanks to our **prior model `p(Y)`** (initial state probability), that represents some *a priori* (before seeing the observation) ideas we have about the `Y` distribution.

## Differences with Generative Learning Algorithms

What **"relation"** is learnt in generative algorithms?
- Remember that in machine learning, the final output is the **prediction of p(`Y`|`X`)**. This quantity is **directly modelled** by discriminative algorithms.
- Note that they **do not have any model for the prior p(`Y`)** and **do not need to model the distribution of the observed variables**.

What does really matter?
- A discriminative algorithm **does not care about how the data was generated**.
- It **simply categorizes** a given signal.
- In other words, it tries to learn **which features from the training examples** are most useful to **discriminate** between the **different possible classes**.

How is inference performed?
- The learnt model p(`Y`|`X`) is **directly used** to **"discriminate"** the value of the target variable `Y`, **given an observation `x`**.
- In other words, if I give you features `x`, you **directly** tell me the **most likely class `y`**.

Discriminative algorithms consider:
- Either a **conditional distribution** (e.g. **‌logistic regression**).
- Or **no distribution** (e.g. **SVM**, **perceptron**).

Therefore, sometimes two kinds of discriminative algorithms are **distinguished**:
- Discriminative **Models**.
- Or Discriminative **Classifiers** (without a model).

Dependancy on data.
- Discriminative Models make **fewer assumptions** on the distributions but **depend heavily on the quality of the training data**.
- The definition of a **generative model is more demanding** since it requires to understand the mechanism when **building generative models** to represent hypotheses.

### Example of discriminative models
- **Naïve Bayes**
- **Bayesian networks**
- **Gaussian Mixture Models**
- **Markov random fields**

### Example of discriminative models
- **‌Logistic regressions**
- **Conditional random fields (CRF)s**
- **Support vector machines (SVM)**
‌- **"Traditional" neural networks**
‌- **Nearest neighbour methods**

# Implications of Stochasticity?
On the [summary](https://github.com/chauvinSimon/hmm_for_autonomous_driving/blob/master/README.md#summary-2), I state that HMMs are **stochastic** since the transitions and the emissions are not deterministic.
Let's **investigate here the implications of the stochasticity**.

We simply **cannot know what value the state sequence took** when it generated a particular observation sequence.

To **estimate the true underlying sequence `path*`**, let's introduce a random variable and let's note it **`path`**.
- Since we have to **consider all possible state sequences**, the estimate of **`path*` must be described as a probability distribution**: p(`path`).
- We have to **maintain believes** about the hidden state.
	- Put it another way, we say that `path` is **simultaneously taking on all possible values**, some more likely than others.

Inference: 
- We need to **sum over all possible values of `path`** in order to **find the probability of an observation sequence having been generated by a model**.
- This operation is like **"integrating away" `path`** because we only care about the **probability** and not about the **value** of `path`.
- The quantity we obtain is called the `alpha` value (or *Forward probability*) and can be computed using the **Forward algorithm**.
- Sometimes, we are just interested in **an approximation to the `alpha` values**.
	- So, instead of **summing over all values of `path`**:
	- `1/2` We just pick **the single most likely value**.
	- `2/2` And we compute the probability of the observation sequence given **that one value of `path`**.

This short reflexion should remind you the **concepts of `one-path` vs `all-path`** and especially the **difference between `Baum-Welch` and `Viterbi-EM`**.

# References:
- [Paper](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) of Andrew Ng on Generative and Discriminative Models.
- [Blog post](https://medium.com/@mlengineer/generative-and-discriminative-models-af5637a66a3) from Prathap Manohar Joshi.  In particular, have a look at the list of questions about the **selection of the model depending on your problem**.