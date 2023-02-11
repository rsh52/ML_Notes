# Statistical Learning with Applications in R

This running markdown document will serve to store notes related to the [EdX StanfordOnline STATSX0001 Statistical Learning](https://www.edx.org/course/statistical-learning) course with companion [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/) text book.

### Resources:

- [EdX StanfordOnline STATSX0001 Statistical Learning](https://www.edx.org/course/statistical-learning)
- [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/)
- [ISLR Tidy Models Labs](https://emilhvitfeldt.github.io/ISLR-tidymodels-labs/)
- [Introduction to Statistical Learning Using R Book Club](https://r4ds.github.io/bookclub-islr/)

### Notation:

- $p$ = number of vars to make predications/columns
- $n$ = number of distinct datapoints/rows in sample
- $x_{ij}$ = $j$th variable for the $i$th observation
- ${X}$ = matrix, bold capitals
- $y$ = prediction variable
- vectors denoted in lower case italicized font
- $∈$ = “is an element of”, “in”
- $ϵ$ = irreducible error
- $IR$ = “real numbers”

# Chapter 1 - Introduction

This is largely syllabus week style topics. Not much here in terms of notes.

### Required Packages

All datasets used in this course are found in the `ISLR2` package.

![Ch 1 Datasets](/images/statistcal_learning/ch1-datasets.png)

# Chapter 2 - Statistical Learning

## Ch 2.1

### Introduction to Regression Models

A vector of inputs can be thought of as $X = (X_1, X_2, ... X_n)$. $X$ indicates the _input_ variables or _predictors_. Also can be termed "independent variables" or "features".

A model can be written as:

$Y = f(X) + ϵ$

Where $ϵ$ is "error." There is always some amount of reducible error quantified by your function, and an _irreducible_ error quantified by $Var(ϵ)$ where "Var" is "variance." The reducible error is what we're tuning and trying to minimize. The average of $ϵ$ will always approximately equal zero.

In the above formula, $Y$ is your _response_ or _dependent_ variable. $f$ is a fixed, but unknown function and represents _systematic_ information that $X$ provides about $Y$.

The below example of the `Income` dataset shows $ϵ$ in the vertical lines, i.e. the average of them is zero:

![CH2 Income Dataset](/images/statistcal_learning/ch2-income-eda.png)

> :bulb: "In essence, statistical learning refers to a set of approaches for estimating
$f$."

### Why estimate $f$?

We estimate $f$ for **prediction** and **inference**.

Consider the equation:

$\hat{Y} = \hat{f}(X)$

When working with **prediction**, $\hat{f}$ is a "black box," i.e. the exact form of it is not our concern, but the accuracy of how it helps us predict $\hat{Y}$ is. 

Since $\hat{f}$ will always contain that "irreducible error," $ϵ$, we talked about earlier, then for a fixed $\hat{f}$ and $X$ we can show the average/expected value to be the squared difference between  the predicted and actual values with $Var(ϵ)$ representing variance in the irreducible error:

![Ch2 Error Averaging Eq](/images/statistcal_learning/ch2-eq23.png)

When working with **inference**, $\hat{f}$ is no longer a black box because the exact form of it needs to be known. There are times when both are needed in combination.

### How do we estimate $f$?

#### **Parametric Methods**

Parametric methods involve a 2-step approach:

1) Make an assumption about the functional form or shape of $f$: "linear" for example.
2) Once a model has been selected, **fit**/**train** the model. One common approach is using "ordinary least squares."

This is referred to as a "parametric" approach because estimating $f$ is done by using a set of parameters. A linear model has a defined function/shape and is much easier to estimate $f$ than using an entirely arbitrary/unknown function. The disadvantge is not typically matching the true unknown form of $f$.

#### **Non-Parametric Methods**

These methods make no explicit assumptions about the form of $f$, instead they look to estimate $f$ as smoothly as possible. The potential for accuracy with different shapes/forms is much greater since no assumptions are made about $f$. The downside is they require a very large number of observations. These functions can also lead to more _overfitting_ of the data.

### The Trade-Off Between Prediction Accuracy and Model Interpretability

- Parametric approaches like linear regression is relatively inflexible
- Non parametric approaches like splines are more flexible

In examples of inference, parametric approaches can be much more interpretable and understandable. Non-parametric ones can be much more difficult to understand how any one predictor is associated with the response.

### Supervised versus Unsupervised Learning

These are the two categories that summarize most statistical learning problems:

- **Supervised**: Fitting a model related to a response and predictors for either prediction or inference. Supervised learning involved labeled data witha response.
- **Unsupervised**: Observation of measurements, but no associated response. There are no labels, instead we are deriving clusters/patterns to understand relationships.

### Regression v. Classification Problems

- **Regression**: Problems with quantitative/numeric responses
- **Classification**: Problems with qualitative/categorical responses.