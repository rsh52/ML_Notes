# Statistical Learning with Applications in R

This running markdown document will serve to store notes related to the [EdX StanfordOnline STATSX0001 Statistical Learning](https://www.edx.org/course/statistical-learning) course with companion [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/) text book.

### Resources:

- [EdX StanfordOnline STATSX0001 Statistical Learning](https://www.edx.org/course/statistical-learning)
- [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/)
- [ISLR Tidy Models Labs](https://emilhvitfeldt.github.io/ISLR-tidymodels-labs/)
- [Introduction to Statistical Learning Using R Book Club](https://r4ds.github.io/bookclub-islr/)

### Notation

- $p$ = number of vars to make predications/columns
- $n$ = number of distinct datapoints/rows in sample
- $x_{ij}$ = $j$th variable for the $i$th observation
- ${X}$ = matrix, bold capitals
- $y$ = prediction variable
- vectors denoted in lower case italicized font
- $∈$ = “is an element of”, “in”
- $ϵ$ = irreducible error
- $IR$ = “real numbers”

# Ch 1 - Introduction

This is largely syllabus week style topics. Not much here in terms of notes.

### Required Packages

All datasets used in this course are found in the `ISLR2` package.

![Ch 1 Datasets](/images/statistical_learning/ch1-datasets.png)

# Chapter 2 - Statistical Learning

## Ch 2.1 - What is Statistical Learning?

### Ch 2.1.1 Introduction to Regression Models & Why esimate $f$?

A vector of inputs can be thought of as $X = (X_1, X_2, ... X_n)$. $X$ indicates the _input_ variables or _predictors_. Also can be termed "independent variables" or "features".

A model can be written as:

$Y = f(X) + ϵ$

Where $ϵ$ is "error." There is always some amount of reducible error quantified by your function, and an _irreducible_ error quantified by $Var(ϵ)$ where "Var" is "variance." The reducible error is what we're tuning and trying to minimize. The average of $ϵ$ will always approximately equal zero.

In the above formula, $Y$ is your _response_ or _dependent_ variable. $f$ is a fixed, but unknown function and represents _systematic_ information that $X$ provides about $Y$.

The below example of the `Income` dataset shows $ϵ$ in the vertical lines, i.e. the average of them is zero:

![CH2 Income Dataset](/images/statistical_learning/ch2-income-eda.png)

> :bulb: "In essence, statistical learning refers to a set of approaches for estimating
$f$."

We estimate $f$ for **prediction** and **inference**.

Consider the equation:

$\hat{Y} = \hat{f}(X)$

When working with **prediction**, $\hat{f}$ is a "black box," i.e. the exact form of it is not our concern, but the accuracy of how it helps us predict $\hat{Y}$ is. 

Since $\hat{f}$ will always contain that "irreducible error," $ϵ$, we talked about earlier, then for a fixed $\hat{f}$ and $X$ we can show the average/expected value to be the squared difference between  the predicted and actual values with $Var(ϵ)$ representing variance in the irreducible error:

![Ch2 Error Averaging Eq](/images/statistical_learning/ch2-eq23.png)

When working with **inference**, $\hat{f}$ is no longer a black box because the exact form of it needs to be known. There are times when both are needed in combination.

### Ch 2.1.2 How do we estimate $f$?

#### **Parametric Methods**

Parametric methods involve a 2-step approach:

1) Make an assumption about the functional form or shape of $f$: "linear" for example.
2) Once a model has been selected, **fit**/**train** the model. One common approach is using "ordinary least squares."

This is referred to as a "parametric" approach because estimating $f$ is done by using a set of parameters. A linear model has a defined function/shape and is much easier to estimate $f$ than using an entirely arbitrary/unknown function. The disadvantge is not typically matching the true unknown form of $f$.

#### **Non-Parametric Methods**

These methods make no explicit assumptions about the form of $f$, instead they look to estimate $f$ as smoothly as possible. The potential for accuracy with different shapes/forms is much greater since no assumptions are made about $f$. The downside is they require a very large number of observations. These functions can also lead to more _overfitting_ of the data.

### Ch 2.1.3 - The Trade-Off Between Prediction Accuracy and Model Interpretability

- Parametric approaches like linear regression is relatively inflexible
- Non parametric approaches like splines are more flexible

In examples of inference, parametric approaches can be much more interpretable and understandable. Non-parametric ones can be much more difficult to understand how any one predictor is associated with the response.

### Ch 2.1.4 - Supervised versus Unsupervised Learning

These are the two categories that summarize most statistical learning problems:

- **Supervised**: Fitting a model related to a response and predictors for either prediction or inference. Supervised learning involved labeled data witha response.
- **Unsupervised**: Observation of measurements, but no associated response. There are no labels, instead we are deriving clusters/patterns to understand relationships.

### Ch 2.1.5 - Regression v. Classification Problems

- **Regression**: Problems with quantitative/numeric responses
- **Classification**: Problems with qualitative/categorical responses.

## Ch 2.2 - Assessing Model Accuracy

### Ch 2.2.1 - Measuring Quality of Fit

In order to assess the extent to which a predicted response value for a given observation is close to the true value, regression relies on **mean squared error** as the most common quantification:

$MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{f}(x_i))^2$

Where $\hat{f}(x_i)$ is the prediction for the $i$th observation. MSE is small for predicted responses close to the true ones.

> :bulb: The **degress of freedom** of a given function is a quantity that summarizes the flexibility of a given curve. As flexibility increases, training MSE declines.

> :bulb: **Overfitting** of data occurs when there is a small training MSE but a high test MSE.

![Ch2 MSE Graphs](/images/statistical_learning/ch2-mse-graphs.png)

In the graphs above, the left hand side shows a true $f$ in black, a linear model in orange, and two smoothing splines in blue and green. On the right we see that the green spline has very low MSE, and high flexibility but poor test MSE. Blue has the best of both worlds, and makes sense seeing how it most visually matches $f$.

In Ch 5 _cross-validation_ will be discussed as an important method for estimating test MSE using training data.

### Ch 2.2.2 - The Bias-Variance Trade-Off

It is possible to show that the expected test MSE, for a given value $x_0$, can always be decomposed into the sum of 3 fundamental quantities:

1) **variance** of $\hat{f}(x_0)$
2) The squared **bias** of  $\hat{f}(x_0)$
3) The variance of the error, $ϵ$

The full equation being:

$E(y_0 - \hat{f}(x_0))^2 = Var(\hat{f}(x_0)) + [Bias(\hat{f}(x_0))]^2 + Var(ϵ)$

$E(y_0 - \hat{f}(x_0))^2$ = Expected Test MSE at $x_0$. In order to minimize test MSE, we need low variance and low bias.

> :bulb: **variance** = amount by which $\hat{f}$ would change if we estimated it using a different training set. High variance means small changes in the training data can result in large change in $\hat{f}$. Generally, more flexible modelling methods have higher variance.

> :bulb: **bias** = The error introduced by appoximating a real life model. Ex: linear regression assumes a linear relationship, but very few real life problems are ever linear. Therefore, linear regression will introduce some bias in the estimate of $f$. Generally, more flexible methods have less bias.

Determining how to balance bias, variance, and MSE is the **bias-variance trade-off**.

### Ch 2.2.3 - The Classification Setting

Up till now, we've focused on the regression setting, but many of these concepts carry over to classification with some modifying. One key change is $y_i$ is now qualitative instead of quantitative.

Here, the most common approach for quanitfying accuracy of our estimate $\hat{f}$, is the training _error rate_, i.e. the proportion of mistakes made if we apply the estimated $\hat{f}$  to the training observations:

$\frac{1}{n} \sum_{i=1}^n I(y_i \neq \hat{y}_i)$

Where:

- $\hat{y}_i$ = predicted class label for the $i$th observation using $\hat{f}$
- $I(y_i \neq \hat{y}_i)$ = "_indicator variable_" equal to 1 if $y_i \neq \hat{y}_i$ and 0 if $y_i = \hat{y}_i$

Unlike the training error rate above, the test error rate for a given set of test observations of the form $(x_0, y_0)$ is given by:

$Ave(I(y_i \neq \hat{y}_i))$

#### **The Bayes Classifier**

$Pr(Y = j|X = x_0)$

The equation above is used to evidence a simple classifier that assigns each observation to the most likely class, given its predictor values. This is a **conditional probability**, i.e. the probability that $Y = j$ given observed predictor $X_0$. This is the **Bayes Classifier** :bulb:.

In a simple example with two possible repsonsible values, the Bayes Classifier predicts class 1 if $Pr(Y = 1|X = x_0)$ > 0.5, and class 2 otherwise. In the image below, this is exemplified by the purple line, known as the **Bayes decision boundary** (i.e. the points at which the probability of either class is exactly 50%):

![Ch2 Bayes Decision Boundary](/images/statistical_learning/ch2-bayes-decision-boundary.png)

Observations that fall on either side of the decision boundary are assigned to those classes.

> :bulb: The _Bayes error rate_ is the lowest possible test error rate produced by the Bayes Classifier. And is given by:

![Ch2 Bayes Error Rate](/images/statistical_learning/ch2-bayes-error-rate.png)

#### **K-Nearest Neighbors**

K-Nearest neighbors (KNN) is one method for classifying obervations based on highest estimated probabilities. Given a positive integer, $K$, and a test observation $x_0$, KNN first looks at the closest values to the observation from the training data ($N_0$) and then estimates the conditional probability for class $j$ as a fraction of the points in $N_0$ whose repsonse values equal $j$.

![Ch2 KNN](/images/statistical_learning/ch2-knn.png)

In the example below with two known classes (blue and orange) and an unknown class (black cross), a $K$ of 3 tells us to look at the first 3 neighbors to the point in question and classify. 

![Ch2 KNN example graph](/images/statistical_learning/ch2-knn-ex-graph.png)

Since there are 2 blue values and 1 orange in the given $K$, the estimated probabilities are 2/3 blue and 1/3 orange for the unknown value.

While the Bayes classifier is an impossible standard to reach, the KNN decision boundary can get surprisingly close to it.

Low $K$'s are very flexible, indicating low bias but high variance. High $K$'s make for less flexibility, and makes for high bias and low variance.

# Chapter 3 - Linear Regression

## Ch 3.1 - Simple Linear Regression

The simple linear regression relationship mathematically is:

$Y \approx \beta_0 + \beta_1 X$

You can say "we are _regressing_ Y on X." Example: We can regress "sales" onto "TV" bu fitting the model:

$sales \approx \beta_0 + \beta_1 \times TV$

- $\beta_0$ = intercept
- $\beta_1$ = slope

Once training data is used to provide estimates for $\hat\beta_0$ and $\hat\beta_1$, future sales can be predicted using a particular value of TV using:

$\hat{y} = \hat\beta_0 +\hat\beta_1x$

- $\hat{y}$ = prediction of Y on the basis of X = $x$
- The hat symbol denotes an estimated value for an unknown param or coefficient

## Ch 3.1.1 Estimating the Coefficients

Using the estimation function above, we can use _least squares_ to try and make the best linear approximation that fits the values.

The **residual** is the difference between an observed value and the predicted value by the linear model (the gray lines in the plot below):

![Ch3 Least Squares](/images/statistical_learning/ch3-least-squares.png)

Each residual at $i$th observation is represented as $e_i = y_i - \hat{y}_i$.

The **residual sum of squares (RSS)** is defined as:

$RSS = e_1^2 + e_2^2 + ... e_n^2$

Or:

$RSS = (y_1-\hat\beta_0-\hat\beta_1x_1)^2 + ...(y_1-\hat\beta_0-\hat\beta_1x_n)^2$

And finally, the **least squares coefficient estimates** the values for the coefficients that minimize the RSS are:

![Ch3 RSS Minimizers Eq](/images/statistical_learning/ch3-rss-minimizers.png)

## Ch 3.1.2 Assess the Accuracy of Coefficient Estimates

In the equation below:

$Y \approx \beta_0 + \beta_1 X + \epsilon$

- $\beta_0$ = intercept - expected value of $Y$ when $X$ is 0.
- $\beta_1$ = slope - average increase in $Y$ associated with a one-unit increase in $X$
- $\epsilon$ = error term for what we miss with this model

Other definitions:

- **Population Regression Line**: best linear approximation to the true relationship between $X$ and $Y$.
- **Least Squares Line**: characterization of the least squares regression coefficient estimates

It is reasonable to assume that for a given population where $\mu$ is the population mean and $\hat\mu$ is the sample mean, that when using many observations they will be equivalent. However, a single estimate of $\hat\mu$ will be off by some amount. To calculate that amount, we compute the **standard error** ($SE$):

$Var(\hat\mu) = SE(\hat\mu)^2 = \frac{\sigma^2}{n}$

Where $\sigma$ is the **standard deviation** of each of the realizations $y_i$ of $Y$. The standard error tells us the average amount that the estimate of $\hat\mu$ differs from the actual value of $\mu$. Thanks to $n$, we know that the more observations we have, the smaller the standard error will be! :bulb:

Similarly, $SE$ can be computed for the intercept and slope values as well:

![Ch3 SE Intercept and Slope](/images/statistical_learning/ch3-se-intercept-slope.png)

Standard Error can also be used to calculate **confidence intervals**. Meaning, with a confidence interval of 95% if we take repeated samples and construct the confidence interval for each sample, 95% of the intervals will contain the true unknown value of the parameter.

In linear regression, the slope and intercept variable confidence intervals take the form:

$\hat\beta_1 \plusmn 2 \cdot SE(\hat\beta_1)$

$\hat\beta_0 \plusmn 2 \cdot SE(\hat\beta_0)$

Where the _dot product_ is the sum of the corresponding products of the described vector. The $\plusmn$ represents the two way interval range.

This can be very useful for quickly determining if a variable has no relationship with the **null hypothesis** ($H_0$).

If $\beta_1$ = 0 then the model eliminates $X$ as a relational variable. However, determining how far from the true value is acceptable is determined by computing the **t-statistic**, responsible for measuring the number of standard deviations that $\beta_1$ is from 0.

:bulb: The **p-value** indicates the likelihood of a substantial association between the predictor and the response due to chance. A small p-value is indicative of an association between the predictor and the response. When one exists, it is reasont to _reject the the null hypothesis_.

## Ch 3.1.3 Assessing the Accuracy of the Model

The quality of a linear regression fit is typically assessed by two related qualities:

- $RSE$ (Residual Standard Error): An estimate of the standard deviation of $\epsilon$, i.e. the average amount that the response will deviate from the true regression line
- $R^2$ statistic: The proportion of variance in $Y$ that can be explained by $X$, i.e. a measure of the linear relationship between $X$ and $Y$

RSE:

$RSE = \sqrt{\frac{1}{n-2}RSS} = \sqrt{\frac{1}{n-2}\sum(y_i - \hat{y_i})^2}$

RSE is considered a measure of a the _lack of fit_ of the model to the data. Small RSE means our model fits the data very well.

$R^2$:

$R^2 = \frac{RSS - RSS}{TSS} = 1 - \frac{RSS}{TSS}$ (Where TSS is the _total sum of squares_)

An $R^2$ statistic close to 1 indicates a large proportion of the variability in the response is explained by the regression. A value close to 0, meaning no explanation, can be caused by a wrong model or high variance, or both.

In the example of sales in relation to television ads, an $R^2$ value of 0.612 can be read as variance in sales reduced by 61%. It is a measurement of the correlation between two variables. The $p$ value and $t$ statistic, however, measureevidence that there is a non zero association.

## Ch 3.2 - Multiple Linear Regression

The multiple linear regression model seeks to combine multiple predictors instead of creating multiple single linear regression models on the predictors separately. Instead, each predictor gets its own slope variable. Therefore, the model takes on the form:

$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \epsilon$

### Ch 3.2.1 - Estimating the Regression Coefficients

Just like in simple linear regression, all $\beta$ values are unknown and must be estimated. This is done almost identically to the RSS method in simple regression:

$RSS = \sum(y_i-\hat{y}_i)^2 = \sum(y_i - \hat\beta_0 - \hat\beta_1x_{i1} - ...\hat\beta_px_{ip})^2$

Whereas before we saw a simple line, with, say, 2 predictors ($p$) we see a plane:

![Ch3 Multi RSS Plane](/images/statistical_learning/ch3-multi-plane.png)

When looking at a multiple linear regression model and assessing the results, evaluating any single predictor is done while holding the other predictors constant. This can result in differences that multiple individual single linear models would relate.

> An absurd example: simple linear regressions done separately might show a positive correlation between shark bites and ice cream sales, but multiple linear regression would show this is not the case with the combined predictor of high temperatures.

### Ch 3.2.2 - Some Important Questions

Multiple linear regression necessitates the following:

1. Is at least one of the predictors $X1,X2, . . . ,Xp$ useful in predicting
the response?
2. Do all the predictors help to explain $Y$ , or is only a subset of the
predictors useful?
3. How well does the model fit the data?
4. Given a set of predictor values, what response value should we predict,
and how accurate is our prediction?

Hypothesis testing is performed by computing the **F-statistic**. A F-statistic close to 1 indiciates no relationship between the response and predictors, but if greater than 1 then indicative that there is.