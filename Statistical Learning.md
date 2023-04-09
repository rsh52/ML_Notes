# Statistical Learning with Applications in R

This running markdown document will serve to store notes related to the [EdX StanfordOnline STATSX0001 Statistical Learning](https://www.edx.org/course/statistical-learning) course with companion [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/) text book.

### Resources:

-   [EdX StanfordOnline STATSX0001 Statistical Learning](https://www.edx.org/course/statistical-learning)
-   [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/)
-   [ISLR Tidy Models Labs](https://emilhvitfeldt.github.io/ISLR-tidymodels-labs/)
-   [Introduction to Statistical Learning Using R Book Club](https://r4ds.github.io/bookclub-islr/)

### Notation

-   $p$ = number of vars to make predications/columns
-   $n$ = number of distinct datapoints/rows in sample
-   $x_{ij}$ = $j$th variable for the $i$th observation
-   ${X}$ = matrix, bold capitals
-   $y$ = prediction variable
-   vectors denoted in lower case italicized font
-   $∈$ = "is an element of", "in"
-   $ϵ$ = irreducible error
-   $IR$ = "real numbers"

# Ch 1 - Introduction

This is largely syllabus week style topics. Not much here in terms of notes.

### Required Packages

All datasets used in this course are found in the `ISLR2` package.

![Ch 1 Datasets](/images/statistical_learning/ch1-datasets.png)

# Chapter 2 - Statistical Learning

## Ch 2.1 - What is Statistical Learning?

### Ch 2.1.1 Introduction to Regression Models & Why esimate $f$?

A vector of inputs can be thought of as $X = (X_1, X_2, ... X_n)$. $X$ indicates the *input* variables or *predictors*. Also can be termed "independent variables" or "features".

A model can be written as:

$Y = f(X) + ϵ$

Where $ϵ$ is "error." There is always some amount of reducible error quantified by your function, and an *irreducible* error quantified by $Var(ϵ)$ where "Var" is "variance." The reducible error is what we're tuning and trying to minimize. The average of $ϵ$ will always approximately equal zero.

In the above formula, $Y$ is your *response* or *dependent* variable. $f$ is a fixed, but unknown function and represents *systematic* information that $X$ provides about $Y$.

The below example of the `Income` dataset shows $ϵ$ in the vertical lines, i.e. the average of them is zero:

![CH2 Income Dataset](/images/statistical_learning/ch2-income-eda.png)

> :bulb: "In essence, statistical learning refers to a set of approaches for estimating $f$."

We estimate $f$ for **prediction** and **inference**.

Consider the equation:

$\hat{Y} = \hat{f}(X)$

When working with **prediction**, $\hat{f}$ is a "black box," i.e. the exact form of it is not our concern, but the accuracy of how it helps us predict $\hat{Y}$ is.

Since $\hat{f}$ will always contain that "irreducible error," $ϵ$, we talked about earlier, then for a fixed $\hat{f}$ and $X$ we can show the average/expected value to be the squared difference between the predicted and actual values with $Var(ϵ)$ representing variance in the irreducible error:

![Ch2 Error Averaging Eq](/images/statistical_learning/ch2-eq23.png)

When working with **inference**, $\hat{f}$ is no longer a black box because the exact form of it needs to be known. There are times when both are needed in combination.

### Ch 2.1.2 How do we estimate $f$?

#### **Parametric Methods**

Parametric methods involve a 2-step approach:

1)  Make an assumption about the functional form or shape of $f$: "linear" for example.
2)  Once a model has been selected, **fit**/**train** the model. One common approach is using "ordinary least squares."

This is referred to as a "parametric" approach because estimating $f$ is done by using a set of parameters. A linear model has a defined function/shape and is much easier to estimate $f$ than using an entirely arbitrary/unknown function. The disadvantge is not typically matching the true unknown form of $f$.

#### **Non-Parametric Methods**

These methods make no explicit assumptions about the form of $f$, instead they look to estimate $f$ as smoothly as possible. The potential for accuracy with different shapes/forms is much greater since no assumptions are made about $f$. The downside is they require a very large number of observations. These functions can also lead to more *overfitting* of the data.

### Ch 2.1.3 - The Trade-Off Between Prediction Accuracy and Model Interpretability

-   Parametric approaches like linear regression is relatively inflexible
-   Non parametric approaches like splines are more flexible

In examples of inference, parametric approaches can be much more interpretable and understandable. Non-parametric ones can be much more difficult to understand how any one predictor is associated with the response.

### Ch 2.1.4 - Supervised versus Unsupervised Learning

These are the two categories that summarize most statistical learning problems:

-   **Supervised**: Fitting a model related to a response and predictors for either prediction or inference. Supervised learning involved labeled data witha response.
-   **Unsupervised**: Observation of measurements, but no associated response. There are no labels, instead we are deriving clusters/patterns to understand relationships.

### Ch 2.1.5 - Regression v. Classification Problems

-   **Regression**: Problems with quantitative/numeric responses
-   **Classification**: Problems with qualitative/categorical responses.

## Ch 2.2 - Assessing Model Accuracy

### Ch 2.2.1 - Measuring Quality of Fit

In order to assess the extent to which a predicted response value for a given observation is close to the true value, regression relies on **mean squared error** as the most common quantification:

$MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{f}(x_i))^2$

Where $\hat{f}(x_i)$ is the prediction for the $i$th observation. MSE is small for predicted responses close to the true ones.

> :bulb: The **degress of freedom** of a given function is a quantity that summarizes the flexibility of a given curve. As flexibility increases, training MSE declines.

> :bulb: **Overfitting** of data occurs when there is a small training MSE but a high test MSE.

![Ch2 MSE Graphs](/images/statistical_learning/ch2-mse-graphs.png)

In the graphs above, the left hand side shows a true $f$ in black, a linear model in orange, and two smoothing splines in blue and green. On the right we see that the green spline has very low MSE, and high flexibility but poor test MSE. Blue has the best of both worlds, and makes sense seeing how it most visually matches $f$.

In Ch 5 *cross-validation* will be discussed as an important method for estimating test MSE using training data.

### Ch 2.2.2 - The Bias-Variance Trade-Off

It is possible to show that the expected test MSE, for a given value $x_0$, can always be decomposed into the sum of 3 fundamental quantities:

1)  **variance** of $\hat{f}(x_0)$
2)  The squared **bias** of $\hat{f}(x_0)$
3)  The variance of the error, $ϵ$

The full equation being:

$E(y_0 - \hat{f}(x_0))^2 = Var(\hat{f}(x_0)) + [Bias(\hat{f}(x_0))]^2 + Var(ϵ)$

$E(y_0 - \hat{f}(x_0))^2$ = Expected Test MSE at $x_0$. In order to minimize test MSE, we need low variance and low bias.

> :bulb: **variance** = amount by which $\hat{f}$ would change if we estimated it using a different training set. High variance means small changes in the training data can result in large change in $\hat{f}$. Generally, more flexible modelling methods have higher variance.

> :bulb: **bias** = The error introduced by appoximating a real life model. Ex: linear regression assumes a linear relationship, but very few real life problems are ever linear. Therefore, linear regression will introduce some bias in the estimate of $f$. Generally, more flexible methods have less bias.

Determining how to balance bias, variance, and MSE is the **bias-variance trade-off**.

### Ch 2.2.3 - The Classification Setting

Up till now, we've focused on the regression setting, but many of these concepts carry over to classification with some modifying. One key change is $y_i$ is now qualitative instead of quantitative.

Here, the most common approach for quanitfying accuracy of our estimate $\hat{f}$, is the training *error rate*, i.e. the proportion of mistakes made if we apply the estimated $\hat{f}$ to the training observations:

$\frac{1}{n} \sum_{i=1}^n I(y_i \neq \hat{y}_i)$

Where:

-   $\hat{y}_i$ = predicted class label for the $i$th observation using $\hat{f}$
-   $I(y_i \neq \hat{y}_i)$ = "*indicator variable*" equal to 1 if $y_i \neq \hat{y}_i$ and 0 if $y_i = \hat{y}_i$

Unlike the training error rate above, the test error rate for a given set of test observations of the form $(x_0, y_0)$ is given by:

$Ave(I(y_i \neq \hat{y}_i))$

#### **The Bayes Classifier**

$Pr(Y = j|X = x_0)$

The equation above is used to evidence a simple classifier that assigns each observation to the most likely class, given its predictor values. This is a **conditional probability**, i.e. the probability that $Y = j$ given observed predictor $X_0$. This is the **Bayes Classifier** :bulb:.

In a simple example with two possible repsonsible values, the Bayes Classifier predicts class 1 if $Pr(Y = 1|X = x_0)$ \> 0.5, and class 2 otherwise. In the image below, this is exemplified by the purple line, known as the **Bayes decision boundary** (i.e. the points at which the probability of either class is exactly 50%):

![Ch2 Bayes Decision Boundary](/images/statistical_learning/ch2-bayes-decision-boundary.png)

Observations that fall on either side of the decision boundary are assigned to those classes.

> :bulb: The *Bayes error rate* is the lowest possible test error rate produced by the Bayes Classifier. And is given by:

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

You can say "we are *regressing* Y on X." Example: We can regress "sales" onto "TV" bu fitting the model:

$sales \approx \beta_0 + \beta_1 \times TV$

-   $\beta_0$ = intercept
-   $\beta_1$ = slope

Once training data is used to provide estimates for $\hat\beta_0$ and $\hat\beta_1$, future sales can be predicted using a particular value of TV using:

$\hat{y} = \hat\beta_0 +\hat\beta_1x$

-   $\hat{y}$ = prediction of Y on the basis of X = $x$
-   The hat symbol denotes an estimated value for an unknown param or coefficient

## Ch 3.1.1 Estimating the Coefficients

Using the estimation function above, we can use *least squares* to try and make the best linear approximation that fits the values.

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

-   $\beta_0$ = intercept - expected value of $Y$ when $X$ is 0.
-   $\beta_1$ = slope - average increase in $Y$ associated with a one-unit increase in $X$
-   $\epsilon$ = error term for what we miss with this model

Other definitions:

-   **Population Regression Line**: best linear approximation to the true relationship between $X$ and $Y$.
-   **Least Squares Line**: characterization of the least squares regression coefficient estimates

It is reasonable to assume that for a given population where $\mu$ is the population mean and $\hat\mu$ is the sample mean, that when using many observations they will be equivalent. However, a single estimate of $\hat\mu$ will be off by some amount. To calculate that amount, we compute the **standard error** ($SE$):

$Var(\hat\mu) = SE(\hat\mu)^2 = \frac{\sigma^2}{n}$

Where $\sigma$ is the **standard deviation** of each of the realizations $y_i$ of $Y$. The standard error tells us the average amount that the estimate of $\hat\mu$ differs from the actual value of $\mu$. Thanks to $n$, we know that the more observations we have, the smaller the standard error will be! :bulb:

Similarly, $SE$ can be computed for the intercept and slope values as well:

![Ch3 SE Intercept and Slope](/images/statistical_learning/ch3-se-intercept-slope.png)

Standard Error can also be used to calculate **confidence intervals**. Meaning, with a confidence interval of 95% if we take repeated samples and construct the confidence interval for each sample, 95% of the intervals will contain the true unknown value of the parameter.

In linear regression, the slope and intercept variable confidence intervals take the form:

$\hat\beta_1 \plusmn 2 \cdot SE(\hat\beta_1)$

$\hat\beta_0 \plusmn 2 \cdot SE(\hat\beta_0)$

Where the *dot product* is the sum of the corresponding products of the described vector. The $\plusmn$ represents the two way interval range.

This can be very useful for quickly determining if a variable has no relationship with the **null hypothesis** ($H_0$).

If $\beta_1$ = 0 then the model eliminates $X$ as a relational variable. However, determining how far from the true value is acceptable is determined by computing the **t-statistic**, responsible for measuring the number of standard deviations that $\beta_1$ is from 0.

:bulb: The **p-value** indicates the likelihood of a substantial association between the predictor and the response due to chance. A small p-value is indicative of an association between the predictor and the response. When one exists, it is reasont to *reject the the null hypothesis*.

## Ch 3.1.3 Assessing the Accuracy of the Model

The quality of a linear regression fit is typically assessed by two related qualities:

-   $RSE$ (Residual Standard Error): An estimate of the standard deviation of $\epsilon$, i.e. the average amount that the response will deviate from the true regression line
-   $R^2$ statistic: The proportion of variance in $Y$ that can be explained by $X$, i.e. a measure of the linear relationship between $X$ and $Y$

RSE:

$RSE = \sqrt{\frac{1}{n-2}RSS} = \sqrt{\frac{1}{n-2}\sum(y_i - \hat{y_i})^2}$

RSE is considered a measure of a the *lack of fit* of the model to the data. Small RSE means our model fits the data very well.

$R^2$:

$R^2 = \frac{RSS - RSS}{TSS} = 1 - \frac{RSS}{TSS}$ (Where TSS is the *total sum of squares*)

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

1.  **Is at least one of the predictors** $X1,X2, . . . ,Xp$ useful in predicting the response?

Hypothesis testing is performed by computing the **F-statistic**. A F-statistic close to 1 indiciates no relationship between the response and predictors, but if greater than 1 then indicative that there is. The importance of how far the F-statistic is from 1 is dependent on context with $n$ and $p$. For a large $n$, F-statistics slightly larger than 1 may still provide evidence against $H_0$, but a larger one is needed if $n$ is small. Any statistical software will calculate the F-statistic given the value of $n$ and $p$, and the $p$-value can determine whether or not to reject $H_0$.

Computing the F-statistic:

$F = \frac{(TSS-RSS/p)}{RSS/(n-p-1)}$

2.  **Do all the predictors help to explain** $Y$ , or is only a subset of the predictors useful?

While its possible that all predictors are associtated with the response, its more common that only a subset truly are. Here we need to use *variable selection*. Ex: If $p=2$, we can consider 4 models: 1) no variables, 2) $X_1$ only, 3) $X_2$ only, 4) both $X_1$ and $X_2$. With large $p$, this becomes unreasonable. Instead one of the following are used:

-   Forward Selection
-   Backward Selection
-   Mixed Selection

3.  **How well does the model fit the data?**

Two of the most common numerical measure of model fit are $RSS$ and $R^2$. Recall a $R^2$ close to 1 indicates a large amount of variance in the response can be explained by the model.

Adding more variables to a model will always increase $R^2$, even if those variables are only weakly associated. If adding a variable only increases the value by a small amount, it may be indicative that the variable can be dropped.

For multiple linear regression, $RSE$ is computed as:

$RSE = \sqrt{\frac{1}{n-p-i}RSS}$

Models with more variables have a higher RSE.

4.  **Given a set of predictor values, what response value should we predict, and how accurate is our prediction?**

There are three sets of uncertainty in predictions:

-   Uncertainty in the estimates of $β_i$
-   Model bias
-   Irreducible error, ϵ

## Ch 3.3 Other Considerations in the Regression Model

### Ch 3.3.1 Qualitative Predictors

This section goes over **_dummy variables_**, which get used for describing _qualitative_ predictors (i.e. non-numeric). For a given set of predictors with $k$ levels, $k-1$ dummy variables are introduced with values of 1 or 0. So, for example, if you want to predict one of the cardinal directions (North, South, East, West), you would have:

$x_1$: 1 if a person is from the South, and 0 if not
$x_2$: 1 if a person is from the North, and 0 if not
$x_3$: 1 if a person is from the East, and 0 if not

And leave West alone since it would be double counting to include as the other three predict it based on its absence. **There will always be one fewer dummy variable than the number of levels**.

In the example results below:

![Ch 3 Multi Pred Balance](/images/statistical_learning/ch3-multi-predicted-balance.png)

With African American being the third predictor, the _predicted balance_ for AA in this data set is the Intercept value (531) while Asian is 531 - 18.69 and Caucasian is 531 - 12.50.

### Ch 3.3.2 Extensions of the Linear Model

Linear models are assumed to be _additive_ and _linear_. Additive meaning the association between a predictor and the response does not depend on the values of the othe predictors.

A **_synergy effect_** (in marketing) or an **_interaction effect_** (in statistics) describes when two predictors offer more of a response than they do separately. Example: if Radio and TV have an impact on sales, but combined their impact is even greater.

**The hierarchy principle**: if we include an interaction in a model, we should also include the _main effects_, even if the p-values associated with their coefficients are not significant.

Polynomial expressions can also be included in linear models to express non-linear relationships:

$Y = \beta_0 + \beta_1X + \beta_2X^2 + ... \beta_nX^n + \epsilon_i$

### Ch 3.3.3 Potential Problems

1) Non-linearity of the response-predictor relationships
    - Residual plots can be used to map out residuals. If a pattern exists, it could be indicative the model is poor.
2) Correlation of error terms
    - Leads to an unwarranted sense of confidence in our model. Residuals can again be plotted, but in the example of time series if it can be seen that adjacent residuals are close in value, it can indiciate a positive correlation. This is considered **_tracking_**.
3) Non-constant variance of error terms
    - Non-constance in error terms can be identified with **_heteroscedasticity_** which is shown in a "funnel shape" in the residuals plot (i.e. the magnitude of the residuals increases with the fitted values).
4) Outliers
    - Residual plots can be used to identify outliers. Outliers can greatly affect RSE and $R^2$ values. Outliers come up with when the response is unusual given the predictor.
5) High-leverage points
    - Conversely, high-leverage occurs when there is an unusual predictor i.e. a predictor value is vastly different from the others. It is very important to identify these value because they can invalidate the entire fit of the model. Computing an observation's leverage is done with the equation defined by the **_leverage statistic_**, which varies for different models.
6) Collinearity
    - Occurrs when two or more predictors are closely related. This can create difficulty in determining the effect a predictor has. There can also exist **_multicollinearity_** when 3+ variables have high correlation even if they do not as pairs. A correlation matrix can help visually determine collineartiy, but not always, and you may want to employ the **_variance inflation factor_** (VIF).

# Chapter 4 - Classification

## Ch 4.1 - An Overview of Classification

General notes on classification, terms like _categorization_ and introduction to the example of the credit card balance versus default dataset

## Ch 4.2 - Why Not Linear Regression?

For a given example where a set of predictors may have 3+ classifiers, linear regression assumes the difference between the classifiers is of the same magnitude. In reality, there is no reason to assume this. Linear regression is better suited for binary classification (0/1) and use of dummy variables.

Two reasons not to perform classification using a regression method:

1) A regression method cannot accommodate a qualitative response with more than two classes
2) a regression method will not provide meaningful estimates of $Pr(Y|X)$, even with just 2 classes.

Using the credit card default dataset, probabilities of defaulting can wind up being _negative_ using linear regression:

![Ch4 Linear Regression Negative Probability](/images/statistical_learning/ch4-linear-reg-negative.png)

## Ch 4.3 - Logistic Regression

Rather than modeling repsonse $Y$ directly, logistic regression models the _probability_ that $Y$ belongs to a particular category.

### Ch 4.3.1 - The Logistic Model

Since probabilities must always fall between 0 and 1, and not negative or greater than 1 as shown above, the logistic function is used to ensure sensible outputs:

$p(X) = \frac{e^{\beta_0 + \beta_1X}}{1+e^{\beta_0 + \beta_1X}}$

Fitting the model is done using a method called _maximum likelihood_. The logistic function always produces an S-shaped curve as shown in the image above.

Taking the formula above and manipulating it we can produces the _odds_, which can take on any value greater than or equal to 0:

$\frac{p(X)}{1-p(X)} = e^{\beta_0 + \beta_1X}$

Taking the _log_ of both sides removes $e$ and gives you the _log odds_ or _logit_.

### Ch 4.3.2 - Estimating the Regression Coefficients

This section covers _maximum likelihood_ which can be though of like least squares for model fitting in linear regression.

Seek esimates for $\beta_0$ and $\beta_1$ such that the predicted probability $\hat{p}(x_i)$ is close to the true value of the result.

Luckily maximum likelihood fitting can be done easily in R.

### Ch 4.3.3 - Making Predictions

Basically you make your model, you plug in your values, you get your prediction :sparkles:

### Ch 4.3.4 - Multiple Logistic Regression

Similar to the multiple linear regression model, the multiple logistic regression model can be written as:

$p(X) = \frac{e^{\beta_0 + \beta_1X+...\beta_pX_p}}{1 + e^{\beta_0 + \beta_1X+...\beta_pX_p}}$

Maximum likelihood is still used to estimate all of the coefficients.

This chapter shows a good example of how multiple logisitc regression can determine opposite end results than single for the same dataset.

### Ch 4.3.5 - Multinomial Logistic Regression

When the repsonse variable is non-singular, we refer to $K > 1$. In the student default example $K=2$ (a student defaults or does not). But in the example of classifying medical condition in an emergency room (stroke, drug oversoe, seizure), $K=3$. Acommodating this is known as _multinomial logistic regression_.

Selection of the baseline is unimportant, but interpretation of the coefficients is very important since it is tied to the choice of baselines.

An alternative to this approach is to use _softmax_ coding, where rather than selecting a baseline class all $K$ classes are treated symmetrically.

## Ch 4.4 - Generative Models for Classification

In these approaches, the distribution of the predictors is modelled separately in each response class. Baye's theroem is then used to flip the estimates. When the distribution of the predictors within each class is assumed to be normal, the models becomes similar to logistic regression. This is used in the following cases:

- Substantial separation between the two classes, which leads to model instability in logistic regression.
- If the distribution of the predictors is approximately normal in each of the classes and the sample size is small, these methods will be more accurate.
- These methods can be extended to cases with more than two response classes.

This section describes the following 3 classifiers that use different estimates for approximating the Bayes classifier:

- Linear discriminant Analysis
- Quadratic Discriminant Analysis
- Naive Bayes

### Ch 4.4.1 - Linear Discriminant Analysis for $p$ = 1

Assume we only have 1 predictor, $p = 1$, and we would like to obtain an estimate for $f_k(x)$. We will classify an observation to the class for which $p_k(x)$ is the greatest.

$f_k(x)$ is assumed to be nomral/Gaussian. In this setting the normal density looks like:

$f_k(x) = \frac{1}{\sqrt{2\pi}\sigma_k}exp(-\frac{1}{2\pi\sigma_k^2}(x-\mu_k)^2)$

- $\mu_k$ = mean parameter for $k$th class
- $\sigma_k^2$ = variance parameter for $k$th class, we assume here that variance is equal across all classes, $K$
- $\pi_k$ = prior probability that an observation belongs to the $k$th class

_LDA_ method approximates the Bayes classifier by plugging in estimates for $\mu_k$, $\sigma_k^2$, and $\pi_k$:

![Ch 4 pi and mu estimates](/images/statistical_learning/ch4-pi-and-mu-estimates.png)

The Bayes decision boudary in effect is show below:

![Ch 4 Bayes Descision p = 1](/images/statistical_learning/ch4-bayes-decision-p1.png)

### Ch 4.4.2 - Linear Discriminant Analysis for $p > 1$

Now we use _LDA_ above with multiple predictors, $p$. We start by assuming that the predictors, $X_p$, are drawn from a _multivariate Gaussian / mulitvariate normal_ distribution. This distribution assumes each predictor follows a one-dimensional normal distribution, with some correlation between each pair of $X$.

If plotting the density functions of two predictors, they will be considered **uncorrelated** if the distribution is a normal bell shape (left), but correlated if the variances are unequal (right):

![Ch 4 LDA Correlation](/images/statistical_learning/ch4-lda-correlation.png)

The Bayes decision boundary in effect is show below:

![Ch 4 Bayes Descision p > 1](/images/statistical_learning/ch4-bayes-decision-p2.png)

Each _pair of classes_ has its own decision boundary.

![Ch4 Confusion Matrix](/images/statistical_learning/ch4-confusion-matrix.png)

When making a confusion matrix, two key terms are of importance:

- Sensitivity: The percentage of true positives (trues correctly captured as trues)
- Specificity: The true negatives (falses correctly captured as falses)

These types of errors for all possible thresholds is captured using the **_receiver operating characteristics_** curve (**ROC**). One axis shows sensitivity (true positive) rates, the other shows false positive rates via 1-specificity. The higher the _area under the curve_ (AUC) the better. An AUC of 0.5 indicates a model performs no better than chance.

Terms defined below:

![Ch4 Matrix terms](/images/statistical_learning/ch4-matrix-terms.png)

### Ch 4.4.3 - Quadratic Discriminant Analysis

Like LDA, assumes observations from each class are drawn from a Gaussian distribution and plugs estimates for the parameters into Bayes' theorem to perform prediction. Unlike LDA, assumes each class has its own covaraince matrix. QDA is much more flexible, and this can lead to higher variance and lower bias. LDA is better than QDA for fewer training observations where reducing variance is crucial. QDA is recommended if the training set is very large so variance is not a major concern.

### Ch 4.4.4 - Naive Bayes

Recall that Bayes' theorem provides an expression for the posterior probability. Naive Bayes' avoids some of the simple, strict assumptions of LDA and QDA by assuming :

> Within the $k$th class, the $p$ predictors are independent.

I.e. we assume there is no relationship between the predictors. Naive Bayes introduces some bias, but reduces variance, and works very well when $n$ is not large enough relative to $p$ for effective join distribution estimates within each class.

## Ch 4.5 - A Comparison of Classification Methods

### Ch 4.5.1 - An Analytical Comparison

- LDA is a special case of QDA
- LDA is a special case of naive Bayes _and_ naive Bayes is a special case of LDA
- Neither QDA nor Naive Bayes are special cases of each other. Naive Bayes can produce more flexible fits, but QDA can be more accurate for predictors that have interactions which are important for discriminating between classes.

Selecting the apporpriate method depends on the true distribution of the predictors in each class, $K$, and other conisderations like $n$ and $p$ (see bias-variance trade off).

Comparison with KNN:

- KNN, being non-parametric, is better than LDA and logistic regression for non linear decision boundaries (with large $n$ and small $p$)
- KNN requires a lot of observations $n$ with relatively few predictors $p$
- QDA may be preferred when $n$ is small or $p$ is somewhat large
- KNN does not tell which predictors are important

## Ch 4.6 - Generalized Linear Models

For these models, we are interested in response, $Y$, that is neither quantitative nor qualitative. An example is _**counts**_ like counts of bike shares in a given time frame. These values are non-negative. This may be similar to census tracking.

A linear model can perform well, but in the example of bikeshare data, may predict _negative_ users. This leads to validity questions of the coefficients arrived at and assumptions made.

Instead, **Poisson Regression** can be a better approach, dictated by the _Poisson Distribution_, and can typically used to be model counts.

The Poisson Regression model requires attention to interpretation. It is also better at handling the **_mean-variance relationship_** of data over linear regression since linear regression always uses a constant value for variance. In the bike share data, usage and variance are both much higher during unfavorable weather conditions. Poisson Regression will also never have negative values.

Linear regression, logistic regression, and Poisson are all members of the _generalized linear model (GLM)_ family. Other examples include Gamma regression and negative binomial regression.

# Chapter 5 - Resampling Methods

This chapter will discuss two of the most common resampling methods:

- **cross-validation**
- **boostrapping**

Other definitions:

- **Model Assessment**: the process of evaluating a model's performance
- **Model Selection**: the process of selecting the proper level of flexibility for a model

## Ch 5.1 - Cross-Validation

Reminder definitions:

- **test error rate**: Average error that results from using a statistical learning method to predict the response on a new observation. Not easy to obtain unless a designated test set is available.
- **training error rate**: Similar to above, but easier to calculate by applying the method to the observations used in its training. Can wind up differing from the test error rate and risk underestimation.

### Ch 5.1.1 - The Validation Set Approach

The validation set approach involves randomly dividing the available set of observations into two parts: a **training set** and a **validation set** / **hold-out set**. The model is fit on the training set and the fitted model is used to predict the responses for the observations in the validation set

The resulting **validation set error rate** is an estimate of the test error rate.

There are two drawbacks to the validation set approach:

1) Test error rates in the validation estimate can be highly variable based on which observations are included in test versus validation sets.
2) By proxy of using fewer/a subset of observations to fit the model, the model's validation set error rate runs a risk of overestimating the test error rate.

### Ch 5.1.2 Leave-One-Out Cross-Validation (LOOCV)

Instead of two subsets of comparable size, LOOCV only includes a single observation in the validation set and the remaining observations make up the training set. The statistical learning method is then fit on the $n-1$ training observations and a prediction is made on the excluded observation.

This can be done $n$ times to produce $n$ MSEs. The LOOCV estimate for the test MSE is the average of these $n$ test error estimates.

Advantages over the validation set approach:

- Far less bias due to $n-1$ fits, versus valdation set approach where the training set is half the size of the original set.
- Doesn't overestimate the eerror rate as much as validation set approach
  - There is no randomness in the training/validation set splits

Downside: computationally expensive to implement. Large $n$ makes for each individual model being slow to fit.

### Ch 5.1.3 - $k$-Fold Cross-Validation

This approach involves randomly dividing the set of observations into $k$ groups/folds of approximately equal size. The first fold is treated as the validation set and the method is fit on the remaining $k-1$ folds. The MSE is then computed on the observations of the held-out fold. this procedure is then repeated $k$ times with each time having a different group of observations treated as the validation set.

The $k$ fold CV estimate is computed by averaging all of the $k$ estimates of the MSE.

> LOOCV is a special case of $k$ fold with $k$ = $n$

Not setting $k=n$ has a computational advantage, $k$ is usually set to 5 or 10. It can also be applied to almost any statistical learning method.

### Ch 5.1.4 - Bias-Variance Trade-Off for $k$ Fold Cross-Validation

Aside from computational advantages, $k$ fold CV can also give more accurate estimates of the test error rate than LOOCV thanks to bias-variance trade off.

LOOCV does a great job of reducing bias, but can have much higher variance when $k$ < $n$. This is because all of the fitted models are trained on a nearly identical set of observations (the difference being the hold out observation). This means the outputs are highly positively correlated with each other.

$k$ fold CV uses outputs less correlated with each other since there is less overlap between the traiing sets.

Using $k = 5$ or $k = 10$ have been shown to empirically yield test error rate estimates that suffer neither from excessively high bias or variance.

### Ch 5.1.5 - Cross-Validation on Classification Problems

Instead of using the MSE to quantify test error, instead we use the number of misclassified observations.

## Ch 5.2 - The Bootstrap

The _bootstrap_ method is a tool for quantifying uncertainty associated with a given estimator or modelling method. Ex: estimate the standard errors of the coefficients from a linear regression fit.

Rather than repeatedly obtaining data sets from the population, we instead obtain distinct data sets by repeatedly sampling observations from the original data set. Bootstrapping involves **repeated sampling with replacement**.

> Why is it called this? It is based off a fable where a baron was thrown in a lake and "pulled himself up by his bootstraps" to get himself out of the lake and save his life. It is the idea that you work with what you've got, we use the data itself to get information about our estimator.

# Chapter 6 - Linear Model Selection and Regularization

Before moving to non-linear models, this chapter will first look at how the linear model can be improved by replacing least squares fitting with alternative fitting methods. These methods have potential to yield better _prediction accuracy_ and _model interpretability_. We will use **feature/variable selection** to improve model interpretability, i.e. exclude irrelevant variables from a model like multiple regression. In this chapter we will discuss three methods for fitting:

1) **Subset Selection**: Identifying a subset of the $p$ predictors believed to be related to the response. Then, fit a model using least squares on the reduced set of variables.
2) **Shrinkage**: Fit a model involving all $p$ predictors, but shrink the estimated coefficients close to zero relative to the least squares estimates. The shrinkage/**regularization** reduces variance.
3) **Dimension Reduction**: Project the $p$ predictors into an $M$-dimensional subspace where $M < p$. This is acieved by computing $M$ different _linear combinations/projections_ of the variables. The projections are then used as predictors to fit a linear regresion model by least squares.

While methods here are applied in this chapter to regression, they can also be applied to classification models.

## Ch 6.1 - Subset Selection

### Ch 6.1.1 - Best Subset Selection

A separate least squares regression is fit for each possible combination of $p$ predictors. Once all resulting models are produces, the goal is to identify which is **best**. Selection of the "best" model is nontrivial

Done using the algorithm below:

1) $M_0$ is the "null model", containing no predictors and predicts the sample mean for each observation.
2) For $k = 1,2,...p$
   
   a) Fit all models that contain exactly $k$ predictors
   
   b) Pick the best among them and call it $M_k$, having the smallest RSS/largest $R^2$
3) Select a single best model from among $M_0....M_p$ using cross validated prediction error, $C_p$ (AIC), BIC, or adjusted $R^2$

Limitations:

- With greater predictors, $p$, the more models that need to be made ($2^p$), i.e. if $p$ is 10 then there are over 1000 models to be considered. With modern computers, its difficult for this method to be used for anything beyoned $p = 40$
- Larger search spaces can lead to overfitting and high variance 

### Ch 6.1.2 - Stepwise Selection

Stepwise methods can address the limits of BSS above.

#### Forward Stepwise Selection

Selects a much smaller set of models, then adds predictors to the model one at a time, until all predictors are in the model. At each step, the variable that give the greatest additional improvement to the fit is added to the model:

1) Let $M_0$ denote the null model, containing no predictors
2) For $k=0,....p-1$:

   a) Consider all $p-k$ models that augment the predictors in $M_k$ with one additional predictors

   b) Choose the best among these $p-k$ models, $M_{k+1}$, determined by smallest RSS or highest $R^2$
3) Select a single best model from among $M_0....M_p$ using cross validated prediction error, $C_p$ (AIC), BIC, or adjusted $R^2$

Instead of over 1 million models when $p=20$, this method results in only 211 models. While forward stepwise selection does well in practice, it is not guaranteed to find the best possible model.

> In the case with $p=3$ predictors, the best possible one variable model has $X_1$ and the best possible two variable model has $X_2$ and $X_3$. Forward selection will fail to select the best two variable model because $M_1$ and $M_2$ will always contain $X_1$.

#### Backward Stepwise Selection

Unlike forward, this method begins with the full leas squares model containing all $p$ predictors and the interatively removes the least useful predictor, one at a time:

1) Let $M_0$ denote the null model, containing no predictors
2) For $k=p,p-1,...,1$:

   a) Consider all $k$ models that contain all but one of the predictors in $M_k$ for a total of $k-1$ predcitors

   b) Choose the best among these $k$ models, $M_{k+1}$, determined by smallest RSS or highest $R^2$
3) Select a single best model from among $M_0....M_p$ using cross validated prediction error, $C_p$ (AIC), BIC, or adjusted $R^2$

Similar to forward stepwise selection, can be applied in settings where $p$ is too large for best subset selection. It requires that $n$ is larger than $p$, whereas forward can be used even if $n<p$.

#### Hybrid Approaches

Generally these 3 methods give similar, but not identical models. Hybrid versions combine forward and backward, mimic best selection and attempt to retain the computational advantages of forward and backward.

### Ch 6.1.3 Choosing the Optimal Model

RSS and $R^2$ are indicative of the training error, but can be poor representations of the test error, and so are not suitable for selecting the best model among these different methods. There are two common approaches:

1) Indirectly estimate the test error via adjustment to the training error to account for bias due to overfitting. This is done with these adjustment methods: $C_p, adjusted  R^2, AIC, BIC$.
2) Directly estimate the test error, using either a validation set or CV approach. This can be advantagious to those described above due to the direct estimate and fewer assumptions about the true underlying model. It can also be used in scenarios where it is hard to predict the degrees of freedom or estimate the error variance ($\sigma^2$). In the past this was computationally prohibitive, so method (1) was preferred. But now we can do this with modern tech!

## Ch 6.2 - Shrinkage Methods

Instead of using least squares to fit a linear model containing a subset of the predictors, we can instead fit a model containing all predictors using a technique that _constrains_ or _regularizes_ the coefficient estimates. I.e., _shrinks_ the coefficient estimates towards zero. This can greatly reduce variance. The two most well known techniques are:

- Ridge regression
- Lasso

### Ch 6.2.1 - Ridge Regression

Recall that in Ch3 how least suqares fitting estiamtes coefficient values that minimize RSS (residual sum of squaers i.e. training error):

![Ch 3 RSS](/images/statistical_learning/ch6-rss.png)

Ridge Regression is very similar, except the coefficients are estimated by minimizing s slightly different quanitity using $\lambda$, a "tuning parameter":

![CH 6 Ridge Regression](/images/statistical_learning/ch6-ridge-regression.png)

Here, the $\lambda$ term introduces a **shrinkage penalty**, and is small when the coefficients ($\beta$) are close to zero. $\lambda$ controls the relative impact of the penalty.

Unlike least squares which produces one set of coefficient estimates, ridge regression produces a different set of estiamtes for each value of $\lambda$.

Ridge regression has an advantage over leas squares thanks to the _bias-variance tradeoff_. Flexibility decreases as $\lambda$ increases, leading to decreased variance but increased bias.

In the figure below, note how a "swet spot" can be found where bias and variance intersect to produce a minimized test mean squared error.

![Ch 6 Ridge Regression Visualized](/images/statistical_learning/ch6-ridge-regression-visualized.png)

### Ch 6.2.2 - The Lasso

Ridg Regression big disadvantage: includes all $p$ predictors in final model, and never sets any of them to exactly 0 (though it may get closeto 0) through the $\lambda$ penalty. Introducing the lasso:

![ch 6 Lasso Eq](/images/statistical_learning/ch6-lasso-eq.png)

This is very similar to ridge regression except that $\beta_j^2$ is replaced with $|\beta_j|$ for the "lasso penalty." The difference is that the lasso penalty can force coefficient estimates to equal exactly zero with sufficiently large $\lambda$. Similar to best subset selection, lasso performs variables selection.

While lasso yields "sparse" models by being only a subset of variables, they tend to be more interpretable than ridge regression ones.

![Ch6 Lasso Visualized](/images/statistical_learning/ch6-lasso-visualized.png)

## Ch 6.3 - Dimension Reduction Methods

Instead of using a subset of the original vars, or shrinking their coefficients towards zero, this method attempts to transform the predictors and then fit a least squares model using the transformed variables. "Dimension reduction" comes from reducing the number of variables or features while retaining the most important information.

$Z_m = \sum_{j=1}^N\phi_{jm}X_j$

Linear regression using the transformed predictors can “often” outperform linear regression using the original predictors.

$M$ represents something less than $p$, our original predictors.

### Ch 6.3.1 Principal Components Regression/Analysis (PCA/PCR)

**PCA** 

While used primarily in unsupervised learning, PCA is also used as a dimension reduction technique for regression. Visually:

![Ch 6 PCA](/images/statistical_learning/ch6-pca.png)

It is a technique for reducing the dimension of an $n x p$ matrix. The first principal component direction of the data is that along which the observations vary the **most**. This is shown below on the left hand figure:

![Ch 6 PCA Projection](/images/statistical_learning/ch6-pca-projection.png)

In PCA, we choose $\phi$ values that capture as much variance as possible.

The second principal component, the blue dashed line, is orthogonal to the first one and capture the second most variation, etc.

PCA can also be interpreted as having the first principal component vector be the line that is as close as possible to the data, i.e. in the example it is the line minimizing the sum of squared perpendicular distances between each point and the line.

The x's in the plot represent the projections of the scores onto the line, when the graph is rotated the distances of those values from the mean values of population and ad spending (the blue dot).

The number of principal components is based on the number of predictors, so in the advertising/population example we only have these two predictors and can at most have two principal components.

**PCR**

In PCR, the key idea is that often a small number of principal components suffice to explain most of the variability in the data and the relationship with the response.

PCR tends to do well in cases where the first few components are sufficient fo capture most of the variation and relationship with the response.

PCR is **not** a feature selection method because each of the $M$ principal components used in the regression is a linear combination of all $p$ original features. PCR is more closely related to ridge regrssion than to lasso, it can even be thought of as a continuous version of ridge regression.

### Ch 6.3.2 Partial Least Squares (PLS)

Since PCR is an unsupervised method (i.e. the response is not used to help determine the principal component directions), a drawback to it is that there is no guarantee the directions that best explain the predictors will also best explain the response.

PLS is a supervised method. First a new set of features ($Z_1,....Z_M$) are identified that are linear combinations of the original features. Then it fits a linear model via least squares using these new $M$ features. PLS attempts to find directions that explain both the repsonse _and_ the predictors.

In practice, PLS is often not as good as ridge or PCR and can reduce bias but increase variance.