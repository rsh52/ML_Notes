# Machine Learning Notes
## Authored by Richard Hanna


## Dictionary

- **Regression**: The process of relating (1)+ dependent variables to (1)+ independent variables. Regression models show if changes observed in DVs are associated with changes in IVs.

## Linear Regression

### Assumptions of Linear Regression

**Anscombe's Quartet** shows 4 spreads of data that all have the same linear regression model, but are obviously not suited for linear regression:

![Anscombe's Quartet](/images/anscombes_quartet.png)

There are **5** assumptions:

1. **Linearity**: There is a linear relationship between Y and each X
2. **Homoscedasticity**: Equal variance
3. **Multivariate Normality**: Normality of error distribution
4. **Independence**: No autocorrelation, no pattern in the data indicating that rows aren't independent of each other. Ex: Stock market prices influencing future markers
5. **Lack of Multicollinearity**: Predictors are not correlated with each other
6. Bonus **Outlier Check**: No outliers that significantly affect the model

### Simple Linear Regression

$\widehat{y} = b_0 + b_1 x_1$

- There is a y-intercept at $b_0$ when $x$ is 0.
- For every unit increase in $b_1$ results in a slope increase
  - ex: $b_1 = 1/2$ means a shift in y of 1 unit and shift in x of 2 units

#### Ordinary Least Squares

A means of determining the best slope line for the regression. By looking at the difference between a true value, $y_i$, and an estimated value based on a residual, $\widehat{y}$, and averaging their differences, the minimal sum will lead to the best model equation.

- residual: ${\epsilon}_i = y_i - \widehat{y}$
- $SUM(y_i - \widehat{y})^2$

Example Graphic:

![Ordinary Least Squares Regression](/images/ordinary_least_squares_reg.png)

### Multiple Linear Regression

$\widehat{y} = b_0 + b_1x_1 + b_2 x_2 ... + b_n x_n$

In this example, there are multiple factors that can influence $\widehat{y}$ with varying slope effects.

#### Dummy Variables

Dummy variables can be useful for non-numeric values, i.e. classification or categorical variables (ex: US State, color, shape). You make a dummy variable for each categorical variable and assign a value of 1 or 0 as to whether or not the variable is `TRUE`.

In our states example, all 50 states would have a corresponding dummy column populated by 1's and 0's.

> The "dummy variable trap": you will have (N - 1) dummy columns because the final column can be surmised by the values of all the rest. Including this would result in "multicollinearity" and keep the model from distinguishing the effects of one column on another (in the case of only 2 dummy variables).

We can represent this in our Multiple Linear Regression equation:

$\widehat{y} = b_0 + b_1x_1 + b_2 x_2 ... + b_n x_n ... b_nD_n$

Where $D_n$ are the dummy variables extrapolated from your categorical value column.

#### P-Values

The P-value is used to assess whether or not you reject the null hypothesis. Statistical significance is the threshold where a sufficinet level of confidence is met to reject the null hypothesis.

#### Building a Model

You must first figure out what potential predictors ($X_n$) you want to keep and what ones you want to throw away (think GIGO garbage in garbage out).

Types of approaches:

1) All in: You keep everything
2) Backwards elimination: 
  a) Step 1: Select a significance level (SL) to stay in the model
  b) Step 2: Fit the full model with all possible predictors
  c) Step 3: Consider the predictor with the highest P-value. If P > SL, go to step 4. Otherwise model is ready!
  d) Step 4: Remove the predictor
  e) Step 5: Fit the model without this variable, return to step 3.
3) Forwards selection:
  a) Step 1: Select a significance level (SL) to stay in the model
  b) Step 2: Fit all simple regression models $y$ ~ $x_n$. Select the one with the lower P-value
  c) Step 3: Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have.
  d) Step 4: Consider the predictor with the lower P-value. If P < SL, go to Step 3 again, otherwise model is ready!