# Machine Learning Notes
## Authored by Richard Hanna


## Dictionary

- **Regression**: The process of relating (1)+ dependent variables to (1)+ independent variables. Regression models show if changes observed in DVs are associated with changes in IVs.

## Simple Linear Regression

$\widehat{y} = b_0 + b_1 x_1$

- There is a y-intercept at $b_0$ when $x$ is 0.
- For every unit increase in $b_1$ results in a slope increase
  - ex: $b_1 = 1/2$ means a shift in y of 1 unit and shift in x of 2 units

### Ordinary Least Squares

A means of determining the best slope line for the regression. By looking at the difference between a true value, $y_i$, and an estimated value based on a residual, $\widehat{y}$, and averaging their differences, the minimal sum will lead to the best model equation.

- residual: ${\epsilon}_i = y_i - \widehat{y}$
- $SUM(y_i - \widehat{y})^2$

Example Graphic:

![Ordinary Least Squares Regression](/images/ordinary_least_squares_reg.png)

## Multiple Linear Regression

$\widehat{y} = b_0 + b_1x_1 + b_2 x_2 ... b_n x_n$