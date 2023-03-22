# Tidymodels Ch5 labs

library(tidymodels)
library(ISLR)

Auto <- tibble(Auto)
Portfolio <- tibble(Portfolio)

set.seed(1)

Auto_split <- initial_split(Auto, strata = mpg, prop = 0.5)
Auto_split

Auto_train <- training(Auto_split)
Auto_test <- testing(Auto_split)

# Modeling goal: predict `mpg` by `horsepower`
# First: simple linear regression and polynomial regression models

lm_spec <- linear_reg() |> 
  set_mode("regression") |> 
  set_engine("lm")

lm_fit <- lm_spec |> 
  fit(mpg ~ horsepower, data = Auto_train)

augment(lm_fit, new_data = Auto_test) |> 
  rmse(truth = mpg, estimate = .pred)

augment(lm_fit, new_data = Auto_train) |> 
  rmse(truth = mpg, estimate = .pred)

poly_rec <- recipe(mpg ~ horsepower, data = Auto_train) |> 
  step_poly(horsepower, degree = 2)

poly_wf <- workflow() |> 
  add_recipe(poly_rec) |> 
  add_model(lm_spec)

poly_fit <- poly_wf |> 
  fit(data = Auto_train)

augment(poly_fit, new_data = Auto_test) |> 
  rmse(truth = mpg, estimate = .pred)

# K-fold CV with hyperparameter tuning

poly_tuned_rec <- recipe(mpg ~ horsepower, data = Auto_train) |> 
  step_poly(horsepower, degree = tune())

poly_tuned_wf <- workflow() |> 
  add_recipe(poly_tuned_rec) |> 
  add_model(lm_spec)

Auto_folds <- vfold_cv(Auto_train, v = 10)

degree_grid <- grid_regular(degree(range = c(1,10)), levels = 10)

tune_res <- tune_grid(
  object = poly_tuned_wf,
  resamples = Auto_folds,
  grid = degree_grid
)

autoplot(tune_res)

collect_metrics(tune_res)
show_best(tune_res, metric = "rmse")
best_degree <- select_by_one_std_err(tune_res, degree, metric = "rmse")
best_degree

final_wf <- finalize_workflow(poly_wf, best_degree)
final_wf
final_fit <- fit(final_wf, Auto_train)
final_fit


# The Bootstrap

Portfolio_boots <- bootstraps(Portfolio, times = 1000)

# Write fx that takes a boot_split and returns the caluclated fx
alpha.fn <- function(split) {
  data <- analysis(split)
  X <- data$X
  Y <- data$Y
  
  (var(Y) - cov(X, Y)) / (var(X) + var(Y) - 2 * cov(X, Y))
}

# Apply alpha.fn to each bootstrap
alpha_res <- Portfolio_boots %>%
  mutate(alpha = map_dbl(splits, alpha.fn))

alpha_res
