# Chapter 4 R Labs ----
# Class Lab ----
# Fitting Logistic Regression Functions in base R
library(ISLR2)

summary(Smarket)
names(Smarket)

pairs(Smarket, cols = Smarket$Direction)

glm.fit <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
      family = binomial, data = Smarket)
summary(glm.fit)


glm.probs <- predict(glm.fit, type = "response")
glm.probs[1:5]
glm.pred <- ifelse(glm.probs>0.5, "Up", "Down")
attach(Smarket)

table(glm.pred, Direction)
mean(glm.pred == Direction)

train <- Year < 2005
glm.fit <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
               family = binomial, data = Smarket, subset = train)

glm.probs <- predict(glm.fit, newdata = Smarket[!train,], type = "response")
glm.pred <- ifelse(glm.probs>0.5, "Up", "Down")
Direction.2005 <- Smarket$Direction[!train]
table(glm.pred, Direction.2005)
mean(glm.pred==Direction.2005)

# Linear Discriminant Analysis in base R

require(ISLR2)
require(MASS)

lda.fit <- lda(Direction~Lag1+Lag2, data = Smarket, subset = Year<2005)
plot(lda.fit)

Smarket.2005 <- subset(Smarket, Year == 2005)

lda.pred <- predict(lda.fit, Smarket.2005)
data.frame(lda.pred)[1:5,]

table(lda.pred$class, Smarket.2005$Direction)
mean(lda.pred$class == Smarket.2005$Direction)

# KNN with base R

library(class)
?knn

attach(Smarket)

Xlag <- cbind(Lag1, Lag2)
train <- Year<2005
knn.pred <- knn(Xlag[train,], Xlog[!train,], Direction[train], k = 1)
table(knn.pred, Direction[!train])
mean(knn.pred == Direction[!train])



# Tidy Models Lab ----

library(tidymodels)
library(ISLR) # For the Smarket data set
library(ISLR2) # For the Bikeshare data set
library(discrim)
library(poissonreg)
library(corrr)

cor_Smarket <- Smarket |> 
  select(-Direction) |> 
  correlate()

rplot(cor_Smarket, colours = c("indianred2", "black", "skyblue1"))

library(paletteer)

cor_Smarket |> 
  stretch() |> 
  ggplot(aes(x, y, fill = r)) +
  geom_tile() +
  geom_text(aes(label = as.character(fashion(r)))) +
  scale_fill_paletteer_c("scico::roma", limits = c(-1,1), direction = -1)

ggplot(Smarket, aes(Year, Volume)) + geom_jitter(height = 0) + theme_minimal()

# Logistic Regression

lr_spec <- logistic_reg() |> 
  set_engine("glm") |># Default
  set_mode("classification") # Default

lr_fit <- lr_spec |> 
  fit(
    Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
    data = Smarket
  )

lr_fit |> pluck("fit") |> summary()

tidy(lr_fit)

predict(lr_fit, new_data = Smarket)
predict(lr_fit, new_data = Smarket, type = "prob")

augment(lr_fit, new_data = Smarket) |> 
  conf_mat(truth = Direction, estimate = .pred_class) |> 
  autoplot(type = "heatmap")

augment(lr_fit, new_data = Smarket) |> 
  accuracy(truth = Direction, estimate = .pred_class)

### Train Test Split

Smarket_train <- Smarket |> 
  filter(Year != 2005)

Smarket_test <- Smarket |> 
  filter(Year == 2005)

lr_fit2 <- lr_spec |> 
  fit(
    Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
    data = Smarket_train
  )

augment(lr_fit2, Smarket_test) |> 
  conf_mat(Direction, .pred_class)

augment(lr_fit2, Smarket_test) |> 
  accuracy(Direction, .pred_class)

lr_fit3 <- lr_spec %>%
  fit(
    Direction ~ Lag1 + Lag2,
    data = Smarket_train
  )

augment(lr_fit3, new_data = Smarket_test) %>%
  conf_mat(truth = Direction, estimate = .pred_class) 

augment(lr_fit3, new_data = Smarket_test) %>%
  accuracy(truth = Direction, estimate = .pred_class) 

### Linear Discriminant Analysis

lda_spec <- discrim_linear() |> 
  set_mode("classification") |> 
  set_engine("MASS")

lda_fit <- lda_spec %>%
  fit(Direction ~ Lag1 + Lag2, data = Smarket_train)

predict(lda_fit, new_data = Smarket_test)
predict(lda_fit, new_data = Smarket_test, type = "prob")

augment(lda_fit, new_data = Smarket_test) %>%
  conf_mat(truth = Direction, estimate = .pred_class) 
augment(lda_fit, new_data = Smarket_test) %>%
  accuracy(truth = Direction, estimate = .pred_class) 


### Quadratic Discriminant Analysis

qda_spec <- discrim_quad() %>%
  set_mode("classification") %>%
  set_engine("MASS")

qda_fit <- qda_spec %>%
  fit(Direction ~ Lag1 + Lag2, data = Smarket_train)

augment(qda_fit, new_data = Smarket_test) %>%
  conf_mat(truth = Direction, estimate = .pred_class) 

augment(qda_fit, new_data = Smarket_test) %>%
  accuracy(truth = Direction, estimate = .pred_class) 


### Naive Bayes

nb_spec <- naive_Bayes() %>% 
  set_mode("classification") %>% 
  set_engine("klaR") %>% 
  set_args(usekernel = FALSE)  

nb_fit <- nb_spec %>% 
  fit(Direction ~ Lag1 + Lag2, data = Smarket_train)

augment(nb_fit, new_data = Smarket_test) %>% 
  conf_mat(truth = Direction, estimate = .pred_class)

augment(nb_fit, new_data = Smarket_test) %>% 
  accuracy(truth = Direction, estimate = .pred_class)


### K-Nearest Neighbors

knn_spec <- nearest_neighbor(neighbors = 3) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_fit <- knn_spec %>%
  fit(Direction ~ Lag1 + Lag2, data = Smarket_train)

knn_fit

augment(knn_fit, new_data = Smarket_test) %>%
  conf_mat(truth = Direction, estimate = .pred_class) 

augment(knn_fit, new_data = Smarket_test) %>%
  accuracy(truth = Direction, estimate = .pred_class) 

### Poisson Regression with Bikeshare Data
pois_spec <- poisson_reg() %>% 
  set_mode("regression") %>% 
  set_engine("glm")

pois_rec_spec <- recipe(
  bikers ~ mnth + hr + workingday + temp + weathersit,
  data = Bikeshare
) %>% 
  step_dummy(all_nominal_predictors())

pois_wf <- workflow() %>% 
  add_recipe(pois_rec_spec) %>% 
  add_model(pois_spec)

pois_fit <- pois_wf %>% fit(data = Bikeshare)

augment(pois_fit, new_data = Bikeshare, type.predict = "response") %>% 
  ggplot(aes(bikers, .pred)) +
  geom_point(alpha = 0.1) +
  geom_abline(slope = 1, linewidth = 1, color = "grey40") +
  labs(title = "Predicting the number of bikers per hour using Poission Regression",
       x = "Actual", y = "Predicted")

pois_fit_coef_mnths <- 
  tidy(pois_fit) %>% 
  filter(grepl("^mnth", term)) %>% 
  mutate(
    term = stringr::str_replace(term, "mnth_", ""),
    term = forcats::fct_inorder(term)
  ) 

pois_fit_coef_mnths %>% 
  ggplot(aes(term, estimate)) +
  geom_line(group = 1) +
  geom_point(shape = 21, size = 3, stroke = 1.5, 
             fill = "black", color = "white") +
  labs(title = "Coefficient value from Poission Regression",
       x = "Month", y = "Coefficient")

pois_fit_coef_hr <- 
  tidy(pois_fit) %>% 
  filter(grepl("^hr", term)) %>% 
  mutate(
    term = stringr::str_replace(term, "hr_X", ""),
    term = forcats::fct_inorder(term)
  )

pois_fit_coef_hr %>% 
  ggplot(aes(term, estimate)) +
  geom_line(group = 1) +
  geom_point(shape = 21, size = 3, stroke = 1.5, 
             fill = "black", color = "white") +
  labs(title = "Coefficient value from Poission Regression",
       x = "hours", y = "Coefficient")

### Comparing Multiple Models

models <- list("logistic regression" = lr_fit3,
               "LDA" = lda_fit,
               "QDA" = qda_fit,
               "KNN" = knn_fit)

preds <- imap_dfr(models, augment, 
                  new_data = Smarket_test, .id = "model")

preds %>%
  select(model, Direction, .pred_class, .pred_Down, .pred_Up)


multi_metric <- metric_set(accuracy, sensitivity, specificity)

preds |> 
  group_by(model) |> 
  multi_metric(truth = Direction, estimate = .pred_class)

preds %>%
  group_by(model) %>%
  roc_curve(Direction, .pred_Down) %>%
  autoplot()
