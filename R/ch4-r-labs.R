# Chapter 4 R Labs

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
