######################################################
######### Chapter 3: Least Squares Estimation ########
######################################################


########### Example 3.1: OLS of LM(1) ############

# Read data from an external file
mydata <- read.table('Stories.txt', header = TRUE)

# Preparing work
View(mydata)
Y <- mydata[,2]    # variable HGHT
X <- mydata[,3]    # variable STORIES
n <- nrow(mydata)  # sample size

# Find OLS estimates by formulae
y <- Y-mean(Y)   # deviation form of Y
x <- X-mean(X)   # deviation form of X
beta1 <- sum(y*x)/sum(x^2)
beta0 <- mean(Y)-beta1*mean(X)
sigma <- sqrt(sum((Y-beta0-beta1*X)^2)/(n-2))
beta0; beta1; sigma

# Fit the model by lm()
fit <- lm(Y~X)   # Fit the model
summary(fit)     # Summary of the fitted model


######### Example 3.2: Freedman #########

# Read data
library(car); data(Freedman); Freedman

# Fit a LIM of crime on all other variables
fit0 <- lm(crime ~ ., data = Freedman)
summary(fit0)

# Remove "density"
fit1 <- lm(crime ~ population + nonwhite, data = Freedman)
summary(fit1)

# F test of fit1 against fit0
# Find SST, SSR and SSE
anova(fit1)
F.stat <- 81829/73863558*96; F.stat
# An alternative way
SSE_F <- sum(fit0$residuals^2)
SSE_R <- sum(fit1$residuals^2)
F.stat2 <- (SSE_R-SSE_F)/SSE_R*fit0$df.residual; F.stat2
# p-value
pf(F.stat, 1, fit0$df.residual, lower.tail=F)

# Predictions
fit1$fitted.values
predict.lm(fit1, newdata = Freedman)
# Confidence intervals
predict.lm(fit1, Freedman, 
           interval = "confidence", level = 0.95)[1,]
# Prediction intervals
predict.lm(fit1, Freedman, 
           interval = "prediction", level = 0.95)[1,]


######### Example 3.3: Rent Expenditure #########

# Read data
setwd("C:/Teaching/DASC7011/DASC7011_202425/CH03")
rents <- read.csv("rents.csv")
y <- rents[,2]; x <- rents[,1]

# Scatter plot 1
plot(x, y, col="blue", xlab = "Income", ylab = "Rent")

# OLSE
fit0 <- lm(y ~ x)
summary(fit0)

# White's test using lmtest::bptest()
library(lmtest)
bptest(fit0)

# Manual GLSE: transform the data and OLS
x1 <- 1/sqrt(x)
x2 <- sqrt(x)
y1 <- y/sqrt(x)
fit1 <- lm(y1 ~ x1+x2-1)   # LM without intercept
summary(fit1)$coefficients
bptest(fit1)

# WLSE: Define weights and WLS
w <- 1/x
fit2 <- lm(y ~ x, weights = w)   
summary(fit2)$coefficients
bptest(fit2)


######### Example 3.4: OLSE with Consistent ESE #########

# White's Robust Standard Errors for rent-income model
library(sandwich)
summary.white <- function(model) {
  print(coeftest(model, vcov. = vcovHC))
  print(waldtest(model, vcov = vcovHC))
}                     # Define a function
summary(fit0)
summary.white(fit0)

# The Poverty data
poverty <- read.csv("poverty1.csv")
names(poverty)[1] = "P"
names(poverty)[2] = "U"

# OLS fit
fit.ols <- lm(P ~ U, data = poverty)
summary(fit.ols)

# Durbin-Watson Test
library(car)
dwt(fit.ols)
library(lmtest)
dwtest(fit.ols)       # Default alternative = "greater"
dwtest(fit.ols, exact = FALSE)
dwtest(fit.ols, alternative = "two.sided")
dwtest(fit.ols, alternative = "less")

# Breusch-Godfrey Test
library(lmtest)
bgtest(fit.ols, order = 2)
bgtest(fit.ols, order = 2, type = "F")
bgtest(fit.ols)

# Cochrane-Orcutt Estimation
library(orcutt) 
fit.co <- cochrane.orcutt(fit.ols)
summary(fit.co)

# Prais-Winsten Estimation for AR(1) Disturbance
library(prais)
prais_winsten(P~U, poverty, iter=50)
fit.pw <- prais_winsten(P~U, poverty)
summary(fit.pw)

# Nonlinear Least Squares
# Define lagged variables
poverty$LP <- NA
poverty$LU <- NA
for (i in 2:24) {
  poverty$LP[i] <- poverty$P[i-1]
  poverty$LU[i] <- poverty$U[i-1]
}
fit.nls <- nls(P ~ rho*LP+(1-rho)*int+slope*(U-rho*LU), 
               data=poverty, 
               start=list(rho=0, int=, slope=0))
summary(fit.nls)

# Newey-West Consistent Standard Error
library(sandwich)
summary.nw <- function(model) {
  print(coeftest(model, vcov. = vcovHAC))
  print(waldtest(model, vcov = vcovHAC))
}                     # Define a function
summary.nw(fit.ols)
summary(fit.ols)


