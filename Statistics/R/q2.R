library(leaps)
library(survey)
library(dplyr)

# read csv file into dataframe car
car <- read.csv('carmpgdat.csv')

# (a) Fitting a mulitple linear regression model

# generate a linear model with normally distributed noise with the model MPG~VOL+HP+SP+WT
# covariates <-cbind('VOL','HP','SP','WT')
lm_car <- lm(MPG~VOL+HP+SP+WT, data = car)


# estimated regression function and residual sum of squares
coefficients(lm_car)
# MPG = 192.43775332 - 0.01564501 * VOL + 0.39221231 * HP - 1.29481848 * SP - 1.85980373 * WT

RSS = sum(lm_car$residuals^2)
# RSS = 1027.381

# (b) Mallows's Cp
# Since the AIC criterion is equivalent to Mallows's Cp, 

# (i) Forward
base <- lm(MPG~WT, data = car)
step(base, scope = list(upper = lm_car, lower=~1), direction = 'forward', trace = TRUE)

# Start:  AIC=240.45
# MPG ~ WT
# 
# Df Sum of Sq    RSS    AIC
# + SP    1    82.981 1383.0 237.68
# + HP    1    35.380 1430.6 240.45
# <none>              1466.0 240.45
# + VOL   1     3.883 1462.1 242.24
# 
# Step:  AIC=237.68
# MPG ~ WT + SP
# 
# Df Sum of Sq    RSS    AIC
# + HP    1    349.37 1033.7 215.80
# + VOL   1     45.97 1337.0 236.90
# <none>              1383.0 237.68
# 
# Step:  AIC=215.8
# MPG ~ WT + SP + HP
# 
# Df Sum of Sq    RSS   AIC
# <none>              1033.7 215.8
# + VOL   1    6.2685 1027.4 217.3
# 
# Call:
#   lm(formula = MPG ~ WT + SP + HP, data = car)
# 
# Coefficients:
#   (Intercept)           WT           SP           HP  
# 194.1296      -1.9221      -1.3200       0.4052

# (ii) Backward
step(lm_car,direction = 'backward', trace = TRUE)

# Start:  AIC=217.3
# MPG ~ VOL + HP + SP + WT
# 
# Df Sum of Sq    RSS    AIC
# - VOL   1      6.27 1033.7 215.80
# <none>              1027.4 217.30
# - HP    1    309.67 1337.0 236.90
# - SP    1    373.36 1400.7 240.72
# - WT    1   1013.76 2041.2 271.59
# 
# Step:  AIC=215.8
# MPG ~ HP + SP + WT
# 
# Df Sum of Sq    RSS    AIC
# <none>              1033.7 215.80
# - HP    1    349.37 1383.0 237.68
# - SP    1    396.97 1430.6 240.45
# - WT    1   1322.87 2356.5 281.37
# 
# Call:
#   lm(formula = MPG ~ HP + SP + WT, data = car)
# 
# Coefficients:
#   (Intercept)           HP           SP           WT  
# 194.1296       0.4052      -1.3200      -1.9221  
# 
# For the forward stepwise approach, the base model is important
# as it will change the outcome. For example for this, if we were
# to start with VOL instead of any of the other covariates, we 
# will end up with also the VOL covariate. By not starting with with
# the VOL covariate we will end up with a model without VOL which 
# corresponds to the backward stepwise approach and also the 
# Zheng-Loh model selection. As for the backward stepwise approach
# we do not have such a problem as we start the model with all the 
# covariates and reduce it down by computing the AIC of the model
# with different covariate missing then remove the covariate that 
# gives the smallest AIC when removed.



# (c) Zheng-Loh Model Selection
# Wald test for the covariates

regTermTest(lm_car, 'VOL', null=NULL,df=Inf, method = "Wald")
regTermTest(lm_car, 'HP', null=NULL,df=Inf, method = "Wald")
regTermTest(lm_car, 'SP', null=NULL,df=Inf, method = "Wald")
regTermTest(lm_car, 'WT', null=NULL,df=Inf, method = "Wald")

# Wald test for VOL
# in lm(formula = MPG ~ VOL + HP + SP + WT, data = car)
# Chisq =  0.4698075  on  1  df: p= 0.49308 
# 
# Wald test for HP
# in lm(formula = MPG ~ VOL + HP + SP + WT, data = car)
# Chisq =  23.20929  on  1  df: p= 1.4529e-06 
# 
# Wald test for SP
# in lm(formula = MPG ~ VOL + HP + SP + WT, data = car)
# Chisq =  27.98266  on  1  df: p= 1.2241e-07 
# 
# Wald test for WT
# in lm(formula = MPG ~ VOL + HP + SP + WT, data = car)
# Chisq =  75.97941  on  1  df: p= < 2.22e-16
# 
# Arranging in descending order we have:
#   WT > SP > HP > VOL
# We can do this as the Chisq test statistic is just the square 
# of the test statistic of the Wald test.
# 
# Let lm_j be the linear model with the jth largest Wald test statistic

n <- nrow(car)

lm_1 <- lm(MPG ~ WT, data = car)
jhat_1 = sum(lm_1$residuals^2) + (1 * RSS/(n-4) * log(n))
print (jhat_1)
# jhat_1 = 1524.045

lm_2 <- lm(MPG ~ WT + SP, data = car)
jhat_2 = sum(lm_2$residuals^2) + (2 * RSS/(n-4) * log(n))
print (jhat_2)
# jhat_2 = 1499.108

lm_3 <- lm(MPG ~ WT + SP + HP, data = car)
jhat_3 = sum(lm_3$residuals^2) + (3 * RSS/(n-4) * log(n)) 
print (jhat_3)
# jhat_3 =  1207.78

jhat_4 = RSS + (4 * RSS/(n-4) * log(n)) 
print (jhat_4)
# jhat_4 = 1259.555

# Thus the Zheng-Loh model selection method will select WT, SP and HP 
# as the covariates for predicting the MPG. Comparing it to (b), we see
# that the Zheng-Loh model selection method gives similar outcome to using
# Mallows's Cp model forward and backward stepwise to selecting a model.