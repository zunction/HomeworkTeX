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
print(lm_car$coefficients)

# MPG = 192.43775332 - 0.01564501 * VOL + 0.39221231 * HP - 1.29481848 * SP - 1.85980373 * WT
RSS = sum(lm_car$residuals^2)
print(RSS)

# (b) Mallow Cp

# (i) Forward

# (ii) Backward

# leaps(x = car[,c('VOL','HP','SP','WT')], y = car[,'MPG'], names =c('VOL','HP','SP','WT'), method = 'Cp')


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
# 
# Let lm_j be the linear model with the jth largest Wald test statistic

lm_1 <- lm(MPG ~ WT, data = car)
jhat_1 = sum(lm_1$residuals^2) + 1 * RSS * log(82) 
print (jhat_1)

lm_2 <- lm(MPG ~ WT + SP, data = car)
jhat_2 = sum(lm_2$residuals^2) + 2 * RSS * log(82) 
print (jhat_2)

lm_3 <- lm(MPG ~ WT + SP + HP, data = car)
jhat_3 = sum(lm_3$residuals^2) + 3 * RSS * log(82) 
print (jhat_3)

jhat_4 = RSS + 4 * RSS * log(82) 
print (jhat_4)


print(RSS)
