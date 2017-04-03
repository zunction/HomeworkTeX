library(leaps)
library(dplyr)

# read csv file into dataframe car
car <- read.csv('carmpgdat.csv')

# generate a linear model with normally distributed noise with the model MPG~VOL+HP+SP+WT
lm <- lm(MPG~VOL+HP+SP+WT, data = car)

# estimated regression function and residual sum of squares
print(lm$coefficients)

# MPG = 192.43775332 - 0.01564501 * VOL + 0.39221231 * HP - 1.29481848 * SP - 1.85980373 * WT
RSS = sum(lm$residuals^2)
print(RSS)




leaps(x = car[,c('VOL','HP','SP','WT')], y = car[,'MPG'], names =c('VOL','HP','SP','WT'), method = 'Cp')



#model1 <- glm(MPG~VOL+HP+SP+WT,data = car)
#plot(model1)
#summary(model1)

#model2 <- glm(MPG~HP+SP+WT,data = car)
#summary(model2)



