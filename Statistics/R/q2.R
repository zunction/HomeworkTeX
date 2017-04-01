library(leaps)

car <- read.csv('carmpgdat.csv')

lm.1 <- lm(MPG~VOL+HP+SP+WT, data = car)

leaps(x = car[,c('VOL','HP','SP','WT')], y = car[,'MPG'], names =c('VOL','HP','SP','WT'))



#model1 <- glm(MPG~VOL+HP+SP+WT,data = car)
#plot(model1)
#summary(model1)

#model2 <- glm(MPG~HP+SP+WT,data = car)
#summary(model2)



