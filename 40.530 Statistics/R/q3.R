library(dplyr)
library(magrittr)


# Reading data with separator tab
raw_riasec <- read.csv('RIASEC.csv',sep = '\t')

# (a)CLEANING UP
# List of realistic traits
realistic_trait <- c('R1','R2','R3','R4','R5','R6','R7','R8')

# Extracting out the realistic traits
raw_realistic <- raw_riasec[,realistic_trait]

# Removing rows with -1 from the dataframe
# realistic <- raw_realistic[raw_realistic != -1, ]

realistic <- raw_realistic %>%
  filter(R1 > 0 &  R2 > 0 & R3 > 0 & R4 > 0 & R5 > 0 & R6 > 0 & R7 > 0 & R8 > 0)
  

# (b)MODEL SELECTION

# Computing the score for the R trait
realistic <- realistic %>%
  rowwise() %>%
  mutate(Rscore = mean(c(R1, R2, R3, R4, R5, R6, R7, R8))) 

# Generating the training and validation set 
tr_realistic <- realistic[1:6500,]
val_realistic <- realistic[-(1:6500),]

# Building the linear model
lm_riasec <- lm(Rscore~R1, data = tr_realistic)
RSS_tr = sum(lm_riasec$residuals^2)
print(RSS_tr)
# RSS_tr = 2902.039

# estimated regression function and residual sum of squares
print(lm_riasec$coefficients)
# R1 = 1.0181052 + 0.4235993 * R1


# (c)VALIDATION
reg.fn <- function(x) 1.0181052 + 0.4235993 * x

val_realistic %<>%
  mutate(pred_Rscore = reg.fn(R1),
         residuals = reg.fn(R1) - Rscore )

avg_RSS_tr = mean(lm_riasec$residuals^2)
avg_RSS_val = mean(val_realistic$residuals^2)

print (avg_RSS_tr) # 0.4464676
print (avg_RSS_val) # 0.5201091

# The residual sum of squares for the validation set using the regression function
# is larger than the residual sum of square for the training set but are of the
# same order. Thus the model generalizes well.


