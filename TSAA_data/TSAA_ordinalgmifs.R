# if you have not downloaded the library yet:
# --> go to packages, install, search the name and click install

# in case your R-version is < 4.1.3 run the below code:
#install.packages("installr")
#library(installr)
#updateR()

library("ordinalgmifs")

# arguments:
## x-parameter:
### Which matrix of covariates should be penalized in the model fititing algorithm.

## model type (stereotype)
## link function
## logit
data <- read.csv('TSAA_original_data.csv')
data

#ordinalgmifs(group, x, data, epsilon)
# group = what should be predicted
# x = 
data[,1]

stereotype.logit <- ordinalgmifs(X ~ 1, x=data[,-1], data=data, epsilon=0.01)

#summary(stereotype.logit)
coef(stereotype.logit)
plot(stereotype.logit)

stereotype.logit$zeta
