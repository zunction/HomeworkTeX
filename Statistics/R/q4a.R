# (a) Using Monte Carlo method with N samples

N <- 100000
I <- function(u){X = rgamma(1.5,1/2.3,n = N); return(sum(1<X & X<2)/N)}
I_hat <- I

# I_hat = 0.20146

# Estimated standard error by resampling Monte Carlo 10000 times.
se_mc <- sqrt(var(sapply(1:N,I)))

# se_mc =   0.001272072

# Estimated standard error using 10000 bootstrap samples.
B <- 10000
base_sample <- rgamma(1.5,1/2.3,n = N)
I_bootstrap <- function(X){X = sample(base_sample, N, replace = TRUE);return(sum(1<X & X<2)/N)}
se_bootstrap <- sqrt(var(sapply(1:B,I_bootstrap)))

# se_bootstrap = 0.001273952


