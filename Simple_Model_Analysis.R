
library(rstan)
library(shinystan)
library(brms)

############################################################
############################################################
##
## Run a simple rstan model and look at the results
## Run the default normal example
##
############################################################
############################################################

n = 1000
mu = 5
sigma = 3
y = rnorm(n, mu, sigma)
df = data.frame(y = y)

stan.data = list(
  N = n,
  y = y
)

stan.fit = stan(file = "Default_Norm.stan",
                data = stan.data,
                chains = 2,
                cores = 2,
                iter = 400,
                warmup = 300)

summary(stan.fit)
######################
## First look how to get the posterior samples
######################

## We use <extract> to get posterior samples
mu.est = extract(stan.fit, par = "mu")[[1]]
length(mu.est)
mean(mu.est)
mu

sigma.est = extract(stan.fit, par = "sigma")[[1]]
length(sigma.est)
mean(sigma.est)
sigma

######################
## Second look some sampling diagnostics
######################
pairs(stan.fit)
stan_ess(stan.fit)
stan_rhat(stan.fit)
stan_plot(stan.fit)
stan_trace(stan.fit)
stan_ac(stan.fit)
launch_shinystan(stan.fit)






######################
## Now let's try to do it with the brms package
######################

brms.fit = brm(y ~ 1, data = df, chains = 2, cores = 2, iter = 400, warmup = 300)
summary(brms.fit)
fixef(brms.fit)
pairs(brms.fit)
plot(brms.fit)
launch_shinystan(brms.fit)





######################
## Brms also lets you use traditional lm and glm functions, like predict
######################
head(resid(brms.fit))
resid(brms.fit, summary = F)[1:5,1:5]
head(predict(brms.fit))
predict(brms.fit, summary = F)[1:5,1:5]




## To get more info, look at
methods(class = class(stan.fit))
methods(class = class(brms.fit))











