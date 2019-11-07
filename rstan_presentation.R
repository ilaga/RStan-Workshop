
## Table of datasets: https://hofmann.public.iastate.edu/data_in_r_sortable.html
## Stan User's Guide: https://mc-stan.org/docs/2_19/stan-users-guide/index.html
## Initialization: https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
## Reference Manual: https://mc-stan.org/docs/2_19/reference-manual/index.html#overview
## rstanarm vignettes: http://mc-stan.org/rstanarm/articles/index.html

## Helpful summary of type declaration in user's guide user-defined functions
## More detailed descriptions in stan reference manual

library(rstan)
library(rstanarm)
library(brms)
library(shinystan)

library(kknn) # For Munich data set


## The next batch of code is useful initilization to speed up rstan
# options(mc.cores = parallel::detectCores())
cores = parallel::detectCores()
rstan_options(auto_write = TRUE)
Sys.setenv(LOCAL_CPPFLAGS = '-march=native')
chains = 4





################################################################################
################################################################################
##
## Example 1: Fit a truncated data model
## (we don't know how many values are missing)
## Likelihood: Exponential distribution, truncated below 1 and above 5
## Prior: Flat
##
## The first fit will be the correctly specified model
## The second fit will ignore the truncation
##
################################################################################
################################################################################

## Assume y comes from an exponential distribution
## Get rid of values below 1 and above 5

set.seed(537)

lambda = 1/3
y.raw.1 = rexp(3000, rate = lambda)
y.trunc = subset(y.raw.1, y.raw.1 > 5 & y.raw.1 < 10)
summary(y.trunc)
length(y.trunc)


## First let's assume we know the truncation points
lookup("dexp")


trunc_data <- list(
  N = length(y.trunc),
  L = 5,
  U = 10,
  y = y.trunc
)

trunc1 <- stan(
  file = "truncated_known.stan",  # Stan program
  data = trunc_data,      # named list of data
  chains = 4,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain
  cores = 4,              # number of cores  
  refresh = 200          # show progress every 'refresh' iterations
)


print(trunc1)


simple_data <- list(
  N = length(y.trunc),
  y = y.trunc
)

fit_simple <- stan(
  file = "simple_exp.stan",  # Stan program
  data = simple_data,      # named list of data
  chains = 4,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain
  cores = 4,              # number of cores  
  refresh = 200          # show progress every 'refresh' iterations
)

print(fit_simple)


launch_shinystan(trunc1)



################################################################################
################################################################################
##
## Fit a censored data model (we now know how many values are missing)
##
################################################################################
################################################################################



set.seed(537)

lambda = 1/3
y.cens.raw = rexp(1000, rate = lambda)
y.cens.1 = subset(y.cens.raw, y.cens.raw < 7)
summary(y.cens.1)
length(y.cens.1)
n.cens.1 = 1000 - length(y.cens.1)
n.obs.1 = length(y.cens.1)


cens_data <- list(
  N_obs = n.obs.1,
  N_cens = n.cens.1,
  y_obs = y.cens.1,
  U = 7
)


fit_cens <- stan(
  file = "censored_exp.stan",  # Stan program
  data = cens_data,      # named list of data
  chains = 4,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain
  cores = 4,              # number of cores  
  refresh = 200          # show progress every 'refresh' iterations
)

## Guess what the output is going to be
print(fit_cens)





print(fit_cens, pars = "lambda")
pairs(fit_cens, pars = c("lambda", "lp__"))
pairs(fit_cens, pars = c("lambda", "lp__", "y_cens[77]", "y_cens[78]"))






# Now let's add some covariates -------------------------------------------


set.seed(537)

n.sample = 1000
doses = c(0, 1, 2, 3, 4)
dose.vec = sample(doses, n.sample, replace = T)
x.mat = model.matrix(~ factor(dose.vec) - 1)
dose.eff = c(0, 0.4, 0.7, 0.95, 1.2)
lambda.vec = as.numeric(x.mat %*% dose.eff) + 1/3
y.dose.raw = rexp(n.sample, rate = lambda.vec)
dat = data.frame(y = y.dose.raw, dose = dose.vec)
dat.obs.2 = subset(dat, y > 2)
dat.cens.2 = subset(dat, y <= 2)

n.cens.2 = nrow(dat.cens.2)
n.obs.2 = nrow(dat.obs.2)

x.obs.2 = model.matrix(~factor(dose) - 1, data = dat.obs.2)
x.cens.2 = model.matrix(~factor(dose) - 1, data = dat.cens.2)


cens_data_dose <- list(
  N_obs = n.obs.2,
  N_cens = n.cens.2,
  K = 5,
  x_obs = x.obs.2,
  x_cens = x.cens.2,
  y_obs = dat.obs.2$y,
  U = 2
)


fit_cens_dose <- stan(
  file = "censored_exp_dose.stan",  # Stan program
  data = cens_data_dose,      # named list of data
  chains = 4,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain
  cores = 4,              # number of cores  
  refresh = 200          # show progress every 'refresh' iterations
)

print(fit_cens_dose, pars = c("beta", "lp__"))
dose.eff + 1/3
print(fit_cens_dose, pars = "y_cens", include = F)
dose.eff[-1] - dose.eff[1]






################################################################################
################################################################################
##
## Example 4: Linear regression
## The linear regression examples use the miete data set
## I've selected a few covariates just for illustration
##
################################################################################
################################################################################


data(miete)
str(miete)
summary(miete)
## Want to predict net rent per sqm (in DM) "nm" using the other predictors
## Just simple linear regression

## Only look at floor space, year, central heating, hot water,
## tiled bathroom, window, kitchen type

miete = miete[,c("nmqm", "wfl", "bj", "bad0", "zh", "ww0", "badkach",
                 "fenster", "kueche")]


# Now fit using the .stan file -------------------------------------------

X = model.matrix(nmqm ~ ., data = miete)


## Prepare the data
lm.data = list(N = nrow(miete),
               y = miete$nmqm,
               K = ncol(X),
               X = X)

fit1 = stan(
  file = "linear_mod.stan",
  data = lm.data,
  chains = 4,
  warmup = 2000,
  iter = 4000,
  cores = 4,
  refresh = 250
)

print(fit1)

help(stan_plot)
help("rstan-plotting-functions")
stan_plot(fit1)
stan_trace(fit1)
stan_scat(fit1, pars = c("beta[1]", "beta[3]")) ## Needs two parameters
pairs(fit1, pars = c("beta[1]", "beta[3]")) ## But really this is similar
stan_hist(fit1)
stan_dens(fit1)
stan_ac(fit1)



# Now fit with strong prior -----------------------------------------------

fit2 = stan(
  file = "linear_mod_prior.stan",
  data = lm.data,
  chains = 4,
  warmup = 2000,
  iter = 4000,
  cores = 4,
  refresh = 250
)


print(fit2, pars = c("beta", "sigma"))




# Historical data ---------------------------------------------------------

## Simulate historical control data from weibull

set.seed(408)
beta_0 = 2
gamma = 1
n_hist = 1000
n_cont = 5
n_treat = 500
y_hist = rweibull(n_hist, shape = beta_0, gamma)
y_cont = rweibull(n_cont, shape = beta_0, gamma)
y_treat = rweibull(n_treat, shape = beta_0, gamma + 2)

y.all = c(y_cont, y_treat)
dose.all = c(rep(0, n_cont), rep("T", n_treat))
dat = data.frame(y = y.all, dose = factor(dose.all))
surv.both = survival::survreg(Surv(y.all, y.all > 0) ~ dose - 1, data = dat)
surv.cont = survival::survreg(Surv(y_cont, y_cont > 0) ~ 1, data = dat)
surv.treat = survival::survreg(Surv(y_treat, y_treat > 0) ~ 1, data = dat)

exp(coef(surv.both))
1/surv.both$scale

exp(coef(surv.cont))
1/surv.cont$scale

exp(coef(surv.treat))
1/surv.treat$scale


hist_data <- list(
  N_c = n_cont,
  N_t = n_treat,
  N_h = n_hist,
  alpha_0 = 1,
  w = 0.5,
  mu_h = 3,
  sigma_h = 1,
  y_hist = y_hist,
  y_cont = y_cont,
  y_treat = y_treat
)



fit_hist <- stan(
  file = "historical.stan",  # Stan program
  data = hist_data,      # named list of data
  chains = 4,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain
  cores = 4,              # number of cores  
  refresh = 200          # show progress every 'refresh' iterations
)

print(fit_hist)


no_hist_data <- list(
  N_c = n_cont,
  N_t = n_treat,
  N_h = n_hist,
  alpha_0 = 0,
  w = 0.5,
  mu_h = 3,
  sigma_h = 1,
  y_hist = y_hist,
  y_cont = y_cont,
  y_treat = y_treat
)



fit_no_hist <- stan(
  file = "historical.stan",  # Stan program
  data = no_hist_data,      # named list of data
  chains = 4,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain
  cores = 4,              # number of cores  
  refresh = 200          # show progress every 'refresh' iterations
)


print(fit_hist)
print(fit_no_hist)
rbind(beta_0, gamma, 2)








################################################################################
################################################################################
##
## Example 4: Custom pdfs and posteriors
## We will define our own prior and likelihood
## We will use the exponential distribution, just for illustration
## This time, my lambda parameter will refer to the mean, rather than the rate
##
################################################################################
################################################################################

set.seed(935)
y.4 = rexp(300, rate = 0.4) # Corresponds to a mean of 2.5


custom_data <- list(
  N = length(y.4),
  y = y.4
)

custom_fit <- stan(
  file = "custom_exp.stan",  # Stan program
  data = custom_data,      # named list of data
  chains = 4,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain
  cores = 4,              # number of cores  
  refresh = 200          # show progress every 'refresh' iterations
)

print(custom_fit)
mean(y.4)












################################################################################
################################################################################
##
## Example 5: brms and rstanarm
## I'll briefly show the truncation problem again
## Mainly I'll use the miete linear regression example
##
################################################################################
################################################################################

## Truncation

tmp.data = data.frame(y = y.trunc)
trunc.brms = brm(y | trunc(lb = 5, ub = 10) ~ 1, family = exponential, cores = 4,
                 chains = 4, data = tmp.data,
                 warmup = 2000, iter = 4000)

## What's going on with our answer?

make_stancode(y | trunc(lb = 5, ub = 10) ~ 1, family = exponential, data = tmp.data)

## We see that the our intercept is exp(-lambda)
summary(exp(-posterior_samples(trunc.brms)[,1]))

# make_standata(y | trunc(lb = 5, ub = 10) ~ 1, family = exponential, data = tmp.data)




## Linear Regresion

rstanarm.fit = stan_lm(nmqm ~ ., data = miete, chains = 4, prior = NULL,
                 cores = cores, iter = 2000)
pairs(rstanarm.fit, pars = c("(Intercept)", "wfl", "bad01"))
summary(rstanarm.fit)
print(rstanarm.fit)


brms.fit = brm(nmqm ~ ., data = miete, chains = 4, cores = cores,
                iter = 2000, family = gaussian())
summary(brms.fit)

methods(class = class(rstanarm.fit))
methods(class = class(fit1))
methods(class = class(brms.fit))

prior_summary(rstanarm.fit)
prior_summary(brms.fit)


## Now let's add some strong briors using the brms package
## There is no way to do this with rstanarm, since the prior is on R^2
prior = c(prior(normal(2, 0.01), class = "Intercept"),
          prior(normal(2, 0.01), class = "b"))
brms.prior = brm(nmqm ~ ., data = miete,
                 cores = cores, chains = 4, iter = 4000,
                 prior = prior)

summary(brms.prior)
prior_summary(brms.prior)


make_stancode(nmqm ~ ., data = miete,
              cores = cores, chains = 4, iter = 4000,
              prior = prior)

make_standata(nmqm ~ ., data = miete,
              cores = cores, chains = 4, iter = 4000,
              prior = prior)




################################################################################
################################################################################
##
## Show functionality of add_criterion and compare
##
################################################################################
################################################################################


## Look at loo using rstan
## We need to add a log_lik calculationg to the stan file

X = model.matrix(nmqm ~ ., data = miete)


## Prepare the data
lm.data = list(N = nrow(miete),
              y = miete$nmqm,
              K = ncol(X),
              X = X)

X2 = model.matrix(nmqm ~ .  - fenster, data = miete)
lm.data.2 = list(N = nrow(miete),
              y = miete$nmqm,
              K = ncol(X2),
              X = X2)

lm.rstan.1 = stan(
  file = "linear_mod_loglik.stan",
  data = lm.data,
  chains = 4,
  warmup = 2000,
  iter = 4000,
  cores = 4,
  refresh = 250
)

lm.rstan.2 = stan(
  file = "linear_mod_loglik.stan",
  data = lm.data.2,
  chains = 4,
  warmup = 2000,
  iter = 4000,
  cores = 4,
  refresh = 250
)


print(lm.rstan.1, pars = "log_lik", include = F)
print(lm.rstan.2, pars = "log_lik", include = F)

loo.1 = loo(lm.rstan.1)
loo.2 = loo(lm.rstan.2)

loo.1
loo.2



## In this section, we will fit multiple models and compare performance using brms

brms.1 = brm(nmqm ~ wfl + bj + bad0 + zh + ww0 + badkach + fenster + kueche,
             data = miete, cores = cores, chains = chains, iter = 4000)
brms.1 = add_criterion(brms.1, "loo", reloo = T)
brms.1 = add_criterion(brms.1, "kfold", k = 10)

brms.2 = brm(nmqm ~ wfl + bj + bad0 + zh + ww0 + badkach + fenster,
             data = miete, cores = cores, chains = chains, iter = 4000)
brms.2 = add_criterion(brms.2, "loo", reloo = T)
brms.2 = add_criterion(brms.2, "kfold", k = 10)

brms.3 = brm(nmqm ~ wfl + bj + bad0 + zh + ww0 + badkach,
             data = miete, cores = cores, chains = chains, iter = 4000)
brms.3 = add_criterion(brms.3, "loo")
brms.3 = add_criterion(brms.3, "kfold", k = 10)

brms.4 = brm(nmqm ~ wfl + bj,
             data = miete, cores = cores, chains = chains, iter = 4000)
brms.4 = add_criterion(brms.4, "loo")
brms.4 = add_criterion(brms.4, "kfold", k = 10)


loo_compare(brms.1, brms.2, brms.3, brms.4, criterion = "loo")
loo_compare(brms.1, brms.2, brms.3, brms.4, criterion = "kfold")


stanplot(brms.1)
stanplot(brms.1, type = "rhat")




################################################################################
################################################################################
##
## There are lots of transformations you can make
## Transformations don't need Jacobians, as they are specified before sampling
## Change of variables do require a Jacobians
## Ex: If y ~ lognormal, can sample log(y) ~ normal(mu, sigma)
##      then at the Jacobian as target += -log(y)
## 
################################################################################
################################################################################
