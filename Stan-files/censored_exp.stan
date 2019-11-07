//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N_obs;
  int<lower=0> N_cens;
  real y_obs[N_obs]; // vector[N]<lower = L, upper = U> foo
  real<lower=max(y_obs)> U;
}

parameters {
  real<lower=U> y_cens[N_cens];
  real<lower=0> lambda;
}

model {
  y_cens ~ exponential(lambda);
  y_obs ~ exponential(lambda);
}

