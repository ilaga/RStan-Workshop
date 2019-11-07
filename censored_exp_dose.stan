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
  int<lower=0> K;
  matrix[N_obs, K] x_obs;
  matrix[N_cens, K] x_cens;
  real y_obs[N_obs];
  real<upper=min(y_obs)> U;
}

parameters {
  vector<lower = 0>[K] beta;
  real<lower = 0,upper=U> y_cens[N_cens];
}

model {
  y_cens ~ exponential(x_cens * beta);
  y_obs ~ exponential(x_obs * beta);
}
generated quantities {
  vector[K] dif = beta - beta[1];
}
