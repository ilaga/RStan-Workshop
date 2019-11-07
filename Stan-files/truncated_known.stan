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
  int<lower=0> N;
  real L;
  real U;
  real<lower=L,upper=U> y[N];
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real<lower=0> lambda;
}

// The model to be estimated. We model the output
// 'y' to be exponentially distributed with rate 'lambda'
model {
  for (n in 1:N)
   y[n] ~ exponential(lambda) T[L,U];
}

