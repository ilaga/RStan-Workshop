data {
  int<lower=0> N; // the number of observations
  int K; // the number of predictors
  real y[N]; // the rent prices
  matrix[N,K] X; // model matrix
}
parameters {
  vector[K] beta; // the coefficients
  real<lower=0> sigma; // the standard deviation
}
transformed parameters{
//  vector[N] linpred;
//  linpred = X * beta;
}
model{
  vector[N] linpred = X * beta;
  //priors
  target += normal_lpdf(beta | 2, 0.01);
  //likelihood
  target += normal_lpdf(y | linpred, sigma);
}

