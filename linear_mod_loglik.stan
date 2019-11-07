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
  
  //likelihood
  target += normal_lpdf(y | linpred, sigma);
}
generated quantities{
  vector[N] log_lik;
  for (n in 1:N) log_lik[n] = normal_lpdf(y[n] | X[n, ] * beta, sigma);
}
