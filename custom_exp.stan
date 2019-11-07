functions {
  real exp_mean_lpdf(vector x, real lambda) {
    vector[num_elements(x)] lprob;
    lprob = log(1/lambda) - x/lambda;
    return sum(lprob);
  }
}
data {
  int<lower=0> N;
  vector[N] y;
}

parameters {
  real<lower=0> lambda;
}

model {
  // target += log(1/lambda) - y / lambda;  // You can do it without the function
  // y ~ exp_mean(lambda); // Or you can use the function
  target += exp_mean_lpdf(y | lambda); // This is equivalent to the line above
  
  // notice that it doesn't have _lpdf when using ~ notation
}

