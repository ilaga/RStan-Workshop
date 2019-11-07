
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

data {
  int<lower=0> N_c;
  int<lower=0> N_t;
  int<lower=0> N_h;
  real<lower=0, upper=1> alpha_0;
  real<lower=0,upper=1> w;
  real mu_h;
  real<lower=0> sigma_h;
  vector[N_h] y_hist;
  vector[N_t] y_treat;
  vector[N_c] y_cont;
}


parameters {
  real<lower=0> beta_0;
  real<lower=0> gamma;
  real beta;
}


model {
  // alpha0 ~ normal(0, 100);
  // sigma0 ~ 
  // beta_0 += weibull_lpdf(alpha0, sigma0) + normal_lpdf;
  target += alpha_0 * weibull_lpdf(y_hist | beta_0, gamma) +
            normal_lpdf(beta_0 | 0, 100) + inv_gamma_lpdf(gamma | 0.0001, 0.0001) +
            log(w * exp(normal_lpdf(beta | mu_h, sigma_h)) +
                (1 - w) * exp(normal_lpdf(beta | 0, sqrt(10)))) +
                // Everything up to here is the joint prior
            weibull_lpdf(y_treat | beta_0, gamma + beta) + weibull_lpdf(y_cont | beta_0, gamma);
}

