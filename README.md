# RStan-Workshop
A short introduction to using RStan.

## Part 1: What is RStan
Stan is mainly a probabilistic programming language to perform Bayesian inference with MCMC sampling via Hamiltonian Monte Carlo with a No-U-Turn Sampler (HMC-NUTS). Rstan is just a user interface to call Stan from R. Stan also has version for Python, shell, MATLAB, and others, which we won't discuss here. RStan is similar to Bayesian inference through BUGS. In order to perform the MCMC sampling, you provide R with a a STAN file (which specifies the model) and an R list with the input data. Models are written in STAN by specifying the priors for the parameters and the likelihoods for the data.

## Part 2: Install RStan

We first have to install RStan inside R. The RStan package is named "rstan" and has only a few dependencies that will also need to be installed. The command to install rstan is:

    install.packages("rstan")

## Part 3: Creating the STAN file
STAN files are made up of three necessary blocks (data, parameters, model) and additional optional blocks (functions, transformed data, etc). We won't explore all optional blocks in this workshop.

### Step 1: Create the actual file
#### Option 1:

  If your RStudio version is new enough, there is an option to create a STAN file using the new document shortcut.
  
  
#### Option 2:

  Alternatively, you can just create a blank file in R, notepad, etc, and save the file with a .stan extension.

### Step 2: Write the data block
The data block is where you declare the observed data (think of declaring variables in C). In the stan file, you need to specify the type of data (int, vector, real, etc), the constraints (lower bounds and upper bounds), and size (Y is a vector of size N). Everything must be explicitly written. An example of this is 


    data {

      int<lower=0> N;

      vector[N] y;

    }
  

### Step 3: Write the parameters block
Next, we declare the parameters that we want sampled. For example, if we assume our data comes from a normal distribution with mean mu and standard deviation sigma, then our parameters section would look something like: 

    parameters {

      real mu;

      real<lower=0> sigma;

    }
  
There are a few things to remember when declaring parameters. First, parameters cannot be discrete. Second, if a parameter has an upper or lower bound, this must be specified.

### Step 4: Write the model block
Finally, we can declare the priors and likelihoods for our data and parameters. What is nice about Stan is that the code looks very similar to how you would naturally discuss priors and likelihoods in a Bayesian framework. Continuing with the normal example, we might say something like, "the mean mu has a normal prior and sigma has an inverse prior. The data is normally distributed." This translates nicely to STAN code as follows: 

    model {

      mu ~ normal(0, 10);

      target += -log(sigma);

      y ~ normal(mu, sigma);

    }
  
Using the "~" notation should feel familiar since this is how we write in on paper. However, the target += -log(sigma) might look a little weird. If we don't want to specify a distribution for a parameter (sigma in this case), we can increment the target on the log-scale. Finally, note that even though "y" is a vector, we can still use "y ~ normal(mu, sigma);" STAN allows vectorization, which greatly decreases computation time.


## Part 3: Creating the data list in R
Before running the Stan model in R, we need create a data list. This list needs to include everything that we declared in the data block. For the above data block example, our data list would look like this:

    stan_data <- list(

      N = length(y),

      Y = y

    )
  
If any of our data parameters had lower or upper bounds, we would also need to specify those.

## Part 4: Calling Stan
Now that we have a stan file and a data list, we can call out model using the "stan" function. See "help(stan)" for more information about arguments. At the bare minimum, stan() requires a "file" argument which specifies the name of the stan file and a "data" argument which is the data list. For our above model, we can implement the MCMC sampling via:

    model <- stan(

      file = "stanfile.stan",

      data = stan_data

    )
  
Other important arguments are chains (how many MCMC chains should be run), warmup (number of burn-in samples), iter (total number of MCMC iteration), and cores (if running in parallel, how many chains should be run concurrently). However, there are lots of more options which can help with debugging and running the model.


