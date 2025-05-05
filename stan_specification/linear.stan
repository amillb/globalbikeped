

data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=0> L;                     // number of city-level covariates (excl. intercept)
  int<lower=0> M;                     // number of country-level covariates (excl. intercept)
  array[N] int<lower=1,upper=J> country_num;
  vector[J] gdp_percapita;            // use length J rather than N
  vector[J] dependency_ratio;
  vector[J] gas_prices;
  vector[N] sndi;
  vector[N] density;
  vector[N] precip;
  vector[N] min_temp;
  vector[N] bikelanes;
  vector[N] slope;
  vector[N] includes_inboundoutbound;
  vector[N] rail_in_city;
  vector[N] max_temp;
  vector[N] population;
  vector[N] min_temp2;
  vector[N] y_bike;
  vector[N] y_walk;
  real bikelanes_global_mean;    // mean undstandardized bikelane value from the full dataset
  real bikelanes_global_stdev;   // standard deviation of the undstandardized bikelane value from the full dataset
} 


transformed data {
    // roughly following https://mc-stan.org/docs/stan-users-guide/multivariate-hierarchical-priors.html
    // individual predictor matrix
    matrix[N, L+1] X; 
    for (i in 1:N) {
        X[i,1] = 1;
        X[i,2] = sndi[i];
        X[i,3] = density[i];
        X[i,4] = precip[i];
        X[i,5] = min_temp[i];
        X[i,6] = bikelanes[i];
        X[i,7] = slope[i];
        X[i,8] = includes_inboundoutbound[i];
        X[i,9] = rail_in_city[i];
        X[i,10] = max_temp[i];
        X[i,11] = population[i];   
        X[i,12] = min_temp2[i];   
    } 

    // group-level predictors
    matrix[M+1, J] Z;          
    for (j in 1:J) {
        Z[1,j] = 1.0;
        Z[2,j] = gdp_percapita[j];
        Z[3,j] = dependency_ratio[j];
        Z[4,j] = gas_prices[j];
    } 

}

parameters {
  vector<lower=0>[L+1] tau_walk;           // scale of diagonals
  vector<lower=0>[L+1] tau_bike;           

  cholesky_factor_corr[(L+1)] L_Omega_walk;
  cholesky_factor_corr[(L+1)] L_Omega_bike;

  matrix[L+1, J] v_walk; 		   // called z in stan manual
  matrix[L+1, J] v_bike;

  matrix[L+1,M+1] gamma_walk;              // country-level coefficients (including the intercept)
  matrix[L+1,M+1] gamma_bike;              // country-level coefficients (including the intercept)  

  real<lower=0> sigma_y_walk;
  real<lower=0> sigma_y_bike;

} 

transformed parameters {

  matrix[L+1, J] beta_walk = gamma_walk * Z + diag_pre_multiply(tau_walk, L_Omega_walk) * v_walk;
  matrix[L+1, J] beta_bike = gamma_bike * Z + diag_pre_multiply(tau_bike, L_Omega_bike) * v_bike;
  
}

model {
  // Weakly informative priors for covariance matrix and second-level parameters
  tau_walk ~ exponential(5); // tighter than recommended by Stan manual
  tau_bike ~ exponential(5);
  to_vector(v_walk) ~ std_normal();
  to_vector(v_bike) ~ std_normal();
  L_Omega_walk ~ lkj_corr_cholesky(2);
  L_Omega_bike ~ lkj_corr_cholesky(2);
  to_vector(gamma_walk) ~ normal(0, 0.1); // tighter than recommended by Stan manual
  to_vector(gamma_bike) ~ normal(0, 0.1);
  sigma_y_walk ~ exponential(5); // tighter than recommended by Stan manual
  sigma_y_bike ~ exponential(5);

  for (n in 1:N) {
    y_bike[n] ~ normal(X[n, ] * beta_bike[, country_num[n]], sigma_y_bike);
    y_walk[n] ~ normal(X[n, ] * beta_walk[, country_num[n]], sigma_y_walk);
  }
}

generated quantities {
    vector[N] y_hat_bike;        
    vector[N] y_hat_walk;        
    vector[N] y_rep_bike;
    vector[N] y_rep_walk;
    int<lower=0, upper=1> mean_gt_bike;
    int<lower=0, upper=1> mean_gt_walk;
    int<lower=0, upper=1> sd_gt_bike;
    int<lower=0, upper=1> sd_gt_walk;

    // marginal effects, array/vector structure is parallel to beta
    // me is only different for the squared term, which will represent the combined
    vector[N] me_walk_mintemp;  
    vector[N] me_bike_mintemp; 

    // commented out until we rerun generated quantities 
    //real<lower=0> lane_delta;
    //real new_standardized_value;
    //matrix[N,100] aas;  // additional active share in each city, for bike lane ratios of 0.01...1.00
    
    for(n in 1:N){
        y_hat_bike[n] = X[n, ] * beta_bike[, country_num[n]]; 
        y_hat_walk[n] = X[n, ] * beta_walk[, country_num[n]]; 

        y_rep_bike[n] = normal_rng(X[n, ] * beta_bike[, country_num[n]], sigma_y_bike);
        y_rep_walk[n] = normal_rng(X[n, ] * beta_walk[, country_num[n]], sigma_y_walk);

        me_bike_mintemp[n] = beta_bike[5, country_num[n]] + 2*beta_bike[12, country_num[n]]*X[n,5];
        me_walk_mintemp[n] = beta_walk[5, country_num[n]] + 2*beta_walk[12, country_num[n]]*X[n,5];

	
    }
   // see https://mc-stan.org/docs/2_27/stan-users-guide/posterior-p-values.html
   {
    mean_gt_bike = mean(y_rep_bike) > mean(y_bike);
    mean_gt_walk = mean(y_rep_walk) > mean(y_walk);
    sd_gt_bike = sd(y_rep_bike) > sd(y_bike);
    sd_gt_walk = sd(y_rep_walk) > sd(y_walk);
   }



}
