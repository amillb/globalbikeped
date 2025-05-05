
// https://statmodeling.stat.columbia.edu/2024/02/28/varying-slopes-and-intercepts-in-stan-still-painful-in-2024/
// https://mc-stan.org/docs/stan-users-guide/regression.html#multivariate-hierarchical-priors.section



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
  real bikelanes_global_mean;    // mean unstandardized bikelane value from the full dataset
  real bikelanes_global_stdev;   // standard deviation of the unstandardized bikelane value from the full dataset

  // country-level means; used to calculate marginal effects
  // need to flag in generated quantities which ones are binary
  // don't need this for country-level variables, as they are all standardized and have mean 0
  vector[J] sndi_mean;
  vector[J] density_mean;
  vector[J] precip_mean;
  vector[J] min_temp_mean;
  vector[J] bikelanes_mean;
  vector[J] slope_mean;
  vector[J] includes_inboundoutbound_mean;
  vector[J] rail_in_city_mean;
  vector[J] max_temp_mean;
  vector[J] population_mean;
  vector[J] min_temp2_mean;
} 



transformed data {
    // roughly following https://mc-stan.org/docs/stan-users-guide/multivariate-hierarchical-priors.html
    // individual predictor matrix
    matrix[N, L+1] X;
    for (i in 1:N) {
        X[i,1] = 1.0;
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
        X[i,12] = min_temp2[i];  // note that squared term marginal effect is hard coded - see and adjust generated quantities block               
    } 

    // group-level predictors
    matrix[M+1, J] Z;          
    for (j in 1:J) {
        Z[1,j] = 1.0;
        Z[2,j] = gdp_percapita[j];
        Z[3,j] = dependency_ratio[j];
        Z[4,j] = gas_prices[j];
    } 



    // country-level means to calculate marginal effects
    matrix[J, L+1] X_means;

    for (j in 1:J) {
        X_means[j,1] = 1.0;
        X_means[j,2] = sndi_mean[j];
        X_means[j,3] = density_mean[j];
        X_means[j,4] = precip_mean[j];
        X_means[j,5] = min_temp_mean[j];
        X_means[j,6] = bikelanes_mean[j];
        X_means[j,7] = slope_mean[j];
        X_means[j,8] = includes_inboundoutbound_mean[j];
        X_means[j,9] = rail_in_city_mean[j];
        X_means[j,10] = max_temp_mean[j];
        X_means[j,11] = population_mean[j];   
        X_means[j,12] = min_temp2_mean[j];  
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

  real<lower = 0> phi_walk;                // dispersion
  real<lower = 0> phi_bike;                // dispersion

} 

transformed parameters {

  matrix[L+1, J] beta_walk = gamma_walk * Z + diag_pre_multiply(tau_walk, L_Omega_walk) * v_walk;
  matrix[L+1, J] beta_bike = gamma_bike * Z + diag_pre_multiply(tau_bike, L_Omega_bike) * v_bike;

  vector[N] mu_walk; 
  vector[N] mu_bike; 

  vector[N] LP_walk;  // linear predictor
  vector[N] LP_bike;
  
  for (n in 1:N) {
    LP_walk[n] = X[n, ] * beta_walk[, country_num[n]];
    LP_bike[n] = X[n, ] * beta_bike[, country_num[n]];
  }

  mu_walk = inv_logit(LP_walk);
  mu_bike = inv_logit(LP_bike);
  
}

model {
  // Weakly informative priors for covariance matrix and second-level parameters
  tau_walk ~ exponential(1);
  tau_bike ~ exponential(1);
  to_vector(v_walk) ~ std_normal();
  to_vector(v_bike) ~ std_normal();
  L_Omega_walk ~ lkj_corr_cholesky(1);
  L_Omega_bike ~ lkj_corr_cholesky(1);
  to_vector(gamma_walk) ~ normal(0, 1); 
  to_vector(gamma_bike) ~ normal(0, 1);
  phi_walk ~ gamma(4, 0.2);  // similar to recommended https://solomonkurz.netlify.app/blog/2023-06-25-causal-inference-with-beta-regression/          
  phi_bike ~ gamma(4, 0.2);    

  y_walk ~ beta_proportion(mu_walk, phi_walk);
  y_bike ~ beta_proportion(mu_bike, phi_bike);


}

generated quantities {
    real<lower=0> lane_delta;
    real new_standardized_value;
    matrix[N,100] aas_w;  // additional active share in each city, for bike lane ratios of 0.01...1.00
    matrix[N,100] aas_b;  
    matrix[N,100] aas_c;  // combined

   {  
   vector[N] y_hat_walk;        
   vector[N] y_hat_bike; 
   y_hat_walk = mu_walk;  // for ease of syntax with linear models
   y_hat_bike = mu_bike;  


   // calculate km displaced
   // this part of the model is where both the walk and bike estimates are used
   // until now, they have not been linked

   for (n in 1:N) { 

       for(j in 1:100) {
           new_standardized_value=(j/100.-bikelanes_global_mean)/bikelanes_global_stdev; // going from j of 1 to 100 to standardized value
           lane_delta = fmax(0, new_standardized_value-bikelanes[n]);  
   	   aas_b[n, j] = inv_logit(X[n, ] * beta_bike[, country_num[n]] + lane_delta*beta_bike[6, country_num[n]]) - y_hat_bike[n];
   	   aas_w[n, j] = inv_logit(X[n, ] * beta_walk[, country_num[n]] + lane_delta*beta_walk[6, country_num[n]]) - y_hat_walk[n];
   	   aas_c[n, j] = fmax(0, aas_b[n, j] + aas_w[n, j]);
        }
   }
  }
}

