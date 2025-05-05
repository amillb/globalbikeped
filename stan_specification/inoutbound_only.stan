
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
        X[i,8] = rail_in_city[i];
        X[i,9] = max_temp[i];
        X[i,10] = population[i];   
        X[i,11] = min_temp2[i];  // note that squared term marginal effect is hard coded - see and adjust generated quantities block               
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
        X_means[j,8] = rail_in_city_mean[j];
        X_means[j,9] = max_temp_mean[j];
        X_means[j,10] = population_mean[j];   
        X_means[j,11] = min_temp2_mean[j];  
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
  vector[N] y_hat_walk;        
  vector[N] y_hat_bike;        
  vector[N] y_rep_walk;	
  vector[N] y_rep_bike;
  int<lower=0, upper=1> mean_gt_walk;
  int<lower=0, upper=1> mean_gt_bike;
  int<lower=0, upper=1> sd_gt_walk;
  int<lower=0, upper=1> sd_gt_bike;

  // marginal effects, array/vector structure is parallel to beta for linear model
  array[N] vector[L+1] me_walk;  
  array[N] vector[L+1] me_bike;   
  array[M+1] vector[N] me_country_walk;
  array[M+1] vector[N] me_country_bike;
  

  y_hat_walk = mu_walk;  // for ease of syntax with linear models
  y_hat_bike = mu_bike;  

  y_rep_walk = to_vector(beta_proportion_rng(y_hat_walk, phi_walk));
  y_rep_bike = to_vector(beta_proportion_rng(y_hat_bike, phi_bike));

  mean_gt_walk = mean(y_rep_walk) > mean(y_walk);
  mean_gt_bike = mean(y_rep_bike) > mean(y_bike);
  sd_gt_walk = sd(y_rep_walk) > sd(y_walk);
  sd_gt_bike = sd(y_rep_bike) > sd(y_bike);


  // marginal effects of city-level variables. See email 8/16/24 from Andy Lin

   for (i in 1:L+1) {
      real epsilon;
      real p2_bike_u;  // marginal effect in positive direction
      real p2_bike_l;  // marginal effect in negative direction
      real p2_walk_u;
      real p2_walk_l;
  
      if(i == 8 || i == 9) { // binary
         for (n in 1:N) {
             if (X[n,i]==1) { // binary variable is 1 for that observation
	        p2_bike_u = mu_bike[n];
	        p2_bike_l = inv_logit(X[n,] * beta_bike[, country_num[n]] - beta_bike[i, country_num[n]] );
                p2_walk_u = mu_walk[n];
                p2_walk_l = inv_logit(X[n,] * beta_walk[, country_num[n]] - beta_walk[i, country_num[n]] );                  
      	      }
             else {
	        p2_bike_u = inv_logit(X[n,] * beta_bike[, country_num[n]] + beta_bike[i, country_num[n]] );
	        p2_bike_l = mu_bike[n];
                p2_walk_u = inv_logit(X[n,] * beta_walk[, country_num[n]] + beta_walk[i, country_num[n]] );                  
                p2_walk_l = mu_walk[n];
      	      }
             me_bike[n][i] = (p2_bike_u - p2_bike_l ) ;
             me_walk[n][i] = (p2_walk_u - p2_walk_l );
          }
      }
      else { // continuous variable
         epsilon = 0.001;

         for (n in 1:N) {
	 
	    if (i == 11) { // combined effect of linear and squared term
               p2_bike_u = inv_logit(X[n,] * beta_bike[, country_num[n]] + epsilon*beta_bike[5, country_num[n]] - X[n,i]*beta_bike[i, country_num[n]] + (X[n,5]+epsilon)^2*beta_bike[i, country_num[n]] );
               p2_bike_l = inv_logit(X[n,] * beta_bike[, country_num[n]] - epsilon*beta_bike[5, country_num[n]] - X[n,i]*beta_bike[i, country_num[n]] + (X[n,5]-epsilon)^2*beta_bike[i, country_num[n]] );
               p2_walk_u = inv_logit(X[n,] * beta_walk[, country_num[n]] + epsilon*beta_walk[5, country_num[n]] - X[n,i]*beta_walk[i, country_num[n]] + (X[n,5]+epsilon)^2*beta_walk[i, country_num[n]] );
               p2_walk_l = inv_logit(X[n,] * beta_walk[, country_num[n]] - epsilon*beta_walk[5, country_num[n]] - X[n,i]*beta_walk[i, country_num[n]] + (X[n,5]-epsilon)^2*beta_walk[i, country_num[n]] );
            }
	    else {
	       p2_bike_u = inv_logit(X[n,] * beta_bike[, country_num[n]] + epsilon*beta_bike[i, country_num[n]] );
	       p2_bike_l = inv_logit(X[n,] * beta_bike[, country_num[n]] - epsilon*beta_bike[i, country_num[n]] );
               p2_walk_u = inv_logit(X[n,] * beta_walk[, country_num[n]] + epsilon*beta_walk[i, country_num[n]] );
               p2_walk_l = inv_logit(X[n,] * beta_walk[, country_num[n]] - epsilon*beta_walk[i, country_num[n]] );
	    }

            me_bike[n][i] = (p2_bike_u - p2_bike_l ) / (2*epsilon);
            me_walk[n][i] = (p2_walk_u - p2_walk_l ) / (2*epsilon);
         }
      }
   }

   // marginal effects of country-level variables. 

   // calculate mean of X beta for the sample
   { real xbsum_bike = 0; 
     real xbsum_walk = 0;
     real mu_bike_u;
     real mu_bike_l;
     real mu_walk_u;
     real mu_walk_l;
     real epsilon = 0.001;

     matrix[M+1, J] Z_u;     
     matrix[M+1, J] Z_l;  

     vector[N] LP_bike_u;
     vector[N] LP_bike_l;
     vector[N] LP_walk_u;  // linear predictor
     vector[N] LP_walk_l;  // linear predictor
    

     for (m in 1:M+1) {
	Z_u = Z;
	Z_l = Z;
	Z_u[m, ] += epsilon;
	Z_l[m, ] -= epsilon;

	// city-level level coefficients plus the marginal effect, in positive direction
  	matrix[L+1, J] me_beta_bike_u = gamma_bike * Z_u + diag_pre_multiply(tau_bike, L_Omega_bike) * v_bike;
	matrix[L+1, J] me_beta_walk_u = gamma_walk * Z_u + diag_pre_multiply(tau_walk, L_Omega_walk) * v_walk;	
        // marginal effect in negative direction
  	matrix[L+1, J] me_beta_bike_l = gamma_bike * Z_l + diag_pre_multiply(tau_bike, L_Omega_bike) * v_bike;
	matrix[L+1, J] me_beta_walk_l = gamma_walk * Z_l + diag_pre_multiply(tau_walk, L_Omega_walk) * v_walk;	

      for (n in 1:N) {
    	LP_bike_u[n] = X[n, ] * me_beta_bike_u[, country_num[n]];
    	LP_walk_u[n] = X[n, ] * me_beta_walk_u[, country_num[n]];
    	LP_bike_l[n] = X[n, ] * me_beta_bike_l[, country_num[n]];
    	LP_walk_l[n] = X[n, ] * me_beta_walk_l[, country_num[n]];
      }

  	me_country_bike[m] = (inv_logit(LP_bike_u) - inv_logit(LP_bike_l)) / (2*epsilon);
  	me_country_walk[m] = (inv_logit(LP_walk_u) - inv_logit(LP_walk_l)) / (2*epsilon);

     }
   }
}

