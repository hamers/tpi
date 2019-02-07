#include "tpi.h"

#include "H5Cpp.h"
using namespace H5;


using namespace std;
  
/*===========================================
  Classes
  ===========================================*/

/* parameters */
parameters_t::parameters_t() // default parameters
{
  OpenMP_max_threads = 1;
  minimum_N_for_OpenMP = 100;
  N_resume = 10;
  minimum_N_tp_calc_for_GPU = 50;
  enable_resume_functionality=false;
  enable_OpenMP=false;
}
  
/* elements */
void elements_t::set_elements(double mass_, double semimajor_axis_, double eccentricity_, double mean_anomaly_, double inclination_, double argument_of_pericenter_, double longitude_of_ascending_node_)
{
  mass = mass_;
  semimajor_axis = semimajor_axis_;
  eccentricity = eccentricity_;
  mean_anomaly = mean_anomaly_;
  inclination = inclination_;
  argument_of_pericenter = argument_of_pericenter_;
  longitude_of_ascending_node = longitude_of_ascending_node_;
}

void elements_t::convert_to_pv(double time, double r_[3], double v_[3], parameters_t *parameters)
{
  convert_from_elements_to_pv(time,parameters->SMBH_mass+mass,semimajor_axis,eccentricity,mean_anomaly,inclination,argument_of_pericenter,longitude_of_ascending_node,r_,v_,parameters);
}

void elements_t::generate_elements(parameters_t *parameters)
{  
  /* Semimajor axes sampled from rho ~ r^(-gamma); eccentricities thermal; orbital angles random */
  double gamma=parameters->fp_gamma;
  if (parameters->fp_generation_mode == 1) // reject semimajor axis and eccenitricty if pericenter distance is less than user-specified minimum pericenter distance    
  {
    double pericenter_distance = 0.0;
    while (pericenter_distance < parameters->fp_minimum_pericenter_distance) 
    {
      semimajor_axis = pow( (pow(parameters->fp_maximum_semimajor_axis,3.0-gamma) - pow(parameters->fp_minimum_semimajor_axis,3.0-gamma))*random_number() \
        + pow(parameters->fp_minimum_semimajor_axis,3.0-gamma), 1.0/(3.0-gamma));
      eccentricity = sqrt(random_number());
      pericenter_distance = semimajor_axis*(1.0 - eccentricity);
    }
  }

  if (parameters->fp_generation_mode == 2) // reject semimajor axis and eccenitricty if GW inspiral time is less than Hubble time
  {
    double C_GR = (5.0/64.0)*pow(speed_of_light,5.0)*pow(gravitational_constant,-3.0)*pow(parameters->SMBH_mass,-2.0)*pow(parameters->fp_mass,-1.0); 
    double t_GR = 0.0,t_Hubble = 13.7e9,e2;
    while (t_GR < t_Hubble)
    {
      semimajor_axis = pow( (pow(parameters->fp_maximum_semimajor_axis,3.0-gamma) - pow(parameters->fp_minimum_semimajor_axis,3.0-gamma))*random_number() \
        + pow(parameters->fp_minimum_semimajor_axis,3.0-gamma), 1.0/(3.0-gamma));
      eccentricity = sqrt(random_number());
      e2 = eccentricity*eccentricity;
      t_GR = C_GR*pow(semimajor_axis,4.0)*pow(1.0-e2,7.0/2.0)/(1.0 + (73.0/24.0)*e2 + (37.0/96.0)*e2*e2); // high-e limit, cf. MAMW11 eq. 28
    }
  }
  mass = parameters->fp_mass;
  mean_anomaly = M_PI*(2.0*random_number()-1.0);
  inclination = acos(2.0*random_number() - 1.0);
  argument_of_pericenter = M_PI*(2.0*random_number()-1.0);
  longitude_of_ascending_node = M_PI*(2.0*random_number()-1.0);
}

/* tp_data */
tp_data_t::tp_data_t()
{
	captured = false;
}

void tp_data_t::set_pv(double r_[3], double v_[3])
{
  for (int c=0; c<3; c++)
  {
    r[c] = r_[c];
    v[c] = v_[c];
  }
}

/* fp_data */
void fp_data_t::set_pv(double r_[3], double v_[3])
{
  for (int c=0; c<3; c++)
  {
    r[c] = r_[c];
    v[c] = v_[c];
  }
}

void convert_elements_to_pv(std::vector<elements_t> *elements,std::vector<fp_data_t> *fp_data,parameters_t *parameters)
{
	int i_fp;
	
  #pragma omp parallel for if (parameters->enable_OpenMP == true && parameters->N_fp >= parameters->minimum_N_for_OpenMP) \
  shared(parameters,elements,fp_data) \
  private(i_fp) \
  default(none)	
	for (i_fp=0; i_fp<parameters->N_fp; i_fp++)
	{
	  convert_from_elements_to_pv
	  (
	    0.0, // time = 0
	    parameters->SMBH_mass + (*elements)[i_fp].mass,
	    (*elements)[i_fp].semimajor_axis,
	    (*elements)[i_fp].eccentricity,
	    (*elements)[i_fp].mean_anomaly,
	    (*elements)[i_fp].inclination,
	    (*elements)[i_fp].argument_of_pericenter,
	    (*elements)[i_fp].longitude_of_ascending_node,
	    (*fp_data)[i_fp].r,
	    (*fp_data)[i_fp].v,
	    parameters
	  );
	  (*fp_data)[i_fp].mass = (*elements)[i_fp].mass;
	}
}

void compute_fp_precession_rates(std::vector<elements_t> *elements, std::vector<fp_data_t> *fp_data, parameters_t *parameters)
{
	int i_fp,i;
  double SMBH_mass=parameters->SMBH_mass,mass,semimajor_axis,eccentricity,sqrtome2,gamma,enclosed_mass,mean_Schwarzschild_rate,mean_mass_precession_rate,total_rate,mass_precession_function,alpha;

  #pragma omp parallel for if (parameters->enable_OpenMP == true && parameters->N_fp >= parameters->minimum_N_for_OpenMP) \
  shared(parameters,elements,fp_data,SMBH_mass,cout) \
  private(i_fp,i,mass,semimajor_axis,eccentricity,enclosed_mass,sqrtome2,gamma,mean_Schwarzschild_rate,mean_mass_precession_rate,mass_precession_function,alpha,total_rate) \
  default(none)	
  for (i_fp=0; i_fp<parameters->N_fp; i_fp++)
  {
    mass = (*elements)[i_fp].mass;
    semimajor_axis = (*elements)[i_fp].semimajor_axis;
    eccentricity = (*elements)[i_fp].eccentricity;
    gamma = parameters->fp_gamma;  
      
    /* Schwarzschild precession */
    mean_Schwarzschild_rate = (3.0/(speed_of_light*speed_of_light))*pow(gravitational_constant*(SMBH_mass + mass),3.0/2.0)*pow(semimajor_axis,-5.0/2.0)/(1.0 - eccentricity*eccentricity);

    /* Mass precession */
    enclosed_mass = 0.0; // Determine enclosed field star mass for mass precession
    for (i=0; i<parameters->N_fp; i++)
    {
      //cout << semimajor_axis/206.265 << " " << (*elements)[i].semimajor_axis/206.265 << endl;
      if ((*elements)[i].semimajor_axis < semimajor_axis) { enclosed_mass += (*elements)[i].mass; }
    }

    /* Approximation to mass precession function G_M(e,gamma) (Merritt 2013 - Dynamics and Evolution of Galactic Nuclei - S. 4.4.1 p. 135-139) */
    sqrtome2 = sqrt(1.0 - eccentricity*eccentricity);
    if (gamma == 0.0) { mass_precession_function = c3div2; }
	  else if (gamma == 1.0) { mass_precession_function = 1.0; }
    else if (gamma == 2.0) { mass_precession_function = 1.0/(1.0 + sqrtome2); }
    else
    {
      if (eccentricity < 0.9) { alpha = c3div2 + gamma*(-c79div60 + gamma*(c7div20 - gamma*c1div30)); }
      else { alpha = c3div2 + gamma*(-c29div20 + gamma*(c11div20 - gamma*c1div10)) ; }
      mass_precession_function = (2.0/(2.0 - gamma))*alpha;
    }

    mean_mass_precession_rate = -sqrt(gravitational_constant*(SMBH_mass + mass)*pow(semimajor_axis,-3.0))*(enclosed_mass/SMBH_mass)*sqrtome2*mass_precession_function;
    //cout << eccentricity << " " << semimajor_axis << " " << mean_Schwarzschild_rate << " " << mean_mass_precession_rate << endl;
  
    if (parameters->include_1PN_terms == true) { total_rate = mean_Schwarzschild_rate + mean_mass_precession_rate; }
    else total_rate = mean_mass_precession_rate;

    (*fp_data)[i_fp].precession_rate = total_rate*parameters->fp_precession_factor;
  }  
}

void initialize_tp_data(std::vector<tp_data_t> *tp_data, std::vector<fp_data_t> *fp_data, parameters_t *parameters)
{
  std::vector<tp_calc_t> tp_calc(parameters->N_tp);	
  int i_tp,c;
  double r_check[3],v_check[3];
  
  #pragma omp parallel for if (parameters->enable_OpenMP == true && parameters->N_tp >= parameters->minimum_N_for_OpenMP) \
  shared(tp_data,tp_calc,fp_data,parameters,cout) \
  private(i_tp,c,r_check,v_check) \
  default(none)  
  for (i_tp = 0; i_tp<parameters->N_tp; i_tp++)
  {	
    transform_r_to_u_d0((*tp_data)[i_tp].r,(*tp_data)[i_tp].u_d0);
    transform_v_to_u_d1((*tp_data)[i_tp].u_d0,(*tp_data)[i_tp].v,(*tp_data)[i_tp].u_d1);

    transform_u_d0_to_r((*tp_data)[i_tp].u_d0,r_check);
    transform_u_d1_to_v((*tp_data)[i_tp].u_d0,(*tp_data)[i_tp].u_d1,v_check);

    #ifdef VERBOSE
      cout << "Transformation checks" << endl;
      cout << "before: r = " << (*tp_data)[i_tp].r[0]<< " " << (*tp_data)[i_tp].r[1]<< " " << (*tp_data)[i_tp].r[2] << "; v = " << (*tp_data)[i_tp].v[0] << " " << (*tp_data)[i_tp].v[1]<< " " << (*tp_data)[i_tp].v[2] << endl;
      cout << "after: r = " << r_check[0] << " " << r_check[1] << " " << r_check[2] << "; v = " << v_check[0] << " " << v_check[1] << " " << v_check[2] << endl;	
    #endif

    (*tp_data)[i_tp].h_d0 = tp_calc[i_tp].h_d0 = c1div2*norm3_squared((*tp_data)[i_tp].v) - gravitational_constant*parameters->SMBH_mass/norm3((*tp_data)[i_tp].r);

    for (c=0; c<3; c++)
    {
      tp_calc[i_tp].r[c] = (*tp_data)[i_tp].r_apo[c] = (*tp_data)[i_tp].r[c];
      tp_calc[i_tp].v[c] = (*tp_data)[i_tp].v_apo[c] = (*tp_data)[i_tp].v[c];
    }
    for (c=0; c<4; c++)
    {
      tp_calc[i_tp].u_d0[c] = (*tp_data)[i_tp].u_d0[c];
      tp_calc[i_tp].u_d1[c] = (*tp_data)[i_tp].u_d1[c];
    }
  }
  
  //cout << "pre gg " << tp_calc[0].u_d0[0] << " " << tp_calc[0].u_d1[0] << " " << tp_calc[0].u_d2[0] << " " << tp_calc[0].u_d3[0] << endl;
  get_gravity(parameters->N_tp,&tp_calc,fp_data,parameters);
  //cout << "post gg " << tp_calc[0].u_d0[0] << " " << tp_calc[0].u_d1[0] << " " << tp_calc[0].u_d2[0] << " " << tp_calc[0].u_d3[0] << endl;
  
  #pragma omp parallel for if (parameters->enable_OpenMP == true && parameters->N_tp >= parameters->minimum_N_for_OpenMP) \
  shared(tp_data,tp_calc,parameters,cout) \
  private(i_tp,c) \
  default(none)    
  for (i_tp=0; i_tp<parameters->N_tp; i_tp++)
  {
    for (c=0; c<4; c++)
    {
      (*tp_data)[i_tp].u_d2[c] = tp_calc[i_tp].u_d2[c];
      (*tp_data)[i_tp].u_d3[c] = tp_calc[i_tp].u_d3[c];
    }
    (*tp_data)[i_tp].h_d1 = tp_calc[i_tp].h_d1;
    (*tp_data)[i_tp].h_d2 = tp_calc[i_tp].h_d2;  
	
    (*tp_data)[i_tp].internal_time_step = time_step_function((*tp_data)[i_tp].u_d0,(*tp_data)[i_tp].u_d2,(*tp_data)[i_tp].u_d3,NULL,false,1e-2,parameters); // do not use u_d4; eta_factor = 1e-2 to ensure initial time-steps are sufficiently small

    (*tp_data)[i_tp].r_norm_hist = norm3((*tp_data)[i_tp].r);
    (*tp_data)[i_tp].t_apo = 0.0;
  }
}  

/*===========================================
  Kepler element conversion routines
  ===========================================*/
void convert_from_pv_to_elements
(
  double mass_A, double mass_B,
  double r[3],
  double v[3],
  double *semimajor_axis,
  double *eccentricity,
  double *true_anomaly,
  double *inclination,
  double *argument_of_pericenter,
  double *longitude_of_ascending_node,
  parameters_t *parameters
)
{
  double total_mass = mass_A + mass_B,GM = gravitational_constant*total_mass;
  double r_length = norm3(r),v_length = norm3(v),r_unit[3];
  int c=0;
  for (c=0; c<3; c++) { r_unit[c] = r[c]/r_length; }
  double GMdr = GM/r_length;
  double energy = c1div2*v_length*v_length - GMdr; // Specific Newtonian orbital energy
  double AM[3],AM_unit[3]; // Specific Newtonian orbital angular momentum
  cross3(r,v,AM);
  double AM_length = norm3(AM);
  for (c=0; c<3; c++) { AM_unit[c] = AM[c]/AM_length; }        

  /* Inclination: angle (rad) between the AM vector and the z-axis; 0 <= i < PI */
  double cos_inclination = AM[2]/AM_length;
  *inclination = acos(cos_inclination);

  /* Longitude of ascending node: angle (rad) between the ascending node vector and the reference direction, in this case chosen to be the x-axis; -PI <= Omega < PI */
  double ascending_node_vector[3],ascending_node_vector_unit[3],z_vector_unit[3] = {0.0,0.0,1.0};
  cross3(z_vector_unit,AM,ascending_node_vector);
  double ascending_node_vector_length = norm3(ascending_node_vector);
  if (ascending_node_vector_length == 0) { ascending_node_vector_unit[0] = 1.0; ascending_node_vector_unit[1] = ascending_node_vector_unit[2] = 0.0; }
  else{ for (c=0; c<3; c++) { ascending_node_vector_unit[c] = ascending_node_vector[c]/ascending_node_vector_length; } }
  *longitude_of_ascending_node = atan2(ascending_node_vector_unit[1], ascending_node_vector_unit[0]);

  /* Argument of pericenter: angle (rad) between the line of apsides/eccentricity vector and the line of nodes/ascending node vector; -PI <= omega < PI */
  double e_vector[3],e_vector_unit[3],v_cross_AM[3],e_cross_AM_unit[3];
  cross3(v,AM,v_cross_AM);
  for (c=0; c<3; c++) { e_vector[c] = (1.0/GM)*v_cross_AM[c] - r_unit[c]; }
  double e_vector_length = norm3(e_vector);
  for (c=0; c<3; c++) { e_vector_unit[c] = e_vector[c]/e_vector_length; }
  cross3(e_vector_unit,AM_unit,e_cross_AM_unit);
  *argument_of_pericenter = atan2(dot3(ascending_node_vector_unit,e_cross_AM_unit), dot3(ascending_node_vector_unit,e_vector_unit));
        
  /* True anomaly: angle (rad) between position vector and the line of apsides/eccentricity vector; -PI <= f < PI */
  *true_anomaly = atan2((-1.0)*dot3(r_unit,e_cross_AM_unit), dot3(r_unit,e_vector_unit));

  /* Semimajor axis and eccentricity */
  if (parameters->include_1PN_terms == true)
  {
    double nu = mass_A*mass_B/(total_mass*total_mass);
    double ninv = dot3(r_unit,v);
    double ninv2 = ninv*ninv;
    double v2 = v_length*v_length;
    double c2 = speed_of_light*speed_of_light;
    energy += c3div8*v2*v2/(c2)*(1.0 - 3.0*nu) + GMdr/(2.0*c2)*( (3.0 + nu)*v2 + nu*ninv2 + GMdr);
    AM_length *= (1.0 + v2/(2.0*c2)*(1.0 - 3.0*nu) + (3.0 + nu)*GMdr/(c2));
            
    *semimajor_axis = -GM/(2.0*energy)*( 1.0 - c1div2*(nu - 7.0)*energy/(c2) );
    *eccentricity = sqrt(1.0 + (2.0*energy)/(GM*GM)*(1.0 + c5div2*(nu - 3.0)*energy/(c2))*(AM_length*AM_length + (nu - 6.0)*GM*GM/(c2))); // this is the 'radial' eccentricity e_r
  }
  else
  {
    *semimajor_axis = -GM/(2.0*energy);
    *eccentricity = sqrt(1.0 + 2.0*AM_length*AM_length*energy/(GM*GM));
  }
}

void convert_from_elements_to_pv
(
  double time,
  double total_mass,
  double semimajor_axis,
  double eccentricity,
  double initial_mean_anomaly,
  double inclination,
  double argument_of_pericenter,
  double longitude_of_ascending_node,
  double r[3],
  double v[3],
  parameters_t * parameters
)
{
  double orbital_period = 2.0*PI*sqrt(pow(semimajor_axis,3.0)/(gravitational_constant*total_mass));
  double mean_anomaly = initial_mean_anomaly + 2.0*M_PI*time/orbital_period;
  double eccentric_anomaly = compute_eccentric_anomaly_from_mean_anomaly(mean_anomaly,eccentricity); 
  double cos_true_anomaly,sin_true_anomaly;
  compute_true_anomaly_from_eccentric_anomaly(eccentric_anomaly,eccentricity,&cos_true_anomaly,&sin_true_anomaly);
  double alpha[3],beta[3];
  compute_alpha_and_beta_from_orbital_angles(inclination,argument_of_pericenter,longitude_of_ascending_node,alpha,beta);

  double eccentricity_p2com = 1.0 - eccentricity*eccentricity;
  double separation = semimajor_axis*eccentricity_p2com/(1.0 + eccentricity*cos_true_anomaly);
  double velocity_tilde = sqrt(gravitational_constant*total_mass/(semimajor_axis*eccentricity_p2com)); // Common velocity factor
  for (int c=0; c<3; c++)
  {
    r[c] = separation*cos_true_anomaly*alpha[c] + separation*sin_true_anomaly*beta[c];
    v[c] = -1.0*velocity_tilde*sin_true_anomaly*alpha[c] + velocity_tilde*(eccentricity + cos_true_anomaly)*beta[c];
  }
}

double compute_eccentric_anomaly_from_mean_anomaly(double mean_anomaly, double eccentricity)
{
/* Newton-Raphson method (not the fastest way, but it is not used in the integrator) */

  double eccentric_anomaly_next = mean_anomaly;
  double epsilon = 1.0e-10;
  double error = 2.0*epsilon;
  double eccentric_anomaly;
  int j=0;
  while (error > epsilon || j < 15){
    j++;
    eccentric_anomaly = eccentric_anomaly_next;
    eccentric_anomaly_next = eccentric_anomaly - (eccentric_anomaly - eccentricity*sin(eccentric_anomaly) - mean_anomaly)/(1.0 - eccentricity*cos(eccentric_anomaly));
    error = fabs(eccentric_anomaly_next - eccentric_anomaly);
  }
  return eccentric_anomaly;
}

void compute_true_anomaly_from_eccentric_anomaly(double eccentric_anomaly, double eccentricity, double *cos_true_anomaly, double *sin_true_anomaly)
{
  *cos_true_anomaly = (cos(eccentric_anomaly) - eccentricity)/(1.0 - eccentricity*cos(eccentric_anomaly));
  *sin_true_anomaly = sqrt(1.0 - eccentricity*eccentricity)*sin(eccentric_anomaly)/(1.0 - eccentricity*cos(eccentric_anomaly));
}

void compute_alpha_and_beta_from_orbital_angles(double inclination, double argument_of_pericenter, double longitude_of_ascending_nodes, double *alpha, double *beta)
{
  double cos_inclination = cos(inclination),sin_inclination = sin(inclination);
  double cos_arg_per = cos(argument_of_pericenter),sin_arg_per = sin(argument_of_pericenter);
  double cos_long_asc_nodes = cos(longitude_of_ascending_nodes),sin_long_asc_nodes = sin(longitude_of_ascending_nodes);

  alpha[0] = cos_long_asc_nodes*cos_arg_per - sin_long_asc_nodes*sin_arg_per*cos_inclination;
  alpha[1] = sin_long_asc_nodes*cos_arg_per + cos_long_asc_nodes*sin_arg_per*cos_inclination;
  alpha[2] = sin_arg_per*sin_inclination;

  beta[0] = -cos_long_asc_nodes*sin_arg_per - sin_long_asc_nodes*cos_arg_per*cos_inclination;
  beta[1] = -sin_long_asc_nodes*sin_arg_per + cos_long_asc_nodes*cos_arg_per*cos_inclination;
  beta[2] = cos_arg_per*sin_inclination;
}


/*===========================================
  Kepler solver, developed by Atakan Gurkan
  ===========================================*/
  
double const tol = 1.0e-15;

long double G_func2(long double q) {
	int l = 3;
	int d = 15;
	int n = 0;
	long double A, B, G;
	
	if(q==0.0) return 1.0;	/* this isn't necessary when first
				   Newt-Raph iteration is done by hand */
	
	A = B = G = 1.0;

	while (fabs(B/G)>tol) {
		l += 2;
		d += 4*l;
		n += 10*l;

		A = d/(d-n*A*q);
		B *= A-1.0;
		G += B;

		l += 2;
		d += 4*l;
		n -= 8*l;

		A = d/(d-n*A*q);
		B *= A-1.0;
		G += B;
	}

	return G;
}

int nkep_drift3(double mu, double *r0, double *v0, double dt){
	long double r0mag, v0mag2;
	long double r0v0;	/* r dot v */
	long double rcalc, dtcalc, terr;
	long double u;	/* (?) universal variable */
	long double beta;	/* (?) vis-a-vis integral */
	long double P;	/* period (for elliptic orbits only) */
	long double dU;
	int n;
	long double q;
	long double U0w2, U1w2;
	long double U, U0, U1, U2, U3;
	long double f, g, F, G;
	long double r1[3], v1[3];
	int d;
	int no_iter;

	long double du1, du2, du3, dqdu, d2qdu2, drdu, d2rdu2, fn, fnp, fnpp, fnppp;

	r0mag  = sqrt(r0[0]*r0[0] + r0[1]*r0[1] + r0[2]*r0[2]);
	v0mag2 = v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2];
	r0v0   = r0[0]*v0[0] + r0[1]*v0[1] + r0[2]*v0[2];
	beta   = 2*mu/r0mag - v0mag2;

	if (beta > 0) {	/* elliptic orbit */
		P = 2*PI*mu/sqrt(beta*beta*beta);
		n = floor((dt + P/2.0 -2*r0v0/beta)/P);
		dU = 2*n*PI/sqrt(beta*beta*beta*beta*beta);
	} else {
		dU = 0.0;
	}

	u = 0;	/* a "better" guess is possible, see footnote at Battin p.219 */
//	u = dt/(4*r0mag); 				  /* N-R step by hand */
//	u = dt/(4*r0mag-2*dt*r0v0/r0mag);
//	u = init_u(dt, mu, r0mag, r0v0, beta);

	no_iter = 0;
	do {
		q = beta*u*u/(1+beta*u*u);
		if (q > 0.5 || no_iter > 12) return DRIFT_FAIL;
		dqdu = 2*beta*u/(1+beta*u*u)/(1+beta*u*u);
		d2qdu2 = 2*beta/(1+beta*u*u) 
		       - 8*beta*beta*u*u / (1+beta*u*u)/(1+beta*u*u)/(1+beta*u*u);
		U0w2 = 1 - 2*q;
		U1w2 = 2*(1-q)*u;
		U = 16.0/15 * U1w2*U1w2*U1w2*U1w2*U1w2 * G_func2(q) + dU;
		U0 = 2*U0w2*U0w2 - 1;
		U1 = 2*U0w2*U1w2;
		U2 = 2*U1w2*U1w2;
		U3 = beta*U + U1*U2/3.0;
		rcalc = r0mag*U0 + r0v0*U1 + mu*U2;
		drdu   = 4*(1-q)*(r0v0*U0 + (mu-beta*r0mag)*U1);
		d2rdu2 = -4*dqdu*(r0v0*U0 + (mu-beta*r0mag)*U1)
		       + (4*(1-q)*4*(1-q))*(-beta*r0v0*U1 + (mu-beta*r0mag)*U0);
		dtcalc = r0mag*U1 + r0v0*U2 + mu*U3;

		fn    = dtcalc-dt;
		fnp   = 4*(1-q)*rcalc;
		fnpp  = 4*(drdu*(1-q) - rcalc*dqdu);
		fnppp = -8*drdu*dqdu - 4*rcalc*d2qdu2 + 4*(1-q)*d2rdu2;

		du1  = -fn/fnp;
		du2  = -fn/(fnp + du1*fnpp/2);
		du3  = -fn/(fnp + du2*fnpp/2 + du2*du2*fnppp/6);

		u += du3;
		no_iter++;
		terr = fabs((dt-dtcalc)/dt);
	} while (terr > tol);
	
	f = 1 - (mu/r0mag)*U2;
	g = r0mag*U1 + r0v0*U2;
	F = -mu*U1/(rcalc*r0mag);
	G = 1 - (mu/rcalc)*U2;

	for(d=0; d<3; d++){
		r1[d] = f*r0[d] + g*v0[d];
		v1[d] = F*r0[d] + G*v0[d];
	}
	for(d=0; d<3; d++){
		r0[d] = r1[d];
		v0[d] = v1[d];
	}

	return DRIFT_SUCCESS;
}

/*===========================================
  I/O routines
  ===========================================*/
void read_parameters(parameters_t *parameters)
{
  double chi[3];
  ifstream input_file(parameters->input_filename.c_str());
  string line;
  if (input_file.is_open())
  {
    for (int line_number=1; line_number<12; line_number++) // read parameters (including tp header)
    {
      if (!input_file.fail())
      {
        getline (input_file,line);
        if (line[0] == '-' && line[1] == '-') 
        {
          if (line_number % 2 == 0)
          {
            if (line_number == 10) { continue; }
            else { cout << "Error reading parameters: only parameters on line 10 of input file can be omitted with \"--\". Terminating program. " << endl; exit(EXIT_FAILURE); }
          }
        }
        istringstream iss (line);        
        if (line_number == 2) { for (int i=0; i<10; i++)
        {
          if (i==0) iss >> parameters->N_tp; if (i==1) iss >> parameters->N_fp; if (i==2) iss >> parameters->end_time; if (i==3) iss >> parameters->number_of_time_steps;
          if (i==4) iss >> parameters->eta; if (i==5) iss >> parameters->save_state_at_apocenter; if (i==6) iss >> parameters->generate_field_elements; if (i==7) iss >> parameters->fp_generation_mode;
          if (i==8) iss >> parameters->use_pos_vel_input; if (i==9) iss >> parameters->write_fp_data;
        } }
        if (line_number == 4) { for (int i=0; i<8; i++) 
        {
          if (i==0) iss >> parameters->SMBH_mass; if (i==1) iss >> chi[0]; if (i==2) iss >> chi[1]; if (i==3) iss >> chi[2];
          if (i==4) iss >> parameters->detect_captures; if (i==5) iss >> parameters->tp_capture_radius; if (i==6) iss >> parameters->tp_minimum_semimajor_axis_for_plunge; if (i==7) iss >> parameters->tp_25PN_mass; 
        } }
        if (line_number == 6) { for (int i=0; i<7; i++)
        {
          if (i==0) iss >> parameters->fp_mass; if (i==1) iss >> parameters->fp_gamma; if (i==2) iss >> parameters->fp_minimum_semimajor_axis; if (i==3) iss >> parameters->fp_maximum_semimajor_axis;
          if (i==4) iss >> parameters->fp_minimum_pericenter_distance; if (i==5) iss >> parameters->fp_precession_factor; if (i==6) iss >> parameters->fp_random_seed;
        } }
        if (line_number == 8) { for (int i=0; i<4; i++)
        {
          if (i==0) iss >> parameters->include_1PN_terms; if (i==1) iss >> parameters->include_2PN_terms; if (i==2) iss >> parameters->include_25PN_terms; if (i==3) iss >> parameters->include_15PN_spin_terms;
          if (i==4) iss >> parameters->include_2PN_spin_terms;
        } }
        if (line_number == 10) { for (int i=0; i<6; i++)
        {
          if (i==0) iss >> parameters->enable_OpenMP; if (i==1) iss >> parameters->OpenMP_max_threads; if (i==2) iss >> parameters->minimum_N_for_OpenMP; if (i==3) iss >> parameters->enable_resume_functionality; if (i==4) iss >> parameters->N_resume; if (i==5) iss >> parameters->minimum_N_tp_calc_for_GPU;
        } }
      }
      else { cout << "Error reading parameters from input file. Terminating program. " << endl; exit(EXIT_FAILURE); }
    }
  }
  else { cout << "Input file not found. Terminating program. " << endl; exit(EXIT_FAILURE); }
  for (int c=0; c<3; c++) { parameters->chi[c] = chi[c]; }
  input_file.close();

  #ifdef VERBOSE
    cout << "Parsed input file. Parameters: " << endl;  
    cout << parameters->N_tp << " " << parameters->N_fp << " " << parameters->end_time << " " << parameters->number_of_time_steps << " " << parameters->eta << " " << parameters->save_state_at_apocenter << " " << parameters->generate_field_elements << " " << parameters->fp_generation_mode << " " << parameters->use_pos_vel_input << " " << parameters->write_fp_data << endl;
    cout << parameters->SMBH_mass << " " << parameters->chi[0] << " " << parameters->chi[1] << " " << parameters->chi[2] << " " << parameters->detect_captures << " " << parameters->tp_capture_radius << " " << parameters->tp_minimum_semimajor_axis_for_plunge << " " << parameters->tp_25PN_mass << endl;
    cout << parameters->fp_mass << " " << parameters->fp_gamma << " " << parameters->fp_minimum_semimajor_axis << " " << parameters->fp_maximum_semimajor_axis << " " << parameters->fp_minimum_pericenter_distance << " " << parameters->fp_precession_factor << endl;
    cout << parameters->include_1PN_terms << " " << parameters->include_2PN_terms << " " << parameters->include_25PN_terms << " " << parameters->include_15PN_spin_terms << " " << parameters->include_2PN_spin_terms << endl;
    cout << parameters->enable_OpenMP << " " << parameters->OpenMP_max_threads << " " << parameters->minimum_N_for_OpenMP << " " << parameters->enable_resume_functionality << " " << parameters->N_resume << endl;
  #endif
}  
  
void read_initial_particle_data(std::vector<tp_data_t> *tp_data, std::vector<fp_data_t> *fp_data, std::vector<elements_t> *tp_elements, std::vector<elements_t> *fp_elements,parameters_t *parameters)
{
  ifstream input_file(parameters->input_filename.c_str());
  std::string line;
  for (int line_number=1; line_number<12; line_number++) { getline (input_file,line); } // read parameters (including tp header)
  
  double m,a,e,ma,incl,ap,lan;
  double r[3],v[3];

  for (int i_tp=0; i_tp<parameters->N_tp; i_tp++) // read test particles
  {
    if (!input_file.fail())
    {
      getline (input_file,line);
      istringstream iss (line);
      if (parameters->use_pos_vel_input == false) // read elements from file and convert to positions and velocities
      {
        for (int i=0; i<6; i++) { if (i==0) iss >> a; if (i==1) iss >> e; if (i==2) iss >> ma; if (i==3) iss >> incl; if (i==4) iss >> ap; if (i==5) iss >> lan; }
        (*tp_elements)[i_tp].set_elements(0.0,a,e,ma,incl,ap,lan); // mass = 0
        (*tp_elements)[i_tp].convert_to_pv(0.0,(*tp_data)[i_tp].r,(*tp_data)[i_tp].v,parameters); // time = 0
      }
      else // read positions and velocities directly from file
      {
        for (int i=0; i<3; i++) { iss >> r[i]; }
        for (int i=0; i<3; i++) { iss >> v[i]; }        
        (*tp_data)[i_tp].set_pv(r,v);
      }
    }
    else { cout << "Error reading test particles from input file. Terminating program. " << endl; exit(EXIT_FAILURE); }
  }
  if (parameters->generate_field_elements == false) // read field elements if neccesary
  {
    getline (input_file,line); // read fp header
    for (int i_fp=0; i_fp<parameters->N_fp; i_fp++)
    {
      if (!input_file.fail())
      {
        getline (input_file,line);
        istringstream iss (line);
        if (parameters->use_pos_vel_input == false) // read elements from file and convert to positions and velocities
        {
          for (int i=0; i<6; i++) { if (i==0) iss >> m; if (i==1) iss >> a; if (i==2) iss >> e; if (i==3) iss >> ma; if (i==4) iss >> incl; if (i==5) iss >> ap; if (i==6) iss >> lan; }
          (*fp_elements)[i_fp].set_elements(m,a,e,ma,incl,ap,lan);
        }
        else // read masses, positions and velocities directly from file
        {
					iss >> (*fp_data)[i_fp].mass;
          for (int c=0; c<3; c++) { iss >> r[c]; }
          for (int c=0; c<3; c++) { iss >> v[c]; }
          (*fp_data)[i_fp].set_pv(r,v);
        }
      }
      else { cout << "Error reading field particles from input file. Terminating program. " << endl; exit(EXIT_FAILURE); }
    }
  }
  input_file.close();
}
  
void clear_output_files(parameters_t *parameters)
{
  std::string filename_pv;
  for (int i_tp=0; i_tp<parameters->N_tp; i_tp++)
  {
    filename_pv = parameters->output_filename + "_test_particle_" + IntToString(i_tp) + "_pos_vel.txt";
    ofstream file_pv(filename_pv.c_str());
    if (file_pv.fail()) { cout << "Error creating file for position and velocity data for test particle " << i_tp << endl; }
    file_pv.close();
  }
  if (parameters->write_fp_data == true)
  {
    for (int i_fp=0; i_fp<parameters->N_fp; i_fp++)
    {
      filename_pv = parameters->output_filename + "_field_particle_" + IntToString(i_fp) + "_pos_vel.txt";
      ofstream file_pv(filename_pv.c_str());
      if (file_pv.fail()) { cout << "Error creating file for position and velocity data for field particle " << i_fp << endl; }
      file_pv.close();
    }
  }
}

void analyze_output_files(parameters_t *parameters)
{
  std::string filename_pv,filename_el,line;
  double time,r[3],v[3];
  double a,e,f,incl,ap,lan;
  bool captured;
  for (int i_tp=0; i_tp<parameters->N_tp; i_tp++)
  {
    filename_pv = parameters->output_filename + "_test_particle_" + IntToString(i_tp) + "_pos_vel.txt";
    filename_el = parameters->output_filename + "_test_particle_" + IntToString(i_tp) + "_elements.txt";
    ifstream file_pv(filename_pv.c_str(),ios::in);
    ofstream file_el(filename_el.c_str());
    cout << "Processing file " << filename_pv << endl;
    if (!file_pv.fail() && !file_el.fail())
    {
      while (!file_pv.eof())
      {
        getline(file_pv,line);
        istringstream iss (line);
        iss >> time;
        for (int c=0; c<3; c++) { iss >> r[c]; }
        for (int c=0; c<3; c++) { iss >> v[c]; }
        iss >> captured;
        
        convert_from_pv_to_elements(0.0,parameters->SMBH_mass,r,v,&a,&e,&f,&incl,&ap,&lan,parameters); // m_TP = 0
		
        ostringstream oss;
        oss.precision(OUTPUT_PRECISION);
        oss.setf(ios::scientific);
        oss << time << " " << a << " " << e << " " << f << " " << incl << " " << ap << " " << lan << " " << captured << endl;
        file_el << oss.str();
      }
    }
    else { cout << "Error opening file for position and velocity data / element data for test particle " << i_tp << endl; }
    file_pv.close();
  }
}

void write_pv_to_disk(double time, std::vector<tp_data_t> *tp_data, std::vector<fp_data_t> *fp_data, parameters_t *parameters)
{
    
    

    
  std::string filename_pv;
  for (int i_tp=0; i_tp<parameters->N_tp; i_tp++)
  {
    filename_pv = parameters->output_filename + "_test_particle_" + IntToString(i_tp) + "_pos_vel.txt";


    //H5File file( filename_pv, H5F_ACC_RDONLY );
    //H5File file( filename_pv, H5F_ACC_TRUNC );



    ofstream file_pv(filename_pv.c_str(),ios::app);
    if (!file_pv.fail())
    {
      ostringstream oss;
      oss.precision(OUTPUT_PRECISION);
      oss.setf(ios::scientific);
      if (parameters->save_state_at_apocenter == true)
      {
        oss << (*tp_data)[i_tp].t_apo;
        for (int c=0; c<3; c++) { oss << " " << (*tp_data)[i_tp].r_apo[c]; }
        for (int c=0; c<3; c++) { oss << " " << (*tp_data)[i_tp].v_apo[c]; }
        oss << " " << (*tp_data)[i_tp].captured << endl;
      }
      else
      {
        oss << time;
        for (int c=0; c<3; c++) { oss << " " << (*tp_data)[i_tp].r[c]; }
        for (int c=0; c<3; c++) { oss << " " << (*tp_data)[i_tp].v[c]; }
        oss << " " << (*tp_data)[i_tp].captured << endl;
      }
      file_pv << oss.str();
    }
    else { cout << "Error opening file for position and velocity data for test particle " << i_tp << endl; }
    file_pv.close();
  }
  if (parameters->write_fp_data == true)
  {
    for (int i_fp=0; i_fp<parameters->N_fp; i_fp++)
    {
      filename_pv = parameters->output_filename + "_field_particle_" + IntToString(i_fp) + "_pos_vel.txt";
      ofstream file_pv(filename_pv.c_str(),ios::app);
      if (!file_pv.fail())
      {
        ostringstream oss;
        oss.precision(OUTPUT_PRECISION);
        oss.setf(ios::scientific);

        oss << time; oss << " " << (*fp_data)[i_fp].mass;
        for (int c=0; c<3; c++) { oss << " " << (*fp_data)[i_fp].r[c]; }
        for (int c=0; c<3; c++) { oss << " " << (*fp_data)[i_fp].v[c]; }
        oss << endl;
        file_pv << oss.str();
      }
      else { cout << "Error opening file for position and velocity data for field particle " << i_fp << endl; }
      file_pv.close();
    }
  }
}

void write_resume_data(double time, int loop_index, std::vector<tp_data_t> *tp_data, std::vector<fp_data_t> *fp_data, parameters_t *parameters)
{
  if (loop_index%parameters->N_resume == 0)
  {
    std::string filename;
    if (parameters->resume_filename.length() == 0) { filename = parameters->output_filename + "_resume_data_" + IntToString(loop_index) + ".txt"; }
    else { filename = parameters->resume_filename + "_resume_data_" + IntToString(loop_index) + ".txt"; }
		cout << "loop index = " << loop_index << " writing resume data" << endl;

	  ofstream file(filename.c_str());
	  if (!file.fail())
	  {
		  for (int i_tp=0; i_tp<parameters->N_tp; i_tp++)
		  {
        ostringstream oss;
        oss.precision(OUTPUT_PRECISION);
        oss.setf(ios::scientific);
        
        oss << time;
        for (int c=0; c<3; c++) { oss << " " << (*tp_data)[i_tp].r[c]; }
        for (int c=0; c<3; c++) { oss << " " << (*tp_data)[i_tp].v[c]; }
        oss << " " << (*tp_data)[i_tp].captured << endl;
        file << oss.str();
      }
      for (int i_fp=0; i_fp<parameters->N_fp; i_fp++)
      {
        ostringstream oss;
        oss.precision(OUTPUT_PRECISION);
        oss.setf(ios::scientific);
        
        oss << time; oss << " " << (*fp_data)[i_fp].mass;
        for (int c=0; c<3; c++) { oss << " " << (*fp_data)[i_fp].r[c]; }
        for (int c=0; c<3; c++) { oss << " " << (*fp_data)[i_fp].v[c]; }
        oss << " " << (*fp_data)[i_fp].precession_rate << endl;
        file << oss.str();
      }
    }
    else { cout << "Error opening file for resume data at loop_index =  " << loop_index << endl; }  
    file.close();
    cout << "loop index = " << loop_index << " finished writing resume data" << endl;
  }
}

void load_resume_data(double *time, int *loop_index, std::vector<tp_data_t> *tp_data, std::vector<fp_data_t> *fp_data, parameters_t *parameters)
{
  int resume_loop_index = *loop_index = parameters->resume_loop_index;
  std::string filename,line;
  if (parameters->resume_filename.length() == 0) { filename = parameters->output_filename + "_resume_data_" + IntToString(resume_loop_index) + ".txt"; }
  else { filename = parameters->resume_filename + "_resume_data_" + IntToString(resume_loop_index) + ".txt"; }

  cout << "loading resume data; resume_loop_index = " << resume_loop_index << endl;

  ifstream file(filename.c_str(),ios::in);
	if (!file.fail())
	{
    for (int i_tp=0; i_tp<parameters->N_tp; i_tp++)
		{
      getline(file,line);
      istringstream iss (line);
        
      iss >> *time;      
      for (int c=0; c<3; c++) { iss >> (*tp_data)[i_tp].r[c]; }
      for (int c=0; c<3; c++) { iss >> (*tp_data)[i_tp].v[c]; }
      iss >> (*tp_data)[i_tp].captured;
    }
    for (int i_fp=0; i_fp<parameters->N_fp; i_fp++)
    {
      getline(file,line);
      istringstream iss (line);

      iss >> *time; iss >> (*fp_data)[i_fp].mass;
      for (int c=0; c<3; c++) { iss >> (*fp_data)[i_fp].r[c]; }
      for (int c=0; c<3; c++) { iss >> (*fp_data)[i_fp].v[c]; }
      iss >> (*fp_data)[i_fp].precession_rate;
    }
  }
  else { cout << "Error loading resume data for resume_loop_index =  " << resume_loop_index << endl; }  
  file.close();
  cout << "finished loading resume data; resume t/yr = " << *time << endl;
}

void trim_output_files(double resume_time, parameters_t *parameters)
/* Removes all data from output files for times larger than the resume time, to prevent clogging of output files */
{
  std::string filename_in,filename_out,line;
  double time = 0;
  
  for (int i_tp=0; i_tp<parameters->N_tp; i_tp++)
  {
    filename_in = parameters->output_filename + "_test_particle_" + IntToString(i_tp) + "_pos_vel.txt";
    filename_out = parameters->output_filename + "_test_particle_" + IntToString(i_tp) + "_pos_vel_temp.txt";
    ifstream file_in(filename_in.c_str(),ios::in);
    ofstream file_out(filename_out.c_str());
    if (!file_in.fail() && !file_out.fail())
    {
      while (time < resume_time)
      {
        getline(file_in,line);
        istringstream iss (line);
        iss >> time;
        file_out << line << endl;
      }
      remove(filename_in.c_str());
      rename(filename_out.c_str(),filename_in.c_str());
    }
    else { cout << "Error opening file for position and velocity data for test particle " << i_tp << endl; }
    file_in.close();
    file_out.close();
  }
}
  
  
/*===========================================
  Assorted small routines
  ===========================================*/

std::string IntToString (int i)
{
  ostringstream ss;
  ss << i;
  return ss.str();
}

double random_number()
{
    double random_number = (double) rand()/((double) RAND_MAX);
    return random_number;
}

void transform_r_to_u_d0(double r[3], double u_d0[4])
{
  if (r[0] >= 0.0)
  {
    u_d0[0] = sqrt(c1div2*(r[0] + norm3(r)));
    u_d0[1] = c1div2*r[1]/u_d0[0];
    u_d0[2] = c1div2*r[2]/u_d0[0];
    u_d0[3] = 0.0;
  }
  else if (r[0] < 0.0)
  {
    u_d0[1] = sqrt(c1div2*(norm3(r) - r[0]));
    u_d0[0] = c1div2*r[1]/u_d0[1];
    u_d0[3] = c1div2*r[2]/u_d0[1];
    u_d0[2] = 0.0;
  }
}

void transform_v_to_u_d1(double u_d0[4], double v[3], double u_d1[4])
{
  u_d1[0] = c1div2*(u_d0[0]*v[0] + u_d0[1]*v[1] + u_d0[2]*v[2]);
  u_d1[1] = c1div2*(-u_d0[1]*v[0] + u_d0[0]*v[1] + u_d0[3]*v[2]);
  u_d1[2] = c1div2*(-u_d0[2]*v[0] - u_d0[3]*v[1] + u_d0[0]*v[2]);
  u_d1[3] = c1div2*(u_d0[3]*v[0] - u_d0[2]*v[1] + u_d0[1]*v[2]);
}

void transform_u_d0_to_r(double u_d0[4], double r[3])
{
  r[0] = u_d0[0]*u_d0[0] - u_d0[1]*u_d0[1] - u_d0[2]*u_d0[2] + u_d0[3]*u_d0[3];
  r[1] = 2.0*(u_d0[0]*u_d0[1] - u_d0[2]*u_d0[3]);
  r[2] = 2.0*(u_d0[0]*u_d0[2] + u_d0[1]*u_d0[3]);
}

void transform_u_d1_to_v(double u_d0[4], double u_d1[4], double v[3])
{
	double r_norm = norm4_squared(u_d0);
  v[0] = 2.0*(u_d0[0]*u_d1[0] - u_d0[1]*u_d1[1] - u_d0[2]*u_d1[2] + u_d0[3]*u_d1[3])/r_norm;
  v[1] = 2.0*(u_d0[1]*u_d1[0] + u_d0[0]*u_d1[1] - u_d0[3]*u_d1[2] - u_d0[2]*u_d1[3])/r_norm;
  v[2] = 2.0*(u_d0[2]*u_d1[0] + u_d0[3]*u_d1[1] + u_d0[0]*u_d1[2] + u_d0[1]*u_d1[3])/r_norm;
}

void LT_u_on_vec3(double u_d0[4], double vec3[3], double result[4])
{
  result[0] = u_d0[0]*vec3[0] + u_d0[1]*vec3[1] + u_d0[2]*vec3[2];
  result[1] = -u_d0[1]*vec3[0] + u_d0[0]*vec3[1] + u_d0[3]*vec3[2];
  result[2] = -u_d0[2]*vec3[0] - u_d0[3]*vec3[1] + u_d0[0]*vec3[2];
  result[3] = u_d0[3]*vec3[0] - u_d0[2]*vec3[1] + u_d0[1]*vec3[2];
}

double dot3(double a[3], double b[3])
{
    double result = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
    return result;
}
double dot4(double a[4], double b[4])
{
    double result = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]);
    return result;
}
double norm3(double v[3])
{
    double result = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    return result;
}
double norm3_squared(double v[4])
{
    double result = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    return result;
}
double norm4(double v[4])
{
    double result = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]);
    return result;
}
double norm4_squared(double v[4])
{
    double result = v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3];
    return result;
}   
void cross3(double a[3], double b[3], double result[3])
{
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2];
    result[2] = a[0]*b[1] - a[1]*b[0];
}

double pround(double x, int precision)
{
  if (x == 0.)
      return x;
  int ex = floor(log10(abs(x))) - precision + 1;
  double div = pow(10.0, (double) ex);
  return floor(x / div + 0.5) * div;
}

double elapsed_time(timespec *start, timespec *end)
{
  return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec)*1.0e-9;
}

void print_execution_times(parameters_t *parameters)
{/* 
  cout << "Printing execution times" << endl;
  cout << "Total: " << parameters->execution_times.total << endl;
  cout << "Shift fp: " << parameters->execution_times.shift_fp << endl;
  cout << "Rotate fp: " << parameters->execution_times.rotate_fp << endl;
  cout << "Determine i_tp_calc: " << parameters->execution_times.determine_i_tp_calc << endl;
  cout << "Predict: " << parameters->execution_times.predict << endl;
  cout << "Correct: " << parameters->execution_times.predict << endl;
  cout << "Evaluate: " << parameters->execution_times.predict << endl;
  cout << "GPU_set_j_particles: " << parameters->execution_times.GPU_set_j_particles << endl;
  cout << "Allocate_arrays_for_sapporo: " << parameters->execution_times.allocate_arrays_for_sapporo << endl;
  cout << "gg_fp: " << parameters->execution_times.gg_fp << endl;
  cout << "gg_PN_acc: " << parameters->execution_times.gg_PN_acc << endl;
  cout << "gg_PN_jerk: " << parameters->execution_times.gg_PN_jerk << endl;
*/
  cout << "Total execution time: " << parameters->execution_times.total << endl;
  cout << "Printing execution times + fractions of total execution time " << endl;
  cout <<  parameters->execution_times.shift_fp << " " << parameters->execution_times.shift_fp/parameters->execution_times.total << "\t shift fp" << endl;
  cout <<  parameters->execution_times.rotate_fp << " " << parameters->execution_times.rotate_fp/parameters->execution_times.total << "\t rotate fp" << endl;
  cout <<  parameters->execution_times.determine_i_tp_calc << " " << parameters->execution_times.determine_i_tp_calc/parameters->execution_times.total << "\t determine i_tp_calc" << endl;
  cout <<  parameters->execution_times.predict << " " << parameters->execution_times.predict/parameters->execution_times.total << "\t predict" << endl;
  cout <<  parameters->execution_times.correct << " " << parameters->execution_times.correct/parameters->execution_times.total<< "\t correct" << endl;
  cout <<  parameters->execution_times.evaluate << " " << parameters->execution_times.evaluate/parameters->execution_times.total << "\t evaluate" << endl;
  cout <<  parameters->execution_times.GPU_set_j_particles << " " << parameters->execution_times.GPU_set_j_particles/parameters->execution_times.total << "\t GPU_set_j_particles" << endl;
  cout <<  parameters->execution_times.allocate_arrays_for_sapporo << " " << parameters->execution_times.allocate_arrays_for_sapporo/parameters->execution_times.total << "\t allocate arrays for sapporo" << endl;
  cout <<  parameters->execution_times.gg_fp_CPU << " " << parameters->execution_times.gg_fp_CPU/parameters->execution_times.total << "\t gg_fp_CPU" << endl;
  cout <<  parameters->execution_times.gg_fp_GPU << " " << parameters->execution_times.gg_fp_GPU/parameters->execution_times.total << "\t gg_fp_GPU" << endl;  
  cout <<  parameters->execution_times.gg_PN_acc << " " << parameters->execution_times.gg_PN_acc/parameters->execution_times.total << "\t gg_PN_acc" << endl;
  cout <<  parameters->execution_times.gg_PN_jerk << " " << parameters->execution_times.gg_PN_jerk/parameters->execution_times.total << "\t gg_PN_jerk" << endl;
  cout << "Sum of above fractions: " << (parameters->execution_times.shift_fp + parameters->execution_times.rotate_fp + parameters->execution_times.determine_i_tp_calc \
    + parameters->execution_times.predict + parameters->execution_times.correct + parameters->execution_times.evaluate + parameters->execution_times.GPU_set_j_particles \
    + parameters->execution_times.allocate_arrays_for_sapporo + parameters->execution_times.gg_fp_CPU + parameters->execution_times.gg_fp_GPU + parameters->execution_times.gg_PN_acc \
    + parameters->execution_times.gg_PN_jerk)/parameters->execution_times.total << endl;
}
