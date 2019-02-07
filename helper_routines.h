//#include "hdf5.h"
//#include "H5Cpp.h"

/*===========================================
  Classes
  ===========================================*/

class execution_times_t
{
  public:
  double total,shift_fp,rotate_fp,determine_i_tp_calc,predict,correct,evaluate,GPU_set_j_particles,allocate_arrays_for_sapporo,gg_fp_CPU,gg_fp_GPU,gg_PN_acc,gg_PN_jerk;
};

class parameters_t
{
  public:
  double end_time,eta,tp_capture_radius,tp_minimum_semimajor_axis_for_plunge,tp_25PN_mass,SMBH_mass,chi[3],fp_mass,fp_gamma,fp_minimum_semimajor_axis,fp_maximum_semimajor_axis,fp_minimum_pericenter_distance,fp_precession_factor;
  int N_tp,N_fp,number_of_time_steps,fp_generation_mode,fp_random_seed;
  std::string input_filename,output_filename,resume_filename,capture_information_filename;
  bool save_state_at_apocenter,generate_field_elements,use_pos_vel_input,write_fp_data,detect_captures;
  bool include_1PN_terms,include_2PN_terms,include_25PN_terms,include_15PN_spin_terms,include_2PN_spin_terms;
  bool enable_resume_functionality,enable_OpenMP;
  int OpenMP_max_threads,minimum_N_for_OpenMP,minimum_N_tp_calc_for_GPU,N_resume,resume_loop_index;
  execution_times_t execution_times;
  parameters_t();
};

class elements_t
{
  public:
  double mass,semimajor_axis,eccentricity,mean_anomaly,true_anomaly,inclination,argument_of_pericenter,longitude_of_ascending_node;
  void set_elements(double mass_, double semimajor_axis_, double eccentricity_, double mean_anomaly_, double inclination_, double argument_of_pericenter_, double longitude_of_ascending_node_);
  void convert_to_pv(double time, double r_[3], double v_[3], parameters_t *parameters);
  void generate_elements(parameters_t *parameters);
};

class fp_data_t
{
  public:
  double r[3],v[3];
  double mass,precession_rate;
  void set_pv(double r_[3], double v_[3]);
};

void convert_elements_to_pv(std::vector<elements_t> *elements,std::vector<fp_data_t> *fp_data,parameters_t *parameters);
void compute_fp_precession_rates(std::vector<elements_t> *elements, std::vector<fp_data_t> *fp_data, parameters_t *parameters);

class tp_data_t
{
  public:
  double r[3],v[3],r_apo[3],v_apo[3];
  double u_d0[4],u_d1[4],u_d2[4],u_d3[4];
  double h_d0,h_d1,h_d2;
  double internal_time_step,r_norm_hist,t_apo;
  bool captured;
  
  tp_data_t();
  void set_pv(double r_[3], double v_[3]);
};

class tp_calc_t
{
  public:
  double r[3],v[3],*r_begin,*v_begin;
  double u_d0[4],u_d1[4],u_d2[4],u_d3[4];
  double *u_d0_begin,*u_d1_begin,*u_d2_begin,*u_d3_begin,u_d4_begin[4],u_d4_end[4];
  double h_d0,h_d1,h_d2,h_d0_begin,h_d1_begin,h_d2_begin;
  double a_PN_tp[3];
  double dtau,dtau2;
};

void initialize_tp_data(std::vector<tp_data_t> *tp_data, std::vector<fp_data_t> *fp_data, parameters_t *parameters);

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
);
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
);
double compute_eccentric_anomaly_from_mean_anomaly(double mean_anomaly, double eccentricity);
void compute_true_anomaly_from_eccentric_anomaly(double eccentric_anomaly, double eccentricity, double *cos_true_anomaly, double *sin_true_anomaly);
void compute_alpha_and_beta_from_orbital_angles(double inclination, double argument_of_pericenter, double longitude_of_ascending_nodes, double *alpha, double *beta);


/*===========================================
  Kepler solver, developed by Atakan Gurkan
  ===========================================*/

#define DRIFT_FAIL -1
#define DRIFT_SUCCESS 0

long double G_func2(long double q);
int nkep_drift3(double mu, double *r0, double *v0, double dt);


/*===========================================
  I/O routines
  ===========================================*/
void read_parameters(parameters_t *parameters);
void read_initial_particle_data(std::vector<tp_data_t> *tp_data, std::vector<fp_data_t> *fp_data, std::vector<elements_t> *tp_elements, std::vector<elements_t> *fp_elements,parameters_t *parameters);
void analyze_output_files(parameters_t *parameters);
void write_pv_to_disk(double time, std::vector<tp_data_t> *tp_data, std::vector<fp_data_t> *fp_data, parameters_t *parameters);
void write_resume_data(double time, int i_loop, std::vector<tp_data_t> *tp_data, std::vector<fp_data_t> *fp_data, parameters_t *parameters);
void load_resume_data(double *time, int *loop_index, std::vector<tp_data_t> *tp_data, std::vector<fp_data_t> *fp_data, parameters_t *parameters);
void trim_output_files(double resume_time, parameters_t *parameters);
void clear_output_files(parameters_t *parameters);

/*===========================================
  Assorted small routines
  ===========================================*/

double random_number();
std::string IntToString (int t);

void transform_r_to_u_d0(double r[3], double u_d0[4]);
void transform_v_to_u_d1(double u_d0[4], double v[3], double u_d1[4]);
void transform_u_d0_to_r(double u_d0[4], double r[3]);
void transform_u_d1_to_v(double u_d0[4], double u_d1[4], double v[3]);
void LT_u_on_vec3(double u_d0[4], double vec[3], double result[4]);
double dot3(double a[3], double b[3]);
double dot4(double a[4], double b[4]);
double norm3(double v[3]);
double norm3_squared(double v[3]);
double norm4(double v[4]);
double norm4_squared(double v[4]);
void cross3(double a[3], double b[3], double result[3]);
double elapsed_time(timespec *start, timespec *end);
double pround(double x, int precision);
void print_execution_times(parameters_t *parameters);
