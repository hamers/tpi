#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <exception>
#include <cmath>
#include <cstdlib>
#include <omp.h> // OpenMP
#include <unistd.h> // for option parsing
#include <ctime>


//#include <gsl/gsl_sf_hyperg.h>
//#include <math.h>

//#define VERBOSE
//#define DETERMINE_EXECUTION_TIMES
//#define USE_GPU

#define PI ((double) 3.14159265358979323846264338327950288419716939937510)
#define gravitational_constant ((double) 4.0*PI*PI)
#define speed_of_light ((double) 63241.1)

#define OUTPUT_PRECISION 15
#define ABSOLUTE_MINIMUM_INTERNAL_TIME_STEP ((double) 1.0e-14)

/* Constants to speed up frequently used code */
double const c1div2 = 1.0/2.0;
double const c1div6 = 1.0/6.0;
double const c1div10 = 1.0/10.0;
double const c1div24 = 1.0/24.0;
double const c1div30 = 1.0/30.0;
double const c1div120 = 1.0/120.0;
double const c3div2 = 3.0/2.0;
double const c3div8 = 3.0/8.0;
double const c5div2 = 5.0/2.0;
double const c7div2 = 7.0/2.0;
double const c7div20 = 7.0/20.0;
double const c8div5 = 8.0/5.0;
double const c11div20 = 11.0/20.0;
double const c17div3 = 17.0/3.0;
double const c29div20 = 29.0/20.0;
double const c53div3 = 53.0/3.0;
double const c79div60 = 79.0/60.0;



#include "helper_routines.h"

void integrator(double time, double time_step, std::vector<tp_data_t> *tp_data, std::vector<fp_data_t> *fp_data, parameters_t *parameters);
double time_step_function(double u_d0[4], double u_d2[4], double u_d3[4], double u_d4[4], bool use_u_d4, double eta_factor, parameters_t *parameters);
void determine_apocenter_function(double global_internal_time, double r_begin[3], double r_end[3], double v_begin[3], double v_end[3], tp_data_t *tp_data, parameters_t *parameters);
void capture_check_function(double global_internal_time, double r_begin[3], double r_end[3], double v_begin[3], double v_end[3], tp_data_t *tp_data, parameters_t *parameters);
void shift_field_particles(double delta_time,std::vector<fp_data_t> *fp_data_t, parameters_t *parameters);
void rotate_field_particles(double delta_time, std::vector<fp_data_t> *fp_data_t, parameters_t *parameters);
void get_gravity(int N_tp_calc, std::vector<tp_calc_t> *tp_calc, std::vector<fp_data_t> *fp_data, parameters_t *parameters);
void GPU_set_j_particles(std::vector<fp_data_t> *fp_data, parameters_t *parameters);
void get_fp_gravity_GPU_start(int N_tp_calc,std::vector<tp_calc_t> *tp_calc,std::vector<fp_data_t> *fp_data,parameters_t *parameters, int *ids, double (*r)[3], double (*v)[3], double *h2, double *pot, double (*a_fp_on_tp)[3], double (*j_fp_on_tp)[3]);
void get_fp_gravity_GPU_retrieve(int N_tp_calc,std::vector<tp_calc_t> *tp_calc,std::vector<fp_data_t> *fp_data,parameters_t *parameters, int *ids, double (*r)[3], double (*v)[3], double *h2, double *pot, double (*a_fp_on_tp)[3], double (*j_fp_on_tp)[3]);
void get_fp_gravity_CPU(int N_tp_calc, std::vector<tp_calc_t> *tp_calc, std::vector<fp_data_t> *fp_data, parameters_t *parameters, double (*a_fp_on_tp)[3], double (*j_fp_on_tp)[3]);
void get_PN_acc(int N_tp_calc, std::vector<tp_calc_t> *tp_calc, parameters_t *parameters);
void get_PN_jerk(int N_tp_calc, std::vector<tp_calc_t> *tp_calc, parameters_t *parameters, double (*a_fp_on_tp_)[3], double (*j_fp_on_tp_)[3]);
