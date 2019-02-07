#include "tpi.h"

#ifdef USE_GPU
#include "g6lib.h"
#endif

using namespace std;

int main(int argc, char *argv[])
{
  cout << "Test Particle Integrator. Version 0.22 (August 2013). Implemented: OpenMP; Sapporo2. " << endl;
  
/*==================
  Input file parsing
  ==================*/
  
  string input_filename,output_filename,resume_filename;
  int opt,mode = 1,resume_loop_index;
  bool entered_input_filename = false,entered_output_filename = false;
  if (argc == 1) { cout << "No command line arguments. Terminating program. " << endl; exit(EXIT_FAILURE); }
  else
  {
		while ((opt = getopt(argc, argv, "car:i:o:f:")) != -1)
		{
			switch (opt)
			{
				case 'c': mode = 1; break;
				case 'a': mode = 2; break;
				case 'r': mode = 3; resume_loop_index = atoi(optarg); break;
				case 'i': input_filename = optarg; entered_input_filename = true; break;
				case 'o': output_filename = optarg; entered_output_filename = true; break;
				case 'f': resume_filename = optarg; break;
      }
    }
    if (! (entered_input_filename || entered_output_filename) ) { cout << "No input/output filenames specified. Terminating program. " << endl; exit(EXIT_FAILURE); }
  }
  
  parameters_t parameters;
  parameters.input_filename = input_filename;
  parameters.output_filename = output_filename;
  parameters.resume_filename = resume_filename;
  parameters.resume_loop_index = resume_loop_index;
  
  read_parameters(&parameters);

  if (parameters.enable_OpenMP == true) 
  {
	omp_set_num_threads(parameters.OpenMP_max_threads);
	cout << "OpenMP is enabled; using at most " << parameters.OpenMP_max_threads << " threads; active if N_loop > " << parameters.minimum_N_for_OpenMP << endl;
  }
  #ifdef USE_GPU
    //cout << "Sapporo 2 is enabled; using GPU if N_tp_calc > " << parameters.minimum_N_tp_calc_for_GPU << endl;
    cout << "Sapporo 2 is enabled. " << endl;
  #endif

  std::vector<tp_data_t> tp_data(parameters.N_tp);
  std::vector<fp_data_t> fp_data(parameters.N_fp);
  std::vector<elements_t> tp_elements(parameters.N_tp);
  std::vector<elements_t> fp_elements(parameters.N_fp);
  
  read_initial_particle_data(&tp_data,&fp_data,&tp_elements,&fp_elements,&parameters);
  
  /*=========================
  Analyse pv data from disk
  =========================*/  
  
  if (mode == 2) { analyze_output_files(&parameters); exit(EXIT_SUCCESS); }
  if (mode == 1) { clear_output_files(&parameters); }


/*=======================
  Initialize tp & fp data
  =======================*/  
  
  double time = 0.0;
  double time_step = parameters.end_time/((double) parameters.number_of_time_steps);
  int loop_index = 0;
  
  if (parameters.generate_field_elements == true)
  {
    for (int i_fp=0; i_fp<parameters.N_fp; i_fp++)
    { 
      fp_elements[i_fp].generate_elements(&parameters);
    }
  }
  if (parameters.use_pos_vel_input == false)
  {
    convert_elements_to_pv(&fp_elements,&fp_data,&parameters);
    compute_fp_precession_rates(&fp_elements,&fp_data,&parameters);
  }
  else { for (int i_fp=0; i_fp<parameters.N_fp; i_fp++) { fp_data[i_fp].precession_rate = 0.0; } }
  
  if (mode == 3) { load_resume_data(&time,&loop_index,&tp_data,&fp_data,&parameters); trim_output_files(time,&parameters); } // above initialization will be overridden

  #ifdef USE_GPU
    #ifdef VERBOSE
      cout << "open GPU" << endl;
    #endif

    int cluster_id=0;
    g6_open_(&cluster_id);

    GPU_set_j_particles(&fp_data,&parameters);
    
    g6_set_ti(0, 0.0);

    #ifdef VERBOSE
      cout << "open GPU - done" << endl;
    #endif  
  #endif

  initialize_tp_data(&tp_data,&fp_data,&parameters);

/*===========
  Integration
  ===========*/  
 
  cout << "Initialization complete, now starting integration. " << endl;

  #ifdef DETERMINE_EXECUTION_TIMES
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC,&start);
  #endif

  while (time < parameters.end_time)
  {
    integrator(time,time_step,&tp_data,&fp_data,&parameters);
    time += time_step;
    
    cout << "t/yr = " << time << "; r_tp[0]/AU = {" << tp_data[0].r[0] << "," << tp_data[0].r[1] << "," << tp_data[0].r[2] << "}" << endl;
    write_pv_to_disk(time,&tp_data,&fp_data,&parameters);
    if (parameters.enable_resume_functionality == true) { write_resume_data(time,loop_index,&tp_data,&fp_data,&parameters); }
    loop_index++;  
  }

  #ifdef USE_GPU
    g6_close_(&cluster_id);
  #endif

  #ifdef DETERMINE_EXECUTION_TIMES
    clock_gettime(CLOCK_MONOTONIC,&end);
    parameters.execution_times.total = elapsed_time(&start,&end);
    print_execution_times(&parameters);
  #endif
  
  exit(EXIT_SUCCESS);
}

/*=========================
  Main integration function
  =========================*/ 
  
void integrator(double time, double time_step, std::vector<tp_data_t> *tp_data, std::vector<fp_data_t> *fp_data, parameters_t *parameters)
{
  #ifdef DETERMINE_EXECUTION_TIMES
    timespec start,end;
  #endif
	
  int i_tp,i_tp_calc,N_tp_calc,N_tp_captured,c,integrator_loop_index=0,N_tp = parameters->N_tp;
  std::vector<double> next_internal_times(N_tp);
  for (i_tp=0; i_tp<N_tp; i_tp++) { next_internal_times[i_tp] = time + (*tp_data)[i_tp].internal_time_step; }
  double global_internal_time = time,global_internal_time_step,minimum_next_internal_time,next_internal_time;
  double next_internal_times_rounded,global_internal_time_rounded;
  bool synchronize = false;

  double dt,dtau,dtau2;
  double u_d0_begin_norm2,u4_num,u5_num,h3_num,h4_num;  
  double internal_time_step_begin,internal_time_step_end,internal_time_step;
  
  std::vector<int> indices_tp_calc; // int vector containing ids of the test particles that are to be integrated in the global internal time step
  std::vector<tp_calc_t> tp_calc;
    
  while(global_internal_time < time + time_step)
  {
    integrator_loop_index += 1;

    #ifdef VERBOSE
      cout << "integrator loop index = " << integrator_loop_index << endl;
    #endif

    #ifdef VERBOSE
      cout << "t_int = " << global_internal_time << "; begin of while; tp 0: r[0] = " << (*tp_data)[0].r[0] << endl;
      //for (int k=0; k<N_tp; k++) { cout << "next_internal_times[" << k << "] = " << next_internal_times[k] << endl; }   
    #endif

/*  ========================================
    Determine next global internal time step
    ========================================  */
    
    indices_tp_calc.clear();
    indices_tp_calc.reserve(N_tp);
    tp_calc.clear();

    minimum_next_internal_time = time + time_step; // starting value: largest possible value of internal time step
    next_internal_time = minimum_next_internal_time;
    for (i_tp=0; i_tp<N_tp; i_tp++) // determine next global internal time and global internal time step
    {
      if ((*tp_data)[i_tp].captured == false) { next_internal_time = next_internal_times[i_tp]; } // only include non-captured test particles
      if (next_internal_time < minimum_next_internal_time) { minimum_next_internal_time = next_internal_time; } // determine minimum of next internal times
    }
    global_internal_time_step = minimum_next_internal_time - global_internal_time;

    if ((global_internal_time + global_internal_time_step) >= (time + time_step)) // t + dt reached, i.e. end of output time interval
    {
      global_internal_time_step = time + time_step - global_internal_time;
      global_internal_time = time + time_step;
      synchronize = true;
    }
    else
    {
      global_internal_time += global_internal_time_step;
      synchronize = false;
    }
    
    #ifdef VERBOSE
      cout << "t_int = " << global_internal_time << "; determined global internal time step dt_int = " << global_internal_time_step << "; synchronize = " << synchronize << endl;
    #endif

/*  ================================
    Shift and rotate field particles
    ================================  */

    #ifdef USE_GPU
      g6_set_ti(0, global_internal_time);
    #else
      shift_field_particles(global_internal_time_step,fp_data,parameters);
      rotate_field_particles(global_internal_time_step,fp_data,parameters);
    #endif
    
/*  ===================================================
    Determine which test particles are to be integrated
    ===================================================  */
    
    #ifdef DETERMINE_EXECUTION_TIMES
      clock_gettime(CLOCK_MONOTONIC,&start);
    #endif

    N_tp_captured=0;

    for (i_tp=0; i_tp<N_tp; i_tp++) // parallelize?
    {
      #ifdef VERBOSE
        //cout.precision(15);
        //cout << "t_int = " << global_internal_time << "; determining tp calc for i_tp = " << i_tp << "; t_int = " << global_internal_time << "; t_next = " << next_internal_times[i_tp] << "; captured = " << (*tp_data)[i_tp].captured << endl;
      #endif
      
      next_internal_times_rounded = pround(next_internal_times[i_tp],15);
      global_internal_time_rounded = pround(global_internal_time,15);

      if ((*tp_data)[i_tp].captured == true) { N_tp_captured++; continue; }
      if ((next_internal_times_rounded < global_internal_time_rounded) && ((*tp_data)[i_tp].captured == false) && (synchronize == false)) { cout << "Internal time error. Terminating program. " << endl; exit(EXIT_FAILURE); continue; }
      if ((next_internal_times_rounded == global_internal_time_rounded || synchronize == true) && ((*tp_data)[i_tp].captured == false)) { indices_tp_calc.push_back(i_tp); }
    }

    if (N_tp_captured == N_tp) { cout << "All test particles have been captured. " << endl; break; }
		if (indices_tp_calc.empty() == true) { cout << "Error: indices_tp_calc is empty. Terminating program. " << endl; exit(EXIT_FAILURE); }
    N_tp_calc = indices_tp_calc.size();

    #ifdef VERBOSE
      cout << "t_int = " << global_internal_time << "; determined indices tp_calc; N_tp_calc = " << N_tp_calc << "; indices_tp_calc = {" << indices_tp_calc[0] << " " << indices_tp_calc[1] << " " << indices_tp_calc[2] << " ...}" << endl;
    #endif    

    //tp_calc.reserve(N_tp_calc);
	  tp_calc.resize(N_tp_calc); // think about this

    #ifdef DETERMINE_EXECUTION_TIMES
      clock_gettime(CLOCK_MONOTONIC,&end);
      parameters->execution_times.determine_i_tp_calc += elapsed_time(&start,&end);
    #endif

/*  =======
    Predict
    =======  */
    
    #ifdef DETERMINE_EXECUTION_TIMES
      clock_gettime(CLOCK_MONOTONIC,&start);      
    #endif

    #pragma omp parallel for if (parameters->enable_OpenMP == true && N_tp_calc >= parameters->minimum_N_for_OpenMP) \
    shared(N_tp_calc,indices_tp_calc,tp_calc,tp_data,synchronize,global_internal_time,next_internal_times) \
    private(i_tp_calc,i_tp,c,dt,u_d0_begin_norm2,dtau,dtau2) \
    default(none)
    for (i_tp_calc=0; i_tp_calc<N_tp_calc; i_tp_calc++)
    {
      i_tp = indices_tp_calc[i_tp_calc];

      tp_calc[i_tp_calc].r_begin = (*tp_data)[i_tp].r;
	    tp_calc[i_tp_calc].v_begin = (*tp_data)[i_tp].v;

      tp_calc[i_tp_calc].u_d0_begin = (*tp_data)[i_tp].u_d0;
      tp_calc[i_tp_calc].u_d1_begin = (*tp_data)[i_tp].u_d1;
      tp_calc[i_tp_calc].u_d2_begin = (*tp_data)[i_tp].u_d2;
      tp_calc[i_tp_calc].u_d3_begin = (*tp_data)[i_tp].u_d3;

      tp_calc[i_tp_calc].h_d0_begin = (*tp_data)[i_tp].h_d0;
      tp_calc[i_tp_calc].h_d1_begin = (*tp_data)[i_tp].h_d1;
      tp_calc[i_tp_calc].h_d2_begin = (*tp_data)[i_tp].h_d2;

      if (synchronize == false) { dt = (*tp_data)[i_tp].internal_time_step; }
      else { dt = global_internal_time + (*tp_data)[i_tp].internal_time_step - next_internal_times[i_tp]; }
      u_d0_begin_norm2 = norm4_squared(tp_calc[i_tp_calc].u_d0_begin);      
      dtau = tp_calc[i_tp_calc].dtau = dt/u_d0_begin_norm2;
      dtau2 = tp_calc[i_tp_calc].dtau2 = dtau*dtau;

      for (c=0; c<4; c++)
      {
        tp_calc[i_tp_calc].u_d0[c] = tp_calc[i_tp_calc].u_d0_begin[c] + dtau*(tp_calc[i_tp_calc].u_d1_begin[c] + dtau*(c1div2*tp_calc[i_tp_calc].u_d2_begin[c] + dtau*c1div6*tp_calc[i_tp_calc].u_d3_begin[c])); // definition of constants in tpi.h
        tp_calc[i_tp_calc].u_d1[c] = tp_calc[i_tp_calc].u_d1_begin[c] + dtau*(tp_calc[i_tp_calc].u_d2_begin[c] + dtau*c1div2*tp_calc[i_tp_calc].u_d3_begin[c]);
      }
      tp_calc[i_tp_calc].h_d0 = tp_calc[i_tp_calc].h_d0_begin + dtau*(tp_calc[i_tp_calc].h_d1_begin + dtau*c1div2*tp_calc[i_tp_calc].h_d2_begin);

      transform_u_d0_to_r(tp_calc[i_tp_calc].u_d0,tp_calc[i_tp_calc].r);
      transform_u_d1_to_v(tp_calc[i_tp_calc].u_d0,tp_calc[i_tp_calc].u_d1,tp_calc[i_tp_calc].v);
    }

    #ifdef VERBOSE
      cout << "t_int = " << global_internal_time << "; done prediction, starting first get_gravity" << endl;
    #endif

    #ifdef DETERMINE_EXECUTION_TIMES
      clock_gettime(CLOCK_MONOTONIC,&end);
      parameters->execution_times.predict += elapsed_time(&start,&end);
    #endif

/*  ========
    Evaluate
    ========  */
    
    get_gravity(N_tp_calc,&tp_calc,fp_data,parameters);

    #ifdef VERBOSE
      cout << "t_int = " << global_internal_time << "; done first get_gravity" << endl;
    #endif      

/*  =======
    Correct
    =======  */
    
    #ifdef DETERMINE_EXECUTION_TIMES
      clock_gettime(CLOCK_MONOTONIC,&start);
    #endif
    
    #pragma omp parallel for if (parameters->enable_OpenMP == true && N_tp_calc >= parameters->minimum_N_for_OpenMP) \
    shared(N_tp_calc,indices_tp_calc,tp_calc,tp_data,global_internal_time,parameters,cout) \
    private(i_tp_calc,i_tp,c,dtau,dtau2,u4_num,u5_num,h3_num,h4_num) \
    default(none)
    for (i_tp_calc=0; i_tp_calc<N_tp_calc; i_tp_calc++)
    {
      i_tp = indices_tp_calc[i_tp_calc];     
      dtau = tp_calc[i_tp_calc].dtau;
      dtau2 = tp_calc[i_tp_calc].dtau2; 
      for (c=0; c<4; c++)
      {
        u4_num = 6.0*(tp_calc[i_tp_calc].u_d2[c] - tp_calc[i_tp_calc].u_d2_begin[c]) - (4.0*tp_calc[i_tp_calc].u_d3_begin[c] + 2.0*tp_calc[i_tp_calc].u_d3[c])*dtau;
        u5_num = 12.0*(tp_calc[i_tp_calc].u_d2_begin[c] - tp_calc[i_tp_calc].u_d2[c]) + 6.0*(tp_calc[i_tp_calc].u_d3_begin[c] + tp_calc[i_tp_calc].u_d3[c])*dtau;
        tp_calc[i_tp_calc].u_d0[c] = (*tp_data)[i_tp].u_d0[c] = tp_calc[i_tp_calc].u_d0[c] + dtau2*(c1div24*u4_num + c1div120*u5_num);
        tp_calc[i_tp_calc].u_d1[c] = (*tp_data)[i_tp].u_d1[c] = tp_calc[i_tp_calc].u_d1[c] + dtau*(c1div6*u4_num + c1div24*u5_num);

        tp_calc[i_tp_calc].u_d4_begin[c] = u4_num/dtau2;
        tp_calc[i_tp_calc].u_d4_end[c] = tp_calc[i_tp_calc].u_d4_begin[c] + u5_num/dtau2;
      }
      h3_num = 6.0*(tp_calc[i_tp_calc].h_d1 - tp_calc[i_tp_calc].h_d1_begin) - (4.0*tp_calc[i_tp_calc].h_d2_begin + 2.0*tp_calc[i_tp_calc].h_d2)*dtau;
      h4_num = 12.0*(tp_calc[i_tp_calc].h_d1_begin - tp_calc[i_tp_calc].h_d1) + 6.0*(tp_calc[i_tp_calc].h_d2_begin + tp_calc[i_tp_calc].h_d2)*dtau;

      tp_calc[i_tp_calc].h_d0 = (*tp_data)[i_tp].h_d0 = tp_calc[i_tp_calc].h_d0 + dtau*(c1div6*h3_num + c1div24*h4_num);

      transform_u_d0_to_r(tp_calc[i_tp_calc].u_d0,tp_calc[i_tp_calc].r);
      transform_u_d1_to_v(tp_calc[i_tp_calc].u_d0,tp_calc[i_tp_calc].u_d1,tp_calc[i_tp_calc].v);

      determine_apocenter_function(global_internal_time,tp_calc[i_tp_calc].r_begin, tp_calc[i_tp_calc].r, tp_calc[i_tp_calc].v_begin, tp_calc[i_tp_calc].v, &((*tp_data)[i_tp]),parameters);
      if (parameters->detect_captures == true)
      {
        capture_check_function(global_internal_time, tp_calc[i_tp_calc].r_begin, tp_calc[i_tp_calc].r, tp_calc[i_tp_calc].v_begin, tp_calc[i_tp_calc].v, &((*tp_data)[i_tp]),parameters);
        if ((*tp_data)[i_tp].captured == true) { cout << "Capture of test particle " << i_tp << " at t_int/yr = " << global_internal_time << "; r/AU = {" << (*tp_data)[i_tp].r[0] << "," << (*tp_data)[i_tp].r[1] << "," << (*tp_data)[i_tp].r[2] << "}." << endl; }
      }
    }

    #ifdef VERBOSE
      cout << "t_int = " << global_internal_time << "; done correction, starting second get_gravity" << endl;
    #endif

    #ifdef DETERMINE_EXECUTION_TIMES
      clock_gettime(CLOCK_MONOTONIC,&end);
      parameters->execution_times.correct += elapsed_time(&start,&end);      
    #endif
      
/*  =================================================================
    Re-evaluate, determine next time-step and save results to tp_data
    =================================================================  */
    
    get_gravity(N_tp_calc,&tp_calc,fp_data,parameters);

    #ifdef VERBOSE
      cout << "t_int = " << global_internal_time << "; done second get_gravity" << endl;
    #endif   

    #ifdef DETERMINE_EXECUTION_TIMES
      clock_gettime(CLOCK_MONOTONIC,&start);
    #endif

    #pragma omp parallel for if (parameters->enable_OpenMP == true && N_tp_calc >= parameters->minimum_N_for_OpenMP) \
    shared(N_tp_calc,indices_tp_calc,tp_calc,tp_data,global_internal_time,next_internal_times,parameters) \
    private(i_tp_calc,i_tp,c,dt,dtau,dtau2,u4_num,u5_num,h3_num,h4_num,internal_time_step_begin,internal_time_step_end,internal_time_step) \
    default(none)
    for (i_tp_calc=0; i_tp_calc<N_tp_calc; i_tp_calc++)
    {
      i_tp = indices_tp_calc[i_tp_calc];
      for (c=0; c<3; c++) // do not copy just the pointer because tp_calc will be cleared in the next iteration
      {
        (*tp_data)[i_tp].r[c] = tp_calc[i_tp_calc].r[c];
        (*tp_data)[i_tp].v[c] = tp_calc[i_tp_calc].v[c];
      }          
      for (c=0; c<4; c++) // u_d0 and u_d1 were copied earlier
      {
			  (*tp_data)[i_tp].u_d2[c] = tp_calc[i_tp_calc].u_d2[c];
        (*tp_data)[i_tp].u_d3[c] = tp_calc[i_tp_calc].u_d3[c];
      }
      (*tp_data)[i_tp].h_d1 = tp_calc[i_tp_calc].h_d1;
      (*tp_data)[i_tp].h_d2 = tp_calc[i_tp_calc].h_d2;

      internal_time_step_begin = time_step_function(tp_calc[i_tp_calc].u_d0_begin, tp_calc[i_tp_calc].u_d2_begin, tp_calc[i_tp_calc].u_d3_begin, tp_calc[i_tp_calc].u_d4_begin,true,1.0,parameters); // use u_d4; eta_factor = 1
      internal_time_step_end = time_step_function(tp_calc[i_tp_calc].u_d0, tp_calc[i_tp_calc].u_d2, tp_calc[i_tp_calc].u_d3, tp_calc[i_tp_calc].u_d4_end, true,1.0,parameters); // use u_d4; eta_factor = 1
      internal_time_step = c1div2*(internal_time_step_begin + internal_time_step_end); // time-symmetric time-steps
      (*tp_data)[i_tp].internal_time_step = internal_time_step;
      next_internal_times[i_tp] = global_internal_time + internal_time_step;
    }

    #ifdef DETERMINE_EXECUTION_TIMES
      clock_gettime(CLOCK_MONOTONIC,&end);
      parameters->execution_times.evaluate += elapsed_time(&start,&end);      
    #endif      

    #ifdef VERBOSE
      cout << "t_int = " << global_internal_time << "; end of while; tp 0: r[0] = " << (*tp_data)[0].r[0] << endl;
      //for (int k=0; k<N_tp; k++) { cout << "next_internal_times[" << k << "] = " << next_internal_times[k] << endl; }   
    #endif

  } // end of while
  cout << " integrator_loop_index after while = " << integrator_loop_index << endl;
}


/*==================
  Time-step function
  ==================*/

double time_step_function(double u_d0[4], double u_d2[4], double u_d3[4], double u_d4[4], bool use_u_d4, double eta_factor, parameters_t *parameters)
{
  double u_d0_norm2 = norm4_squared(u_d0);
  double u_d2_norm = norm4(u_d2);
  double u_d3_norm = norm4(u_d3);
  double u_d4_norm,time_array[2],eta=parameters->eta*eta_factor;
    
  /* time_array uses normal, i.e. non-regularized, time */
  time_array[0] = u_d0_norm2*eta*u_d2_norm/u_d3_norm;
  int n=1;
  if (use_u_d4 == true) { n=2; u_d4_norm = norm4(u_d4); time_array[1] = u_d0_norm2*eta*sqrt(u_d2_norm/u_d4_norm); }

  double time_step = time_array[0];
  for (int i=0; i<n; i++) { if (time_array[i] < time_step) { time_step = time_array[i]; } } // find minimum of time_array

  /* transform the time-step into block form */
  int k=0;
  double block_time_step = 0.0;
  if (time_step < ABSOLUTE_MINIMUM_INTERNAL_TIME_STEP)
  {
    cout << "Warning, specified time-step precision of " << ABSOLUTE_MINIMUM_INTERNAL_TIME_STEP << " not sufficient" << endl;
    block_time_step = ABSOLUTE_MINIMUM_INTERNAL_TIME_STEP;
  }
  else
  {
    while (block_time_step < time_step)
    {
      block_time_step = ABSOLUTE_MINIMUM_INTERNAL_TIME_STEP*pow(2.0,k);
      k++;
    }
  }

  #ifdef VERBOSE
    cout << "time step function; time_array = {" << time_array[0] << ", " << time_array[1] << "}; time_step = " << time_step << "; block_time_step = " << block_time_step << "; k= " << k << endl;
  #endif

  return block_time_step;
}


/*============================
  Determine apocenter function
  ============================*/

void determine_apocenter_function(double global_internal_time, double r_begin[3], double r_end[3], double v_begin[3], double v_end[3], tp_data_t *tp_data, parameters_t *parameters)
{
  double r0 = tp_data->r_norm_hist;
  double r1 = norm3(r_begin);
  double r2 = norm3(r_end);
    
  double d1 = r1-r0;
  double d2 = r2-r1;
  tp_data->r_norm_hist = r1;

  if (d1 != 0 && d2 != 0)
  {
    d1 /= fabs(d1);
    d2 /= fabs(d2);
    if (d1 == 1.0 && d2 == -1.0) // apocenter
    {
      for (int c=0; c<3; c++)
      {
        tp_data->r_apo[c] = r_begin[c];
        tp_data->v_apo[c] = v_begin[c];
      }
      tp_data->t_apo = global_internal_time;
    }
  }
}


/*======================
  Capture check function
  ======================*/

void capture_check_function(double global_internal_time, double r_begin[3], double r_end[3], double v_begin[3], double v_end[3], tp_data_t *tp_data, parameters_t *parameters)
{
  double r_dif[3];
  for (int c=0; c<3; c++) { r_dif[c] = r_end[c] - r_begin[c]; }

  bool r_less_than_rcapt = false;
  double rb_dot_rdif = dot3(r_begin,r_dif);
  double rdif_dot_rdif = dot3(r_dif,r_dif);
  double discriminant2 = rb_dot_rdif*rb_dot_rdif - rdif_dot_rdif*(dot3(r_begin,r_begin) - parameters->tp_capture_radius*parameters->tp_capture_radius);
  double s1,s2;
  if (discriminant2 >= 0.0)
  {
    s1 = (-rb_dot_rdif + sqrt(discriminant2))/rdif_dot_rdif;
    s2 = (-rb_dot_rdif - sqrt(discriminant2))/rdif_dot_rdif;
    if ((s1 > 0.0 && s1 < 1.0) || (s2 > 0.0 && s2 < 1.0)) { r_less_than_rcapt = true; }
  }

  if (r_less_than_rcapt == true) // r < r_capt
  {
    tp_data->captured = true;
  }
}


/*=======================================
  Field particle shift & rotate functions
  =======================================*/

void shift_field_particles(double delta_time,std::vector<fp_data_t> *fp_data, parameters_t *parameters)
/* Convergence for the Kepler solver for delta_time does not always occur at one call of nkep_drift3. To obtain a solution in this case, decrease delta_time until the nkep_drift3 returns success and thus iterate untill delta_time is reached. */
{
  #ifdef DETERMINE_EXECUTION_TIMES
    timespec start,end;
    clock_gettime(CLOCK_MONOTONIC,&start);
  #endif

  double time,mu,time_step,end_time;
  int i_fp,result;

  #pragma omp parallel for if (parameters->enable_OpenMP == true && parameters->N_fp >= parameters->minimum_N_for_OpenMP) \
  shared(delta_time,fp_data,parameters,cout) \
  private(i_fp,result,mu,time,time_step,end_time) \
  default(none)
  for (i_fp=0; i_fp<parameters->N_fp; i_fp++)
  {
/*
    double r0[3],v0[3];
    for (int c=0; c<3; c++)
    {
      r0[c] = (*fp_data)[i_fp].r[c];
      v0[c] = (*fp_data)[i_fp].v[c];
    }
    double r_length = norm3(r0);
    double r_length2 = r_length*r_length;
    double r_length3 = r_length2*r_length;
    double v_length2 = norm3_squared(v0);
    double energy = (1.0/2.0)*v_length2 - gravitational_constant*parameters->SMBH_mass/r_length;
    double a = gravitational_constant*parameters->SMBH_mass/(-2.0*energy);
    double r_dot_v = dot3(r0,v0);
    double dt_aj = 0.01*parameters->eta*r_length2/sqrt(3.0*r_dot_v*r_dot_v + r_length2*v_length2);
    double dt_dyn = parameters->eta*sqrt(r_length3/(gravitational_constant*parameters->SMBH_mass));
    double dt_ex = 0.01*0.70710678118654746*sqrt(r_length3/(parameters->SMBH_mass*(1.0-r_length/(2.0*a))));
        cout << "pre " << dt_ex << endl;
    dt_ex = 0.01*2.0*PI*r_length/norm3(v0);
    cout << "post " << dt_ex << endl;
    //double dt = min(dt_aj,dt_dyn);
    cout << "shift delta time = " << delta_time << "; dt_aj = " << dt_aj << " dt_dyn = " << dt_dyn << " dt_ex " << dt_ex/10.0 << " a " << a << endl;
*/
    mu = gravitational_constant*((*fp_data)[i_fp].mass + parameters->SMBH_mass);
    time = 0.0; time_step = delta_time; end_time = delta_time;
    //end_time = dt_ex;
    while (time < end_time) 
    {
      result = nkep_drift3(mu, (*fp_data)[i_fp].r, (*fp_data)[i_fp].v, time_step);
      if (result == DRIFT_FAIL) { time_step /= 2.0; }
      else
      {
        time += time_step;
        time_step = end_time - time;
      }
    }
    if (result == DRIFT_FAIL) { cout << "Error shifting field particle " << i_fp << " with delta time " << delta_time << endl; }
/*
    cout.precision(20);
    double dr[3];
    for (int c=0; c<3; c++) { dr[c] = r0[c] - (*fp_data)[i_fp].r[c]; }
    double dr_length = norm3(dr);
    double change = dr_length/r_length;
    if (change > 0.1) { cout << "i_fp " << i_fp << "; change of r: " << change << endl; }
  }
  exit(EXIT_SUCCESS);*/
  }
  #ifdef DETERMINE_EXECUTION_TIMES
    clock_gettime(CLOCK_MONOTONIC,&end);
    parameters->execution_times.shift_fp += elapsed_time(&start,&end);
  #endif
}

void rotate_field_particles(double delta_time,std::vector<fp_data_t> *fp_data, parameters_t *parameters)
{
  #ifdef DETERMINE_EXECUTION_TIMES
    timespec start,end;
    clock_gettime(CLOCK_MONOTONIC,&start);
  #endif
    
	int i_fp;

  #pragma omp parallel for if (parameters->enable_OpenMP == true && parameters->N_fp >= parameters->minimum_N_for_OpenMP) \
	private(i_fp) shared(delta_time,fp_data,parameters) \
	default(none)
	for (i_fp=0; i_fp<parameters->N_fp; i_fp++)
  {
    double rotation_angle = delta_time*(*fp_data)[i_fp].precession_rate;
    double cos_rotation_angle = cos(rotation_angle),sin_rotation_angle = sin(rotation_angle);
    double *r = (*fp_data)[i_fp].r, *v = (*fp_data)[i_fp].v;
        
    double AM[3],AM_cross_r[3],AM_cross_v[3];
    cross3(r,v,AM);
    cross3(AM,r,AM_cross_r);
    cross3(AM,v,AM_cross_v);
    double AM_norm = norm3(AM);
        
    for (int c=0; c<3; c++)
    {
      r[c] = cos_rotation_angle*r[c] + sin_rotation_angle*AM_cross_r[c]/AM_norm;
      v[c] = cos_rotation_angle*v[c] + sin_rotation_angle*AM_cross_v[c]/AM_norm;
    }
  }
  
  #ifdef DETERMINE_EXECUTION_TIMES
    clock_gettime(CLOCK_MONOTONIC,&end);
    parameters->execution_times.rotate_fp += elapsed_time(&start,&end);
  #endif  
}


/*=====================
  Get gravity functions
  =====================*/
void get_gravity(int N_tp_calc, std::vector<tp_calc_t> *tp_calc, std::vector<fp_data_t> *fp_data, parameters_t *parameters)
/*  Input: r, v, u_d0, u_d1, h_d0
    Output: u_d2, u_d3, h_d1, h_d2  */
{
  double (*a_fp_on_tp)[3],(*j_fp_on_tp)[3];
  try { a_fp_on_tp = new double[N_tp_calc][3]; j_fp_on_tp = new double[N_tp_calc][3]; }
  catch (bad_alloc&) { cout << "Error allocating memory for a/j fp_on_tp in gg. Terminating program. " << endl; exit(EXIT_FAILURE); }

  #ifdef USE_GPU
    int *ids; // used by Sapporo
    double (*r)[3],(*v)[3],*h2,*pot; // used by Sapporo

    #ifdef DETERMINE_EXECUTION_TIMES
      timespec start_GPU,end_GPU;
      clock_gettime(CLOCK_MONOTONIC,&start_GPU);        
    #endif

    try { ids = new int[N_tp_calc]; r = new double[N_tp_calc][3]; v = new double[N_tp_calc][3]; h2 = new double[N_tp_calc]; pot = new double[N_tp_calc]; }
    catch (bad_alloc&) { cout << "Error allocating memory for Sapporo arrays. Terminating program. " << endl; exit(EXIT_FAILURE); }

    #ifdef DETERMINE_EXECUTION_TIMES
      clock_gettime(CLOCK_MONOTONIC,&end_GPU);
      parameters->execution_times.allocate_arrays_for_sapporo += elapsed_time(&start_GPU,&end_GPU);
      clock_gettime(CLOCK_MONOTONIC,&start_GPU);
    #endif

    get_fp_gravity_GPU_start(N_tp_calc,tp_calc,fp_data,parameters,ids,r,v,h2,pot,a_fp_on_tp,j_fp_on_tp);
      
#if 0
    double pos[3],vel[3],null3[3]={0.0,0.0,0.0},mass,eps2;
    double ppos[3],pvel[3];
    for (int i_fp=0; i_fp<5; i_fp++)
    {
      get_j_part_data
      (
        i_fp, //addr
        parameters->N_fp, //nj
        pos,
        vel,
        null3, //acc
        null3, //jrk
        ppos, //ppos
        pvel, //pvel
        mass,
        eps2,
        i_fp
      );
      cout << "i_fp="<<i_fp<< " Sapporo2 pos " << pos[0] << " " << pos[1] << " " << pos[2] << " " << " Sapporo2 vel " << vel[0] << " " << vel[1] << " " << vel[2] << endl;
      //cout << "i_fp="<<i_fp<< " t = " << global_internal_time << " Sapporo2 ppos " << ppos[0] << " " << ppos[1] << " " << ppos[2] << " " << " Sapporo2 pvel " << pvel[0] << " " << pvel[1] << " " << pvel[2] << endl;
      cout << "i_fp="<<i_fp<< " CPU pos " << (*fp_data)[i_fp].r[0] << " " << (*fp_data)[i_fp].r[1] << " " << (*fp_data)[i_fp].r[2] << " " << " CPU vel " << (*fp_data)[i_fp].v[0] << " " << (*fp_data)[i_fp].v[1] << " " << (*fp_data)[i_fp].v[2] << endl;        
     }
#endif
   
  #else
    get_fp_gravity_CPU(N_tp_calc,tp_calc,fp_data,parameters,a_fp_on_tp,j_fp_on_tp);
  #endif

  get_PN_acc(N_tp_calc,tp_calc,parameters);

  #ifdef USE_GPU
    get_fp_gravity_GPU_retrieve(N_tp_calc,tp_calc,fp_data,parameters,ids,r,v,h2,pot,a_fp_on_tp,j_fp_on_tp);
    //cout << "test " << a_fp_on_tp[0][0] << endl;
    
    #ifdef DETERMINE_EXECUTION_TIMES
      clock_gettime(CLOCK_MONOTONIC,&end_GPU);
      parameters->execution_times.gg_fp_GPU += elapsed_time(&start_GPU,&end_GPU);
    #endif  

    delete[] ids; delete[] r; delete[] v; delete[] h2; delete[] pot;    
  #endif

  for (int i=0; i<N_tp_calc; i++)
  {
    if (a_fp_on_tp[i][0] != a_fp_on_tp[i][0]) { cout << "warning a_fp_on_tp[i][0] is nan; i = " << i << endl; }
    if (j_fp_on_tp[i][0] != j_fp_on_tp[i][0]) { cout << "warning j_fp_on_tp[i][0] is nan; i = " << i << endl; }
  }

  get_PN_jerk(N_tp_calc,tp_calc,parameters,a_fp_on_tp,j_fp_on_tp);

  delete[] a_fp_on_tp; delete[] j_fp_on_tp;
}


#ifdef USE_GPU
void GPU_set_j_particles(std::vector<fp_data_t> *fp_data, parameters_t *parameters)
{
  #ifdef VERBOSE
    cout << "GPU_set_j_particles" << endl;
  #endif

  #ifdef DETERMINE_EXECUTION_TIMES
    timespec start,end;
    clock_gettime(CLOCK_MONOTONIC,&start);
  #endif

  int cluster_id=0,i_fp;
  double null0=0,null3[3] = {0.0,0.0,0.0},tj=0.0;
  double acc[3];
  acc[0] = parameters->SMBH_mass;

  for (int i_fp=0; i_fp<parameters->N_fp; i_fp++)
  {
	  acc[1] = (*fp_data)[i_fp].precession_rate;
    g6_set_j_particle_
    (
      &cluster_id,
      &i_fp,
      &i_fp,
      &tj, // tj
      &null0, // dtj
      &((*fp_data)[i_fp].mass),
      null3, // k18
      null3, // j6
      acc, // a2
      (*fp_data)[i_fp].v,
      (*fp_data)[i_fp].r
    );
  }

  #ifdef DETERMINE_EXECUTION_TIMES
    clock_gettime(CLOCK_MONOTONIC,&end);
    parameters->execution_times.GPU_set_j_particles += elapsed_time(&start,&end);
  #endif

  #ifdef VERBOSE
    cout << "GPU_set_j_particles - done" << endl;
  #endif

}

void get_fp_gravity_GPU_start(int N_tp_calc,std::vector<tp_calc_t> *tp_calc,std::vector<fp_data_t> *fp_data,parameters_t *parameters, int *ids, double (*r)[3], double (*v)[3], double *h2, double *pot, double (*a_fp_on_tp)[3], double (*j_fp_on_tp)[3])
{
  #ifdef VERBOSE
    cout << "fp_gravity_GPU_start" << endl;
  #endif
	
  int cluster_id=0,npipes = g6_npipes_(),npart;
  double eps2=0.0;
  
  for (int i_tp_calc=0; i_tp_calc<N_tp_calc; i_tp_calc++)
  {
	  ids[i_tp_calc] = i_tp_calc;
	  for (int c=0; c<3; c++)
	  {
      r[i_tp_calc][c] = (*tp_calc)[i_tp_calc].r[c];  
      v[i_tp_calc][c] = (*tp_calc)[i_tp_calc].v[c];
    }
    h2[i_tp_calc] = 0.0;
    pot[i_tp_calc] = 0.0;
    //cout << "fp_gravity_GPU_start r " << r[i_tp_calc][0] << " " << r[i_tp_calc][1] << " " << r[i_tp_calc][2] << endl;
    //cout << "fp_gravity_GPU_start ids " << ids[i_tp_calc] << endl;
    //cout << "fp_gravity_GPU_start h2 " << h2[i_tp_calc] << endl;
    //cout << "fp_gravity_GPU_start pot " << pot[i_tp_calc] << endl;    
  }
  
  for (int i=0; i<N_tp_calc; i+=npipes)
  {
	//cout << "i " << i << " pr " << r+i << endl;
	//cout << "temp " << *(*(r+i)+1) << endl;
    npart = min(N_tp_calc-i,npipes);
    //cout << "npart = " << npart << endl;
    g6calc_firsthalf_
    (
      &cluster_id,
      &(parameters->N_fp),
      &npart,
      ids+i,
      r+i,
      v+i,
      a_fp_on_tp+i,
      j_fp_on_tp+i,
      pot+i, // potential; not used here
      &eps2, // softening; set to 0
      h2+i // h2 (GRAPE legacy)
    );
  }

  #ifdef VERBOSE
    cout << "fp_gravity_GPU_start - done" << endl;
  #endif  
}

void get_fp_gravity_GPU_retrieve(int N_tp_calc,std::vector<tp_calc_t> *tp_calc,std::vector<fp_data_t> *fp_data,parameters_t *parameters, int *ids, double (*r)[3], double (*v)[3], double *h2, double *pot, double (*a_fp_on_tp)[3], double (*j_fp_on_tp)[3])
{
  #ifdef VERBOSE
    cout << "fp_gravity_GPU_retrieve" << endl;
  #endif
	
  int cluster_id=0,npipes = g6_npipes_(),npart;
  double eps2=0.0;
  
  for (int i=0; i<N_tp_calc; i+=npipes)
  {
    npart = min(N_tp_calc-i,npipes);
    g6calc_lasthalf_
    (
      &cluster_id,
      &(parameters->N_fp),
      &npart,
      ids+i,
      r+i,
      v+i,
      &eps2, // softening; set to 0
      h2+i, // h2 (GRAPE legacy)
      a_fp_on_tp+i,
      j_fp_on_tp+i,
      pot+i // potential; not used here
    );
  }

  #ifdef VERBOSE
    cout << "fp_gravity_GPU_retrieve - done" << endl;
  #endif  
}
#endif

void get_fp_gravity_CPU(int N_tp_calc, std::vector<tp_calc_t> *tp_calc, std::vector<fp_data_t> *fp_data, parameters_t *parameters, double (*a_fp_on_tp)[3], double (*j_fp_on_tp)[3])
{
  #ifdef DETERMINE_EXECUTION_TIMES
    timespec start,end;
    clock_gettime(CLOCK_MONOTONIC,&start);
  #endif

	int i_tp_calc,c,i_fp;
	double fp_mass,g[3],gd[3],g_l,g_l2,g_l3,g_dot_gd,a_temp,a_fp_on_tp_x,a_fp_on_tp_y,a_fp_on_tp_z,j_fp_on_tp_x,j_fp_on_tp_y,j_fp_on_tp_z;

  for (i_tp_calc=0; i_tp_calc<N_tp_calc; i_tp_calc++)
  {
    a_fp_on_tp_x = 0.0; a_fp_on_tp_y = 0.0; a_fp_on_tp_z = 0.0;
    j_fp_on_tp_x = 0.0; j_fp_on_tp_y = 0.0; j_fp_on_tp_z = 0.0; 
         
    #pragma omp parallel for if (parameters->enable_OpenMP == true && parameters->N_fp >= parameters->minimum_N_for_OpenMP) \
    shared(i_tp_calc,fp_data,tp_calc,parameters,cout) \
    private(i_fp,c,fp_mass,g,gd,g_l,g_l2,g_l3,g_dot_gd,a_temp) \
    reduction(+:a_fp_on_tp_x,a_fp_on_tp_y,a_fp_on_tp_z,j_fp_on_tp_x,j_fp_on_tp_y,j_fp_on_tp_z) \
    default(none)
		for (i_fp=0; i_fp<parameters->N_fp; i_fp++)
		{
		  fp_mass = (*fp_data)[i_fp].mass;
      for (c=0; c<3; c++)
			{
			  g[c] = (*fp_data)[i_fp].r[c] - (*tp_calc)[i_tp_calc].r[c];
				gd[c] = (*fp_data)[i_fp].v[c] - (*tp_calc)[i_tp_calc].v[c];
			}
			g_l = norm3(g);
			g_l2 = g_l*g_l;
			g_l3 = g_l*g_l2;
			g_dot_gd = dot3(g,gd);
			for (c=0; c<3; c++) // inelegant method because OpenMP reduction does not support arrays
			{
				a_temp = fp_mass*g[c]/g_l3;
        if (c==0) { a_fp_on_tp_x += a_temp; j_fp_on_tp_x += fp_mass*gd[c]/g_l3 - 3.0*a_temp*g_dot_gd/g_l2; }
        if (c==1) { a_fp_on_tp_y += a_temp; j_fp_on_tp_y += fp_mass*gd[c]/g_l3 - 3.0*a_temp*g_dot_gd/g_l2; }
        if (c==2) { a_fp_on_tp_z += a_temp; j_fp_on_tp_z += fp_mass*gd[c]/g_l3 - 3.0*a_temp*g_dot_gd/g_l2; }
			}
    }
    for (c=0; c<3; c++)
		{ 
      if (c==0) { a_fp_on_tp[i_tp_calc][c] = a_fp_on_tp_x; j_fp_on_tp[i_tp_calc][c] = j_fp_on_tp_x; }
      if (c==1) { a_fp_on_tp[i_tp_calc][c] = a_fp_on_tp_y; j_fp_on_tp[i_tp_calc][c] = j_fp_on_tp_y; }
      if (c==2) { a_fp_on_tp[i_tp_calc][c] = a_fp_on_tp_z; j_fp_on_tp[i_tp_calc][c] = j_fp_on_tp_z; }
    }

/*
      double a_fp_on_tp_[3],j_fp_on_tp_[3];
      for (c=0; c<3; c++) { a_fp_on_tp_[c] = 0.0; j_fp_on_tp_[c] = 0.0; }

			for (i_fp=0; i_fp<parameters->N_fp; i_fp++)
			{
				fp_mass = (*fp_data)[i_fp].mass;
				for (c=0; c<3; c++)
				{
					g[c] = (*fp_data)[i_fp].r[c] - (*tp_calc)[i_tp_calc].r[c];
					gd[c] = (*fp_data)[i_fp].v[c] - (*tp_calc)[i_tp_calc].v[c];
//					cout << "diag 1 " << (*tp_calc)[i_tp_calc].r[c] << endl;
				}
				g_l = norm3(g);
				g_l2 = g_l*g_l;
				g_l3 = g_l*g_l2;
				g_dot_gd = dot3(g,gd);
				for (c=0; c<3; c++)
				{
					a_temp = fp_mass*g[c]/g_l3;
					//cout.precision(15);
//					cout << "diag 2 " << a_temp << endl;
					a_fp_on_tp_[c] += a_temp;
					j_fp_on_tp_[c] += fp_mass*gd[c]/g_l3 - 3.0*a_temp*g_dot_gd/g_l2;
				}
			}
      cout.precision(25);
      double temp;
      for (c=0; c<3; c++) { temp = a_fp_on_tp[i_tp_calc][c]/a_fp_on_tp_[c]; if (temp != 1) { cout << "c = " << c << "; OMP/Single " << a_fp_on_tp[i_tp_calc][c]/a_fp_on_tp_[c] << endl; } }
*/
	}

  #ifdef DETERMINE_EXECUTION_TIMES
    clock_gettime(CLOCK_MONOTONIC,&end); 
    parameters->execution_times.gg_fp_CPU += elapsed_time(&start,&end);
  #endif    
}

void get_fp_gravity_CPU_serial(int N_tp_calc, std::vector<tp_calc_t> *tp_calc, std::vector<fp_data_t> *fp_data, parameters_t *parameters, double (*a_fp_on_tp)[3], double (*j_fp_on_tp)[3])
{
  #ifdef DETERMINE_EXECUTION_TIMES
    timespec start,end;
    clock_gettime(CLOCK_MONOTONIC,&start);
  #endif
    
  int i_tp_calc,c,i_fp;
  double fp_mass,g[3],gd[3],g_l,g_l2,g_l3,g_dot_gd,a_temp;

  for (i_tp_calc=0; i_tp_calc<N_tp_calc; i_tp_calc++)
  {
		for (c=0; c<3; c++) { a_fp_on_tp[i_tp_calc][c] = 0.0; j_fp_on_tp[i_tp_calc][c] = 0.0; }

		for (i_fp=0; i_fp<parameters->N_fp; i_fp++)
		{
			fp_mass = (*fp_data)[i_fp].mass;
			for (c=0; c<3; c++)
			{
				g[c] = (*fp_data)[i_fp].r[c] - (*tp_calc)[i_tp_calc].r[c];
				gd[c] = (*fp_data)[i_fp].v[c] - (*tp_calc)[i_tp_calc].v[c];
			}
			g_l = norm3(g);
			g_l2 = g_l*g_l;
			g_l3 = g_l*g_l2;
			g_dot_gd = dot3(g,gd);
			for (c=0; c<3; c++)
			{
				a_temp = fp_mass*g[c]/g_l3;
				cout.precision(15);
				a_fp_on_tp[i_tp_calc][c] += a_temp;
				j_fp_on_tp[i_tp_calc][c] += fp_mass*gd[c]/g_l3 - 3.0*a_temp*g_dot_gd/g_l2;
			}
		}
	}
  
  #ifdef DETERMINE_EXECUTION_TIMES
    clock_gettime(CLOCK_MONOTONIC,&end); 
    parameters->execution_times.gg_fp_CPU += elapsed_time(&start,&end);
  #endif      
}

void get_PN_acc(int N_tp_calc, std::vector<tp_calc_t> *tp_calc, parameters_t *parameters)
{
  #ifdef DETERMINE_EXECUTION_TIMES
    timespec start,end;
    clock_gettime(CLOCK_MONOTONIC,&start);
  #endif

  double SMBH_mass=parameters->SMBH_mass,tp_25PN_mass=parameters->tp_25PN_mass,GMbh = gravitational_constant*SMBH_mass,GMtot,total_mass;
  double c2 = speed_of_light*speed_of_light,GMbhdc2 = GMbh/c2,GMbh2dc2 = GMbh*GMbh/c2;
  double r[3],v[3],r_l,r_l2,r_l3,r_l4,r_l5,r_dot_v,v_dot_v,r_dot_chi,v_cross_r[3],v_cross_chi[3],*chi=parameters->chi;
  double GMbh2drc4,GMbhdr,nu,GM2dc5nu,G2Mbh2dc3drl3,G3Mbh3dc4chi2drl5,chi_l2;
  int i_tp_calc,c;

  #pragma omp parallel for if (parameters->enable_OpenMP == true && N_tp_calc >= parameters->minimum_N_for_OpenMP) \
  shared(N_tp_calc,tp_calc,parameters,SMBH_mass,tp_25PN_mass,GMbh,GMtot,total_mass,c2,GMbhdc2,GMbh2dc2,chi,nu,GM2dc5nu,chi_l2) \
  private(i_tp_calc,c,r,v,r_l,r_l2,r_l3,r_l4,r_l5,r_dot_v,v_dot_v,r_dot_chi,v_cross_r,v_cross_chi,GMbh2drc4,GMbhdr,G2Mbh2dc3drl3,G3Mbh3dc4chi2drl5) \
  default(none)
  for (i_tp_calc=0; i_tp_calc<N_tp_calc; i_tp_calc++)
  {
		for (c=0; c<3; c++) // to avoid accessing tp_calc in the calculations below
		{
			r[c] = (*tp_calc)[i_tp_calc].r[c];
			v[c] = (*tp_calc)[i_tp_calc].v[c];
	  }
    r_l = norm3(r);
    r_l2 = r_l*r_l;
    r_l3 = r_l*r_l2;
    r_dot_v = dot3(r,v);
    
    if (parameters->include_1PN_terms == true) { r_l4 = r_l*r_l3; v_dot_v = dot3(v,v); }
    if (parameters->include_2PN_terms == true)
    {
		  if (parameters->include_1PN_terms == false) { r_l4 = r_l*r_l3; }
		  GMbh2drc4 = GMbh2dc2/(c2*r_l4);
		  GMbhdr = GMbh/r_l;
		}
    if (parameters->include_25PN_terms == true)
    {
      if (parameters->include_1PN_terms == false) { r_l4 = r_l*r_l3; v_dot_v = dot3(v,v); }
      total_mass = SMBH_mass + tp_25PN_mass;
      nu = SMBH_mass*tp_25PN_mass/(total_mass*total_mass);
      GMtot = gravitational_constant*total_mass;
      GM2dc5nu = c8div5*GMtot*GMtot*nu/(speed_of_light*c2*c2);
    }
    if (parameters->include_15PN_spin_terms == true) 
    {
      r_dot_chi = dot3(r,chi);
      cross3(v,r,v_cross_r);
      cross3(v,chi,v_cross_chi);
      G2Mbh2dc3drl3 = 2.0*GMbh2dc2/(speed_of_light*r_l3);
    }
    if (parameters->include_2PN_spin_terms == true) 
    {
      if (parameters->include_1PN_terms == false) { r_l4 = r_l*r_l3; }
      if (parameters->include_15PN_spin_terms == false) { r_dot_chi = dot3(r,chi); }
      chi_l2 = norm3_squared(chi);
      r_l5 = r_l*r_l4;
      G3Mbh3dc4chi2drl5 = c3div2*GMbh2dc2*GMbh*chi_l2/(c2*r_l5);
    }

    for (c=0; c<3; c++)
    {
      (*tp_calc)[i_tp_calc].a_PN_tp[c] = 0.0;
      if (parameters->include_1PN_terms == true ) { (*tp_calc)[i_tp_calc].a_PN_tp[c] += 4.0*GMbhdc2*r_dot_v*v[c]/r_l3 + 4.0*GMbh2dc2*r[c]/r_l4 - GMbhdc2*v_dot_v*r[c]/r_l3; }
      if (parameters->include_2PN_terms == true ) { (*tp_calc)[i_tp_calc].a_PN_tp[c] += GMbh2drc4*(2.0*r_dot_v*r_dot_v*r[c]/r_l2 - 9.0*GMbhdr*r[c] - 2.0*r_dot_v*v[c]); }
      if (parameters->include_25PN_terms == true) { (*tp_calc)[i_tp_calc].a_PN_tp[c] += -GM2dc5nu/r_l3*( (v_dot_v + 3.0*GMtot/r_l)*v[c] - (3.0*v_dot_v + c17div3*GMtot/r_l)*r_dot_v*r[c]/r_l2 ); }
      if (parameters->include_15PN_spin_terms == true) { (*tp_calc)[i_tp_calc].a_PN_tp[c] += -G2Mbh2dc3drl3*( -v_cross_chi[c] + 3.0*r_dot_chi*v_cross_r[c]/r_l2); }
      if (parameters->include_2PN_spin_terms == true) { (*tp_calc)[i_tp_calc].a_PN_tp[c] += -G3Mbh3dc4chi2drl5*( r_dot_chi*( 5.0*r_dot_chi*r[c]/r_l2 - 2.0*chi[c]) - r[c]); }
    }    
  }

  #ifdef DETERMINE_EXECUTION_TIMES
    clock_gettime(CLOCK_MONOTONIC,&end);
    parameters->execution_times.gg_PN_acc += elapsed_time(&start,&end);
  #endif  
}

void get_PN_jerk(int N_tp_calc, std::vector<tp_calc_t> *tp_calc, parameters_t *parameters, double (*a_fp_on_tp_)[3], double (*j_fp_on_tp_)[3])
/* Notes: the total acceleration a_tot_tp (Newton SMBH + PN SMBH + Newton fp) is used to calculate PN jerks.
   The external acceleration/jerk a_ext_tp/j_ext_tp (PN SMBH + Newton fp) are used in the regularized equations of motion. */
{
  #ifdef DETERMINE_EXECUTION_TIMES
    timespec start,end;
    clock_gettime(CLOCK_MONOTONIC,&start);
  #endif
    
  double SMBH_mass=parameters->SMBH_mass,tp_25PN_mass=parameters->tp_25PN_mass,GMbh = gravitational_constant*SMBH_mass,GMtot,total_mass;
  double c2 = speed_of_light*speed_of_light,GMbhdc2 = GMbh/c2,GMbh2dc2 = GMbh*GMbh/c2;
  double r[3],v[3],r_l,r_l2,r_l3,r_l4,r_l5,r_l6,r_l7,r_l8,r_dot_v,v_dot_v,r_dot_chi,v_cross_r[3],a_cross_r[3],v_cross_chi[3],a_cross_chi[3],*chi=parameters->chi;
  double r_dot_a,v_dot_a,v_dot_chi;
  double GMbh2drc4,GMbhdr,nu,GM2dc5nu,G2Mbh2dc3drl3,G3Mbh3dc4chi2drl5,chi_l2;
  
  double a_fp_on_tp[3],j_fp_on_tp[3],a_PN_tp[3],j_PN_tp[3],a_tot_tp[3],a_ext_tp[3],j_ext_tp[3];
  double u_d0[4],u_d1[4],u_d2[4],h_d0,h_d1,LT_u_d0_a[4],LT_u_d1_a[4],LT_u_d0_j[4];
  int i_tp_calc,c;
  
  #pragma omp parallel for if (parameters->enable_OpenMP == true && N_tp_calc >= parameters->minimum_N_for_OpenMP) \
  shared(N_tp_calc,parameters,tp_calc,a_fp_on_tp_,j_fp_on_tp_,SMBH_mass,tp_25PN_mass,GMbh,GMtot,total_mass,c2,GMbhdc2,GMbh2dc2,chi,nu,GM2dc5nu,chi_l2) \
  private(i_tp_calc,c,r,v,r_l,r_l2,r_l3,r_l4,r_l5,r_l6,r_l7,r_l8,r_dot_v,v_dot_v,r_dot_chi,v_cross_r,a_cross_r,v_cross_chi,a_cross_chi,r_dot_a,v_dot_a,v_dot_chi, \
    GMbh2drc4,GMbhdr,G2Mbh2dc3drl3,G3Mbh3dc4chi2drl5,a_fp_on_tp,j_fp_on_tp,a_PN_tp,j_PN_tp,a_tot_tp,a_ext_tp,j_ext_tp,u_d0,u_d1,u_d2,h_d0,h_d1,LT_u_d0_a,LT_u_d1_a,LT_u_d0_j) \
  default(none)
  for (i_tp_calc=0; i_tp_calc<N_tp_calc; i_tp_calc++)
  {
    for (c=0; c<3; c++) // to avoid accessing tp_calc in the calculations below
    {
      r[c] = (*tp_calc)[i_tp_calc].r[c];
      v[c] = (*tp_calc)[i_tp_calc].v[c];
      a_PN_tp[c] = (*tp_calc)[i_tp_calc].a_PN_tp[c];
      a_fp_on_tp[c] = a_fp_on_tp_[i_tp_calc][c];
      j_fp_on_tp[c] = j_fp_on_tp_[i_tp_calc][c];
    }
    r_l = norm3(r);
    r_l2 = r_l*r_l;
    r_l3 = r_l*r_l2;
    r_l4 = r_l*r_l3;
    r_dot_v = dot3(r,v);

    for (c=0; c<3; c++) // total acceleration
    {
      a_fp_on_tp[c] *= gravitational_constant; j_fp_on_tp[c] *= gravitational_constant;
      a_tot_tp[c] = -GMbh*r[c]/r_l3 + a_PN_tp[c] + a_fp_on_tp[c];
    }
            
    if (parameters->include_1PN_terms == true) { r_l5=r_l*r_l4; r_l6=r_l*r_l5; v_dot_v=dot3(v,v); r_dot_a=dot3(r,a_tot_tp); v_dot_a=dot3(v,a_tot_tp); }
    if (parameters->include_2PN_terms == true)
    {
		  if (parameters->include_1PN_terms == false) { r_l4 = r_l*r_l3; v_dot_v=dot3(v,v); r_dot_a=dot3(r,a_tot_tp); }
		  GMbh2drc4 = GMbh2dc2/(c2*r_l4);
		  GMbhdr = GMbh/r_l;
		}    
    if (parameters->include_25PN_terms == true)
    {
      if (parameters->include_1PN_terms == false) { r_l5=r_l*r_l4; r_l6=r_l*r_l5; v_dot_v=dot3(v,v); r_dot_a=dot3(r,a_tot_tp); v_dot_a=dot3(v,a_tot_tp); }
      total_mass = SMBH_mass + tp_25PN_mass;
      nu = SMBH_mass*tp_25PN_mass/(total_mass*total_mass);
      GMtot = gravitational_constant*total_mass;
      GM2dc5nu = c8div5*GMtot*GMtot*nu/(speed_of_light*c2*c2);
      r_l7 = r_l*r_l6;
      r_l8 = r_l*r_l7;
    }
    if (parameters->include_15PN_spin_terms == true) 
    {
      r_dot_chi = dot3(r,chi);
      v_dot_chi = dot3(v,chi);
      cross3(v,r,v_cross_r);
      cross3(a_tot_tp,r,a_cross_r);
      cross3(v,chi,v_cross_chi);
      cross3(a_tot_tp,chi,a_cross_chi);      
      G2Mbh2dc3drl3 = 2.0*GMbh2dc2/(speed_of_light*r_l3);
    }
    if (parameters->include_2PN_spin_terms == true) 
    {
      if (parameters->include_15PN_spin_terms == false) { r_dot_chi = dot3(r,chi); v_dot_chi = dot3(v,chi); }
      chi_l2 = norm3_squared(chi);
      r_l5 = r_l*r_l4;
      G3Mbh3dc4chi2drl5 = c3div2*GMbh2dc2*GMbh*chi_l2/(c2*r_l5);
    }

    for (c=0; c<3; c++) // PN jerks
    {
      j_PN_tp[c] = 0.0;
      if (parameters->include_1PN_terms == true)
      {
        j_PN_tp[c] += 4.0*GMbhdc2*r_dot_v*a_tot_tp[c]/r_l3 - 12.0*GMbhdc2*r_dot_v*r_dot_v*v[c]/r_l5 + 4.0*GMbhdc2*(v_dot_v \
          + r_dot_a)*v[c]/r_l3 + 4.0*GMbh2dc2*v[c]/r_l4 - 16.0*GMbh2dc2*r_dot_v*r[c]/r_l6 - GMbhdc2*v_dot_v*v[c]/r_l3 \
          + 3.0*GMbhdc2*r_dot_v*v_dot_v*r[c]/r_l5 - 2.0*GMbhdc2*v_dot_a*r[c]/r_l3;
      }
      if (parameters->include_2PN_terms == true)
      {
        j_PN_tp[c] += GMbh2drc4*(-2.0*(v_dot_v + r_dot_a)*v[c] - 2.0*r_dot_v*a_tot_tp[c] - 9.0*GMbhdr*v[c] \
          + (2.0*r_dot_v/r_l2)*(5.0*r_dot_v*v[c] + 2.0*(v_dot_v + r_dot_a)*r[c]) + 45.0*GMbhdr*r_dot_v*r[c]/r_l2 \
          - 12.0*r_dot_v*r_dot_v*r_dot_v*r[c]/r_l4);
      }      
      if (parameters->include_25PN_terms == true)
      {
        j_PN_tp[c] += -GM2dc5nu*( (2.0*v_dot_a*v[c] + v_dot_v*a_tot_tp[c])/r_l3 + 3.0*GMtot*a_tot_tp[c]/r_l4 \
          - 3.0*(2.0*v_dot_v*r_dot_v*v[c] + v_dot_v*v_dot_v*r[c] + v_dot_v*r_dot_a*r[c] + 2.0*r_dot_v*v_dot_a*r[c])/r_l5 \
          - GMtot*(c53div3*r_dot_v*v[c] + c17div3*v_dot_v*r[c] + c17div3*r_dot_a*r[c])/r_l6 \
          + 15.0*v_dot_v*r_dot_v*r_dot_v*r[c]/r_l7 + 34.0*GMtot*r_dot_v*r_dot_v*r[c]/r_l8 );
      }
      if (parameters->include_15PN_spin_terms == true)
      {
        j_PN_tp[c] += -G2Mbh2dc3drl3*( -a_cross_chi[c] + (3.0/r_l2)*( r_dot_v*v_cross_chi[c] + v_dot_chi*v_cross_r[c] \
          + r_dot_chi*a_cross_r[c]) - (15.0/r_l4)*r_dot_chi*r_dot_v*v_cross_r[c] );
      }
      if (parameters->include_2PN_spin_terms == true)
      {
        j_PN_tp[c] += -G3Mbh3dc4chi2drl5*( (5.0*r_dot_chi/r_l2)*(r_dot_chi*v[c] + 2.0*v_dot_chi*r[c] \
          - 7.0*r_dot_chi*r_dot_v*r[c]/r_l2 + 2.0*r_dot_v*chi[c]) - 2.0*v_dot_chi*chi[c] - v[c] + 5.0*r_dot_v*r[c]/r_l2);
      }

      /* external acceleration/jerk */
      a_ext_tp[c] = a_PN_tp[c] + a_fp_on_tp[c];
      j_ext_tp[c] = j_PN_tp[c] + j_fp_on_tp[c];
    }

    /* Regularized equations of motion */
    for (c=0; c<4; c++) // to avoid accessing tp_calc in the calculations below
    {
			u_d0[c] = (*tp_calc)[i_tp_calc].u_d0[c];
			u_d1[c] = (*tp_calc)[i_tp_calc].u_d1[c];
	  }
    h_d0 = (*tp_calc)[i_tp_calc].h_d0;

    LT_u_on_vec3(u_d0,a_ext_tp,LT_u_d0_a);
    LT_u_on_vec3(u_d1,a_ext_tp,LT_u_d1_a);
    LT_u_on_vec3(u_d0,j_ext_tp,LT_u_d0_j);

    h_d1 = 2.0*dot4(u_d1,LT_u_d0_a);
    for (c=0; c<4; c++)
    {
      u_d2[c] = c1div2*h_d0*u_d0[c] + c1div2*r_l*LT_u_d0_a[c];
      (*tp_calc)[i_tp_calc].u_d3[c] = c1div2*h_d1*u_d0[c] + c1div2*h_d0*u_d1[c] + dot4(u_d0,u_d1)*LT_u_d0_a[c] \
        + c1div2*r_l*LT_u_d1_a[c] + c1div2*r_l2*LT_u_d0_j[c];
        
      (*tp_calc)[i_tp_calc].u_d2[c] = u_d2[c];
    }
    (*tp_calc)[i_tp_calc].h_d1 = h_d1;
    (*tp_calc)[i_tp_calc].h_d2 = 2.0*dot4(u_d2,LT_u_d0_a) + 2.0*dot4(u_d1,LT_u_d1_a) + 2.0*r_l*dot4(u_d1,LT_u_d0_j);
  }
  
  #ifdef DETERMINE_EXECUTION_TIMES
    clock_gettime(CLOCK_MONOTONIC,&end);
    parameters->execution_times.gg_PN_jerk += elapsed_time(&start,&end);
  #endif    
}
