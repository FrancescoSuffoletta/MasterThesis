"""
Author : Francesco Benigno Suffoletta

The script is used to set up the parallelization
of simulations and the simulations themselves to run
various simulations of the HORUS vehicle. 

"""
# Import packages for parallelization 
import multiprocessing 
from functools import partial 

# Import needed basic packages
import time
import os
import numpy as np
import math 
import shutil
import logging

# Import packages for the simulations from TUDAT
from tudatpy.kernel.interface import spice_interface 
from tudatpy.data import save2txt
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, environment, propagation_setup
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime
from tudatpy.kernel.math import interpolators

# Import the other required for the simulations
import Utils

run_simulations = True

spice_interface.load_standard_kernels()

# Here we start defining the functions for the simulations and parallelization 

def move_json_file( source_path, destination_dir ) :

    """
    The function moves the .json file produced by each simulation containing the
    parameters for the PINN training from the Main Guidance script from its original 
    directory to the desired final one.

    Parameters
    ----------
    source path : str
        The path containing the .json file with the training parameters
    destination_dir : str 
        The final path to contain the .json file with the training parameters

    Returns
    -------
    None

    """

    # Ensure the destination directory exists; if not, create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Define the actual destination path for the .json file with training parameters
    destination_path = os.path.join( destination_dir, "Train.json" )

    # Move the file
    shutil.move(source_path, destination_path) 

def perturb_initial_conditions( nominal_initial_state ) : 

    """
    The function provides perturbation for the nominal initial state to create many 
    different simulations that will be needed for the training of the PINN framework

    Parameters
    ----------
    nominal_initial_state : np.ndarray 
        The 6x1 array in spherical coordinates to be perturbed
    
    Returns
    -------
    perturbed_initial_state : np.ndarray
        The 6x1 perturbed initial state in spherical coordinates

    """

    # The initial state is provided in spherical coordinates and in order it contains :
    # Distance from Earth c.o.m. , latitude, longitude, velocity, flight path angle, heading angle
    perturbed_initial_state = np.zeros((6, 1))
    perturbed_initial_state[0] = np.random.normal( nominal_initial_state[0] , 150/3 , 1) # [m]
    perturbed_initial_state[1] = np.random.normal( nominal_initial_state[1] , np.deg2rad(0.1)/3 , 1) # [rad]
    perturbed_initial_state[2] = np.random.normal( nominal_initial_state[2] , np.deg2rad(0.1)/3 , 1) # [rad]
    perturbed_initial_state[3] = np.random.normal( nominal_initial_state[3] , 15/3 , 1) # [m/s]
    perturbed_initial_state[4] = np.random.normal( nominal_initial_state[4] , np.deg2rad(0.1)/3 , 1) # [rad]
    perturbed_initial_state[5] = np.random.normal( nominal_initial_state[5] , np.deg2rad(0.1)/3 , 1) # [rad]

    return perturbed_initial_state


def delete_half_compiled_files(directory, keyword):
    """
    Delete files in the specified directory that contain the given keyword.
    
    Parameters
    ----------
    directory : str
        The path to the directory where the files are located.
    keyword : str
        The keyword to match files for deletion.
    
    Returns
    -------
    None
    """
    try:
        # List all files in the directory
        for filename in os.listdir(directory):
            # Check if the file name contains the keyword
            if keyword in filename:
                # Construct the full file path
                file_path = os.path.join(directory, filename)
                # Check if it's a file and remove it
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                # If it's a directory and you want to remove it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Deleted directory: {file_path}")
    except Exception as e:
        print(f"Error while deleting files: {str(e)}")

def produce_simulation( path_definition, guidance_type, wind, summary_csv=None ) :

    """
    The function produces re-entry simulations given different 
    initial conditions that are perturbed with respect to the
    nominal ones.

    Parameters
    ----------
    path_definition : str
        The final part of the path to distinguish different 
        outcomes of the simulations
    guidance_type : str 
        Type of algorithm to be used
    wind : bool
        Whether wind is included in simulations or not

    Returns
    -------
    Success : bool
        Flag indicating whether the simulation reached TAEM conditions without violating the constraints
    
    """
    # Choose whether output of the propagation is written to files
    write_results_to_file = True
    # Get path of current directory
    current_dir = os.path.dirname(__file__)
    # Set simulation start epoch
    simulation_start_epoch = 0.0  # s
    # Set termination conditions (Maximum duration of the simulation equals one Julian Day)
    maximum_duration = constants.JULIAN_DAY  # s
    termination_altitude = 25.0E3  # m
    termination_angle = 0.75 # deg
    # Define the boolean statements for the current propagation
    verification = False
    main_sim = True  
    ##########################
    ### DEFINE ENVIRONMENT ###
    ##########################
    bodies_to_create, global_frame_origin, global_frame_orientation = Utils.create_bodies()
    # Create body settings
    body_settings = Utils.get_environmental_settings(bodies_to_create, global_frame_origin, global_frame_orientation, simulation_start_epoch, verification, wind)
    # Create bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)
    # Create the vehicle with all its parameters and its aerodynamic database
    Utils.add_HORUS_to_body_system(bodies, main_sim)
    #Define Initial State
    nominal_initial_state = np.array((bodies.get_body("Earth").shape_model.average_radius + 122e3, np.deg2rad(-22.3), np.deg2rad(-106.7), 7435.5, np.deg2rad(-1.43), np.deg2rad(70.75)))
    initial_state = perturb_initial_conditions( nominal_initial_state ) 
    # Retrieve dependent variables to save
    dependent_variables_to_save = Utils.get_dependent_variable_save_settings( bodies, main_sim, guidance_type, wind, 0, 0, 0, 0, 0, 0 )
    # Define the termination settings 
    termination_settings = Utils.get_termination_settings(simulation_start_epoch,
                                                        maximum_duration,
                                                        termination_altitude,
                                                        termination_angle,
                                                        bodies,
                                                        main_sim,
                                                        guidance_type, 
                                                        wind)
    # Define the body to propagate
    bodies_to_propagate = ["HORUS"]
    # Define central bodies of propagation
    central_bodies = ["Earth"]
    # Define the propagator settings
    propagator_settings = Utils.get_propagator_settings(bodies,
                            simulation_start_epoch,
                            termination_settings,
                            dependent_variables_to_save,
                            initial_state,
                            guidance_type,
                            main_sim,
                            wind,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            trim = True,
                            elevons = True)
                            
    propagator_settings.integrator_settings = Utils.get_integrator_settings(0.5 , simulation_start_epoch, True)
    # Start propagation and keep track of its duration
    start_time = time.time()
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
    )
    end_time = time.time()
    propagation_time = end_time - start_time 
    # Save dependent variables as they are needed to evaluate whether the simulations are successful or not
    dependent_variables = dynamics_simulator.dependent_variable_history
    # Convert the dependent variables from a dictionary to a numpy array
    dependent_variables_array = result2array(dependent_variables)
    # Depending on how the propagation went, we store the results in different folders, based on the given constraints
    max_heat_flux = max( dependent_variables_array[:,24] ) # [kW/m^2]
    max_g_load = max( dependent_variables_array[:,11] ) # [-]
    final_mach = dependent_variables_array[-1,1] # [-]
    final_altitude = dependent_variables_array[-1,2] # [m]
    final_angular_distance = np.rad2deg(dependent_variables_array[-1,33]) # [deg]
    final_energy = np.rad2deg(dependent_variables_array[-1, 37])
    # Define if conditions for storing the results
    if wind : 
        if guidance_type == "Predictor-Corrector":
            if ( max_heat_flux < 530 ) and ( max_g_load < 3.0 ) and ( final_angular_distance < 1 ) and ( final_mach < 3 ) and ( final_altitude < 30000 ) :
                output_path = current_dir + '\\SimulationOutput\\Wind\\PredCorr\\Success\\' + path_definition if write_results_to_file else None
                print('Success')
            else  :
                output_path = current_dir + '\\SimulationOutput\\Wind\\PredCorr\\Fail\\' + path_definition if write_results_to_file else None
                print('Fail')
        elif guidance_type in ("PINN", "PINN_baseline") :
            if ( max_heat_flux < 530 ) and ( max_g_load < 3.0 ) and ( final_angular_distance < 1 ) and ( final_mach < 3 ) and ( final_altitude < 30000 ) :
                output_path = current_dir + f'\\SimulationOutput\\Wind\\{guidance_type}\\Success\\' + path_definition if write_results_to_file else None
                print('Success')
            elif (( max_heat_flux < 530 ) and ( max_g_load < 3.0 )) and (( final_angular_distance > 1 ) or ( final_mach > 3 ) or ( final_altitude > 30000 )) :
                output_path = current_dir + f'\\SimulationOutput\\Wind\\{guidance_type}\\Acceptable\\' + path_definition if write_results_to_file else None
                print('Acceptable')
            else :
                output_path = current_dir + f'\\SimulationOutput\\Wind\\{guidance_type}\\Fail\\' + path_definition if write_results_to_file else None
                print('Fail')
    else :
        if guidance_type == "Predictor-Corrector":
            if ( max_heat_flux < 530 ) and ( max_g_load < 3.0 ) and ( final_angular_distance < 1 ) and ( final_mach < 3 ) and ( final_altitude < 30000 ) :
                output_path = current_dir + '\\SimulationOutput\\NoWind\\PredCorr\\Success\\' + path_definition if write_results_to_file else None
                print('Success')
            else  :
                output_path = current_dir + '\\SimulationOutput\\NoWind\\PredCorr\\Fail\\' + path_definition if write_results_to_file else None
                print('Fail')
        elif guidance_type in ("PINN", "PINN_baseline") :
            if ( max_heat_flux < 530 ) and ( max_g_load < 3.0 ) and ( final_angular_distance < 1 ) and ( final_mach < 3 ) and ( final_altitude < 30000 ) :
                output_path = current_dir + f'\\SimulationOutput\\NoWind\\{guidance_type}\\Success\\' + path_definition if write_results_to_file else None
                print('Success')
            elif (( max_heat_flux < 530 ) and ( max_g_load < 3.0 )) and (( final_angular_distance > 1 ) or ( final_mach > 3 ) or ( final_altitude > 30000 )) :
                output_path = current_dir + f'\\SimulationOutput\\NoWind\\{guidance_type}\\Acceptable\\' + path_definition if write_results_to_file else None
                print('Acceptable')
            else :
                output_path = current_dir + f'\\SimulationOutput\\NoWind\\{guidance_type}\\Fail\\' + path_definition if write_results_to_file else None
                print('Fail')

    # Create the directory if it does not exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save results as .dat and.json files
    Utils.write_propagation_results_to_file(
        dynamics_simulator,
        output_path,
        propagation_time,
        max_heat_flux,
        max_g_load,
        final_mach,
        final_altitude,
        final_angular_distance)

    # ── Optional per-run metrics row (for AI-vs-baseline comparison batches) ────
    if summary_csv is not None :
        import csv
        hf_ok = max_heat_flux < 530 ; g_ok = max_g_load < 3.0
        reached = ( final_angular_distance < 1 ) and ( final_mach < 3 ) and ( final_altitude < 30000 )
        classification = ( 'Success' if (hf_ok and g_ok and reached)
                           else 'Acceptable' if (hf_ok and g_ok)
                           else 'Fail' )
        write_header = not os.path.exists(summary_csv)
        os.makedirs(os.path.dirname(summary_csv), exist_ok=True) if os.path.dirname(summary_csv) else None
        with open(summary_csv, 'a', newline='') as fcsv :
            w = csv.writer(fcsv)
            if write_header :
                w.writerow(['run','guidance_type','wind','max_heat_flux_kW','max_g_load',
                            'final_mach','final_altitude_m','final_angular_distance_deg',
                            'heat_flux_violation','g_load_violation','reached_target','classification',
                            'propagation_time_s'])
            w.writerow([path_definition, guidance_type, int(bool(wind)),
                        float(max_heat_flux), float(max_g_load), float(final_mach),
                        float(final_altitude), float(final_angular_distance),
                        int(not hf_ok), int(not g_ok), int(reached), classification,
                        float(propagation_time)])

    return output_path

def main():

    """
    Standalone batch entry point (single guidance type).
    NOTE: for the paired AI-vs-baseline comparison study use
    run_comparison_batch.py, which drives produce_simulation()
    in memory-isolated subprocesses with paired seeds.
    """

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    wind = False

    # Guidance type: 'Predictor-Corrector', 'PINN' or 'PINN_baseline'
    guidance_type = 'PINN'

    # Define the number of simulations to run
    n_simulations = 1

    for j in range(n_simulations) :
        # Generate random numbers to store results
        random_integer = np.random.randint(1, 1E6)
        # Define unique path definitions for each run
        path_definitions = f"run_{random_integer}"
        produce_simulation( path_definitions, guidance_type, wind )
        logging.info("Simulation completed successfully.")

if run_simulations and __name__ == "__main__":
    main()
