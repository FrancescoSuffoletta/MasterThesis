"""
Author : Francesco Benigno Suffoletta

The function contains all the utility functions that 
are used to model and perform simulations of the HORUS-2B
re-entry vehicle. More in particular, the utilities in here
help with the creation of the bodies (Earth, Horus), the 
setup of acceleration and environmental bodies, interpolators,
aerodynamic description of the vehicle and other various
utilities

"""

import numpy as np
import time
import scipy.integrate as integrate
import os
import json

import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.math import geometry
from tudatpy.util import result2array

from MainGuidance import HORUSGuidanceAndTrim
from HelperGuidance import HORUSHelperGuidance

#Define utility functions 

def create_bodies():

    #Function creates the bodies for the various simulations

    bodies_to_create = ['Earth']
    # Define coordinate system
    global_frame_origin = 'Earth'
    global_frame_orientation = 'J2000'

    return bodies_to_create, global_frame_origin, global_frame_orientation

def get_initial_state(simulation_start_epoch: float,
                      bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                      spherical_initial_state: np.ndarray) -> np.ndarray:
    """
    Converts the initial state to inertial coordinates.

    The initial state is expressed in Earth-centered spherical coordinates.
    These are first converted into Earth-centered cartesian coordinates,
    then they are finally converted in the global (inertial) coordinate
    system.

    Parameters
    ----------
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    spherical_initial_state: array
        6x1 array containing spherical coordinates. It contains, in order:
        distance from Earth c.o.m, latitude, longitude, speed, flight path angle and heading angle.

    Returns
    -------
    initial_state_inertial_coordinates : np.ndarray
        The initial state of the vehicle expressed in inertial coordinates.
    """
    # Set initial spherical elements
    radial_distance = spherical_initial_state[0]
    latitude = spherical_initial_state[1]
    longitude = spherical_initial_state[2]
    speed = spherical_initial_state[3]
    flight_path_angle = spherical_initial_state[4]
    heading_angle = spherical_initial_state[5]

    # Convert spherical elements to body-fixed cartesian coordinates
    initial_cartesian_state_body_fixed = element_conversion.spherical_to_cartesian_elementwise(
        radial_distance, latitude, longitude, speed, flight_path_angle, heading_angle)
    # Get rotational ephemerides of the Earth
    earth_rotational_model = bodies.get_body('Earth').rotation_model
    # Transform the state to the global (inertial) frame
    initial_cartesian_state_inertial = environment.transform_to_inertial_orientation(
        initial_cartesian_state_body_fixed,
        simulation_start_epoch,
        earth_rotational_model)

    return initial_cartesian_state_inertial

def get_termination_settings(simulation_start_epoch: float,
                             maximum_duration: float,
                             termination_altitude: float,
                             termination_angle : float, 
                             bodies,
                             main : bool, 
                             guidance_type : str,
                             wind : bool) \
        -> tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings:
    """
    Get the termination settings for the simulation.

    Termination settings currently include:
    - simulation time (one day)
    - lower and upper altitude boundaries (0-100 km)

    Parameters
    ----------
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    maximum_duration : float
        Maximum duration of the simulation [s].
    termination_altitude : float
        Minimum altitude [m].

    Returns
    -------
    hybrid_termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object.
    """
    # Create single PropagationTerminationSettings objects
    # Time
    time_termination_settings = propagation_setup.propagator.time_termination(
        simulation_start_epoch + maximum_duration,
        terminate_exactly_on_final_condition=True
    )

    if main :
        # Altitude
        lower_altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
            dependent_variable_settings=propagation_setup.dependent_variable.altitude('HORUS', 'Earth'),
            limit_value=termination_altitude,
            use_as_lower_limit=True,
            terminate_exactly_on_final_condition=False)
        # Add Angular Distance as termination settings
        guidance = HORUSGuidanceAndTrim( bodies , guidance_type, wind )
        termination_variable = propagation_setup.dependent_variable.custom_dependent_variable( guidance.get_angular_distance, 1 )

        angular_distance_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings = termination_variable,
        limit_value = np.deg2rad(termination_angle), # rad
        use_as_lower_limit = True)


        # Define list of termination settings
        termination_settings_list = [time_termination_settings,
                                    lower_altitude_termination_settings,
                                    angular_distance_termination_settings]
    else : 

        guidance2 = HORUSHelperGuidance( bodies , 0, 0, 0, 0, 0, 0 )
        termination_variable = propagation_setup.dependent_variable.custom_dependent_variable( guidance2.get_angular_distance, 1 )

        # Altitude
        lower_altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
            dependent_variable_settings=propagation_setup.dependent_variable.altitude('HORUS_2', 'Earth'),
            limit_value=termination_altitude,
            use_as_lower_limit=True,
            terminate_exactly_on_final_condition=False)

        angular_distance_termination_settings = propagation_setup.propagator.dependent_variable_termination(
            dependent_variable_settings = termination_variable,
            limit_value = np.deg2rad(termination_angle), # rad
            use_as_lower_limit = True)

        # Define list of termination settings
        termination_settings_list = [time_termination_settings,
                                    lower_altitude_termination_settings,
                                    angular_distance_termination_settings]
    

    # Create termination settings object (when either the time of altitude condition is reached: propagation terminates)
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                  fulfill_single_condition=True)

    return hybrid_termination_settings

def get_dependent_variable_save_settings( bodies, main, guidance_type, wind, bank_angle, angle_of_attack, bodyflap_deflection, elevons_deflection, previous_bank, current_sign ) -> list:

    """
    Retrieves the dependent variables to save.

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation
    main : bool
        Statement for the main propagation or the helper from the predictor-corrector
    verification : bool
        Statement for the verification or progress runs 
    bank_angle : float 
        In case of helper propagation, constant bank angle for the predictor_corrector
    angle_of_attack : float
        In case of helper propagation, initial angle of attack from the main propagation
    bodyflap_deflection : float
        In case of helper propagation, initial bodyflap deflection from the main propagation
    elevons_deflection : float 
        In case of helper propagation, initial elevons deflection from the main propagation

    Returns
    -------
    dependent_variables_to_save : list[tudatpy.kernel.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.

    """
    
    if main :

        guidance = HORUSGuidanceAndTrim( bodies, guidance_type, wind  )

        dependent_variables_to_save = [propagation_setup.dependent_variable.mach_number('HORUS', 'Earth'), #1
                                    propagation_setup.dependent_variable.altitude('HORUS', 'Earth'), #2
                                    propagation_setup.dependent_variable.total_acceleration_norm('HORUS'), #3
                                    propagation_setup.dependent_variable.latitude('HORUS', 'Earth'), #4
                                    propagation_setup.dependent_variable.longitude('HORUS', 'Earth'), #5
                                    propagation_setup.dependent_variable.heading_angle('HORUS', 'Earth'), #6
                                    propagation_setup.dependent_variable.flight_path_angle('HORUS', 'Earth'), #7
                                    propagation_setup.dependent_variable.angle_of_attack('HORUS', 'Earth'), #8
                                    propagation_setup.dependent_variable.bank_angle('HORUS', 'Earth'), #9
                                    propagation_setup.dependent_variable.sideslip_angle('HORUS', 'Earth'), #10
                                    propagation_setup.dependent_variable.local_aerodynamic_g_load('HORUS', 'Earth'), #11
                                    propagation_setup.dependent_variable.density('HORUS','Earth'), #12
                                    propagation_setup.dependent_variable.temperature('HORUS', 'Earth'), #13
                                    propagation_setup.dependent_variable.dynamic_pressure('HORUS', 'Earth'), #14
                                    propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.aerodynamic_type, 'HORUS', 'Earth'), #15
                                    propagation_setup.dependent_variable.aerodynamic_force_coefficients('HORUS'), # 16, 17, 18
                                    propagation_setup.dependent_variable.aerodynamic_moment_coefficients('HORUS'),# 19, 20, 21
                                    propagation_setup.dependent_variable.airspeed('HORUS', 'Earth'), #22
                                    propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.spherical_harmonic_gravity_type, 'HORUS', 'Earth'), #23
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance.computeHeatFlux, 1 ), #24
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance.get_bodyflap_deflection, 1 ), #25 
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance.get_bodyflap_contribution, 3 ), #26, 27, 28
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance.get_elevons_deflection, 1 ), #29
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance.get_elevons_contribution, 3 ), #30, 31, 32
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance.get_angular_distance, 1 ), #33
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance.get_max_deadband_error, 1 ), #34
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance.get_current_heading_error, 2 ), #35, 36
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance.get_current_energy, 1 ), # 37
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance.updateData, 3 )] # 40, 41, 42
    else : 

        guidance2 = HORUSHelperGuidance( bodies, bank_angle, angle_of_attack, bodyflap_deflection, elevons_deflection, previous_bank, current_sign )

        dependent_variables_to_save = [propagation_setup.dependent_variable.mach_number('HORUS_2', 'Earth'), #1
                                    propagation_setup.dependent_variable.altitude('HORUS_2', 'Earth'), #2
                                    propagation_setup.dependent_variable.total_acceleration_norm('HORUS_2'), #3
                                    propagation_setup.dependent_variable.latitude('HORUS_2', 'Earth'), #4
                                    propagation_setup.dependent_variable.longitude('HORUS_2', 'Earth'), #5
                                    propagation_setup.dependent_variable.heading_angle('HORUS_2', 'Earth'), #6
                                    propagation_setup.dependent_variable.flight_path_angle('HORUS_2', 'Earth'), #7
                                    propagation_setup.dependent_variable.angle_of_attack('HORUS_2', 'Earth'), #8
                                    propagation_setup.dependent_variable.bank_angle('HORUS_2', 'Earth'), # 9
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance2.get_current_energy, 1 ), #10
                                    propagation_setup.dependent_variable.airspeed('HORUS_2', 'Earth'), #11
                                    propagation_setup.dependent_variable.local_aerodynamic_g_load('HORUS_2', 'Earth'), #12
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance2.computeHeatFlux, 1 ), #13
                                    propagation_setup.dependent_variable.custom_dependent_variable( guidance2.updateData2, 3 )] # 13, 14, 15

    return dependent_variables_to_save


def get_integrator_settings(fixed_step_size: float,
                            simulation_start_epoch: float,
                            verification : bool) \
                                -> tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings:
    
    """
    Function provides the settings for the integrator used within the simulations

    Parameters
    ----------
    fixed_step_size : float
        Time Step size to be used during the propagations 
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    verification : bool
        Boolean statement accounting for verification or simulation runs

    Returns
    -------
    integrator_settings : object[tudatpy.kernel.numerical_simulation.propagation_setup.integrator.runge_kutta_4]
        Settings for the integrator to be used for the simulations

    """


    integrator = propagation_setup.integrator

    if verification :
        integrator_settings = integrator.runge_kutta_4(simulation_start_epoch,
                                                    fixed_step_size)
    else :
        coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_89
        tolerance = 10.0 ** -14.0
        integrator_settings = integrator.runge_kutta_variable_step_size(simulation_start_epoch,
                                                                        0.5,
                                                                        coefficient_set,
                                                                        np.finfo(float).eps,
                                                                        np.inf,
                                                                        tolerance,
                                                                        tolerance)

    return integrator_settings

def get_propagator_settings(bodies,
                            simulation_start_epoch,
                            termination_settings,
                            dependent_variables_to_save,
                            spherical_initial_state,
                            guidance_type,
                            main,
                            wind,
                            bank_angle,
                            angle_of_attack,
                            bodyflap_deflection,
                            elevons_deflection,
                            previous_bank_angle,
                            sign_bank,
                            trim = True,
                            elevons = True,
                            current_propagator = propagation_setup.propagator.cowell):
    """
    The function defines the propagator settings for the HORUS-2B propagation. Note that for now, 
    a simple aerodynamic definition with constant angles is here displayed. Nonetheless,
    it needs to be integrated with the guidance of the vehicle

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object.
    dependent_variables_to_save : list[tudatpy.kernel.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    spherical_initial_state : np.ndarray
        Array containing the initial state of the vehicle expressed in spherical coordinates
    verification : bool
        Boolean statement to check whether it is a verification of research simulation
    main : bool
        Boolean statement to produce settings either for the main or the pred-corr helper guidance
    bank_angle : float
        Constant value of the bank angle to provide for the predictor-corrector helper guidance
    angle_of_attack : float
        Current value of the AoA to initialize the predictor part of the guidance law
    bodyflap_deflection : float
        Current value of the bodyflap deflection to initialize the predictor part of the guidance law
    elevons_deflection : float
        Current value of the elevons deflection to intialize the predictor part of the guidance law
    current_propagator : object[tudatpy.kernel.numerical_simulation.propagation_setup.propagator.cowell]
        Propagator to be used for the simulations 

    Returns
    -------
    propagator_settings : list[tudatpy.kernel.numerical_simulation.propagation_setup.propagator.translational]
        List of settings for the propagator.

    """

    pitch_trim = trim
    use_elevons = elevons

    acceleration_settings_on_vehicle = {
        'Earth': [propagation_setup.acceleration.aerodynamic(),
                  propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)]
    }

    # Create the aerodynamic guidance object, it changes whether we are currently
    # propagating in the main or the pred-corr cycle 
    if main : 
        # Define bodies that are propagated and their central bodies of propagation
        bodies_to_propagate = ['HORUS']
        central_bodies = ['Earth']
        # Define the accelerations on the vehicle
        acceleration_settings = {'HORUS': acceleration_settings_on_vehicle}
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies,
            acceleration_settings,
            bodies_to_propagate,
            central_bodies)
        # Define the guidance model
        aerodynamic_guidance_object = HORUSGuidanceAndTrim( bodies , guidance_type, wind )
        rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
            'Earth', '', 'HORUS_Fixed', aerodynamic_guidance_object.getAerodynamicAngles )
        environment_setup.add_rotation_model( bodies, 'HORUS', rotation_model_settings )
    else : 
        # Define bodies that are propagated and their central bodies of propagation
        bodies_to_propagate = ['HORUS_2']
        central_bodies = ['Earth']
        # Define the accelerations on the vehicle
        acceleration_settings = {'HORUS_2': acceleration_settings_on_vehicle}
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies,
            acceleration_settings,
            bodies_to_propagate,
            central_bodies)
        aerodynamic_guidance_object = HORUSHelperGuidance( bodies , bank_angle, angle_of_attack, bodyflap_deflection, elevons_deflection, previous_bank_angle, sign_bank )
        rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
            'Earth', '', 'HORUS2_Fixed', aerodynamic_guidance_object.getAngles )
        environment_setup.add_rotation_model( bodies, 'HORUS_2', rotation_model_settings )
        
    # Retrieve initial state
    initial_state = get_initial_state(simulation_start_epoch,bodies,spherical_initial_state)

    propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                     acceleration_models,
                                                                     bodies_to_propagate,
                                                                     initial_state,
                                                                     simulation_start_epoch,
                                                                     None,
                                                                     termination_settings,
                                                                     current_propagator,
                                                                     output_variables=dependent_variables_to_save)
    return propagator_settings

def get_environmental_settings(bodies_to_create,
                                global_frame_origin : str,
                                global_frame_orientation : str,
                                initial_time : float,
                                verification : bool, 
                                wind : bool ):
    
    """
    The function defines the environmental settings for the simulations to come as described in the third 
    chapter of the thesis.

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    global_frame_origin : str
        Defines the frame origin of the central body
    global_frame_orientation : str
        Defines the frame orientation of the central body
    initial_time : float
        Start of the simulation [s] with t=0 at J2000.
    wind : bool
        Whether to include wind in the simulations

    Returns
    -------
    body_settings = object[tudatpy.kernel.numerical_simulation.environment_setup.body_settings]
        List of the body settings for the environment where to simulate the trajectory of HORUS 2-B

    """

    body_settings = environment_setup.get_default_body_settings(bodies_to_create,
        global_frame_origin,
        global_frame_orientation)

    #Select the desired atmospheric model for the propagation
    body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.us76()

    #Define parameters for the Earth's oblate spheroid shape, then select it for the environment
 
    body_radius = 6371.0E3
    if verification :
        flattening_parameter = 0.0
    else : 
        flattening_parameter = 1 / 298.257223563

    body_settings.get("Earth").shape_settings = environment_setup.shape.oblate_spherical(body_radius, flattening_parameter)

    #Define the parameters for the Earth's rotation model from SPICE
    original_frame = "J2000"
    target_frame = "IAU_Earth"
    
    body_settings.get("Earth").rotation_model_settings = environment_setup.rotation_model.simple(original_frame, 
            "Earth_Fixed_Simplified",
            np.ones((3,3)),
            0,
            7.292115E-5)

    
    body_settings.get("Earth").gravity_field_settings.associated_reference_frame = "Earth_Fixed_Simplified"

    #To be changed when needed
    if wind :
        # Create the custom wind settings
        custom_wind = environment_setup.atmosphere.custom_wind_model(
            wind_function,
            environment.AerodynamicsReferenceFrames.vertical_frame)
        # Apply the custom wind settings to the Earth atmosphere settings
        body_settings.get("Earth").atmosphere_settings.wind_settings = custom_wind
    else : 
        pass
    
    return body_settings


def wind_function(h, lon, lat, time):

    """
    # Meridional wind (pointing North) depends on latitude [deg] and time [sec since J2000]
    if time > 0  :
        wind_Xv = lat*10/time
    else : 
        wind_Xv = 0.0
    # Zonal wind (pointing West) only depends on the longitude [deg]
    wind_Yv = 5/lon
    # Vertical wind (pointing out of the centre of the Earth) only depends on the altitude [m]
    wind_Zv = 1000/h
    # Return the custom wind

    """
    
    # Wind amplitudes [m/s]. Reduced from 20 m/s (which made even the baseline fail
    # ~33% and is unphysically strong). The VERTICAL component (z) is kept small: real
    # vertical winds are ~0 and it maps almost directly onto the flight-path angle, so
    # it is the most disruptive/least realistic term. Tune these to set wind intensity.
    amplitude_x = 5.0
    amplitude_y = 5.0
    amplitude_z = 1.0

    # Frequencies / scales for variation
    # e.g. one cycle every ~600 seconds (10 minutes) plus small dependence on lat/lon
    freq_time = 2.0 * np.pi / 60.0  # ~1-minute period
    freq_lat  = 0.2
    freq_lon  = 0.2

    # Example of altitude scale for vertical wind 
    # (the fraction h/1.0e4 modifies the phase or amplitude based on altitude)
    alt_scale = h / 1.0e4

    # X-component: mild sinusoidal function of time + latitude
    wind_Xv = amplitude_x * np.sin(freq_time * time + freq_lat * lat)

    # Y-component: mild sinusoidal function of time + longitude
    wind_Yv = amplitude_y * np.cos(freq_time * time + freq_lon * lon)

    # Z-component: mild sinusoidal function of time + altitude
    wind_Zv = amplitude_z * np.sin(freq_time * time + alt_scale)

    return [wind_Xv, wind_Yv, wind_Zv]

#########################################################################################
####### WE DEFINE HERE THE AERODYNAMIC COEFFICIENTS AND PARAMETERS OF THE HORUS2B #######
#########################################################################################

# The following two functions are needed to read the aerodynamic database of HORUS-2B

def extract_2D_database_HORUS(file_path: str) -> list[ np.ndarray , np.ndarray , np.ndarray ] :

    """
    The function extracts and reads the aerodynamic files for the HORUS 2-B. More precisely, 
    the 2D files to be read are the aerodynamic coefficients of force and momentum,
    given a discretized set of Mach numbers and angles of attack.

    Parameters
    ----------
    file_path : str
        The file path where to find the desired .txt files
    
    Returns
    -------
    table_2D : np.ndarray
        2D table containing the aerodynamic coefficients of HORUS-2B
    var1_lst : np.ndarray
        1-D array containing the independent values on the rows of the matrix
    var2_lst
        1-D array containing the independent values on the columns of the matrix
    
    """

    try:

        with open(file_path) as f:

            n_independent_variables = int(f.readline())

            assert n_independent_variables == 2, (
                f"Expected 2 independent variables in Utilis.extract_2D_database_HORUS, "
                f"but {n_independent_variables} were found." 
                )

            # Get range of var_1
            var1_lst_str = f.readline()
            var1_lst = np.fromstring(var1_lst_str, dtype=float, sep=' ')

            # Get range of var_2
            var2_lst_str = f.readline()
            var2_lst = np.fromstring(var2_lst_str, dtype=float, sep=' ')

            # Skip empty line
            f.readline()

            # Create database as np.ndarray
            table_2D = np.empty(( var1_lst.size , var2_lst.size ))

            # Fill database row by row
            for row in range( var1_lst.size ):
                current_row_str = f.readline()
                table_2D[row,:] = np.fromstring(current_row_str, dtype=float, sep=' ')

    except AssertionError as error:
        print(error)

    except FileNotFoundError:
        print(f"File not found: {file_path}")

    except ValueError:
        print(f"Could not convert the line to an integer.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return table_2D, var1_lst, var2_lst


def extract_3D_database_HORUS(file_path: str) -> list[ dict , np.ndarray , np.ndarray ] :

    """
    The function extracts and reads the aerodynamic files for the HORUS 2-B. More precisely, 
    the 3D files to be read are the aerodynamic coefficients of force and momentum,
    given a discretized set of Mach numbers, angles of attack and control deflection angles.

    Parameters
    ----------
    file_path : str
        The file path where to find the desired .txt files

    Returns
    -------
    database : dict
        Dictionary that relates a 2D database with AoA and deflection of control surfaces to each Mach number
    var1_lst : ndarray
        List of independent variables corresponding to each row of 2D database
    var2_lst : ndarray
        List of independent variables corresponding to each row of 2D database

    """

    try:
        with open(file_path) as f:

            n_independent_variables = int(f.readline())

            assert n_independent_variables == 3, (
                f"Expected 3 independent variables in Utilis.extract_3D_database_HORUS, "
                f"but {n_independent_variables} were found." 
                )

            # Get range of var_1
            var1_lst_str = f.readline()
            var1_lst = np.fromstring(var1_lst_str, dtype=float, sep=' ')

            # Get range of var_2
            var2_lst_str = f.readline()
            var2_lst = np.fromstring(var2_lst_str, dtype=float, sep=' ')
            
            # Get range of var_3
            var3_lst_str = f.readline()
            var3_lst = np.fromstring(var3_lst_str, dtype=float, sep=' ')

            # Create the dictionary
            database = {}

            # Fill the dictionary looping over the various Mach numbers
            for v in var3_lst:

                # Skip empty line
                f.readline()

                # Create 2D database as np.ndarray
                table_2D = np.empty(( var1_lst.size , var2_lst.size ))

                # Fill database row by row
                for row in range( var1_lst.size ):
                    current_row_str = f.readline()
                    table_2D[row,:] = np.fromstring(current_row_str, dtype=float, sep=' ')

                # Add 2D table to current var3 key in database dictionary
                database[v] = table_2D

    except AssertionError as error:
        print(error)

    except FileNotFoundError:
        print(f"File not found: {file_path}")

    except ValueError:
        print(f"Could not convert the line to an integer.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return database , var1_lst, var2_lst
 

def add_HORUS_to_body_system(bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                            main : bool):
    
    """
    It creates the HORUS body object and adds it to the body system, setting its parameters and aerodynamic interface based on the 
    reference parameters of the winged vehicle

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    main : bool
        Flag to specify whether we are in the main or the pred-corr cycle propagations

    Returns
    -------
    none

    """
    # Define HORUS parameters
    HORUS_mass = 26029.0 #kg
    HORUS_ref_area = 110.0 #m^2
    HORUS_ref_length = 23 #m

    # Specify whether we are creating the bodies in the main propagation or the helper one
    if main : 
        # Create new vehicle object and add it to the existing system of bodies
        bodies.create_empty_body('HORUS')
        bodies.get_body('HORUS').set_constant_mass(HORUS_mass)
    else : 
        bodies.create_empty_body('HORUS_2')
        bodies.get_body('HORUS_2').set_constant_mass(HORUS_mass)

    #SET UP FUNCTION FOR THE AERODYNAMIC INTERFACE !
    current_dir = os.path.dirname(__file__)
    
    CD_file = os.path.join(current_dir, 'body_coefficients', 'CD0.txt')
    CL_file = os.path.join(current_dir, 'body_coefficients', 'CL0.txt')
    Cm_file = os.path.join(current_dir, 'body_coefficients', 'Cm0.txt')

    CDb_file = os.path.join(current_dir, 'ctrl_surf_coefficients', 'dCDb.txt')
    CLb_file = os.path.join(current_dir, 'ctrl_surf_coefficients', 'dCLb.txt')
    Cmb_file = os.path.join(current_dir, 'ctrl_surf_coefficients', 'dCmb.txt')

    CDe_file = os.path.join(current_dir, 'ctrl_surf_coefficients', 'dCDw.txt')
    CLe_file = os.path.join(current_dir, 'ctrl_surf_coefficients', 'dCLw.txt')
    Cme_file = os.path.join(current_dir, 'ctrl_surf_coefficients', 'dCmw.txt')

    #Here we create the interface first. If the vehicle is trimmed it is alright, if not, trim the vehicle!
    get_HORUS_coefficient_interface( bodies, main, CL_file, CD_file, Cm_file, CDb_file, CLb_file, Cmb_file, CDe_file, CLe_file, Cme_file, HORUS_ref_area, HORUS_ref_length ) 

def get_HORUS_coefficient_interface(bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies, 
                                    main : bool,
                                    pathCL0 : str, 
                                    pathCD0 : str, 
                                    pathCm0 : str,  
                                    pathCDb : str, 
                                    pathCLb : str, 
                                    pathCmb : str,
                                    pathCDe : str,
                                    pathCLe : str,
                                    pathCme : str, 
                                    HORUS_ref_area : float, 
                                    HORUS_ref_length : float):

    """ 
    The function creates the baseline aerodynamic interface of the HORUS-2B vehicle.

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    main : bool
        Flag to specify whether we are in the main or the pred-corr cycle propagations
    pathCL0 : str
        It is the path to reach the .txt files of the CL0 database
    pathCD0 : str
        It is the path to reach the .txt files of the CD0 database
    pathCm0 : str
        It is the path to reach the .txt files of the Cm0 database
    pathCDb : str 
        It is the path to reach the .txt  files of the CDb database
    pathCLb : str 
        It is the path to reach the .txt  files of the CDb database
    pathCmb : str 
        It is the path to reach the .txt  files of the Cmb database 
    pathCDe : str 
        It is the path to reach the .txt  files of the CDe database  
    pathCLe : str 
        It is the path to reach the .txt  files of the CLe database  
    pathCme : str 
        It is the path to reach the .txt  files of the Cme database
    HORUS_ref_area : float
        Reference area of HORUS-2B  
    HORUS_ref_length : float
        Reference length of the HORUS-2B

    Returns
    -------
    None

    """

    #Define the files for the force coefficients (Note that the side force coefficients are neglected)
    force_coefficients_files = {0: pathCD0, 2: pathCL0}
    #Define the files for the moment coefficients (Note that only the Cm is introduced here)
    moment_coefficients_files = {0: pathCm0}

    #Define now the control surface files
    force_coefficients_files_b = {0: pathCDb, 2: pathCLb}
    #Define the files for the moments of the control surface
    moment_coefficients_files_b = {0: pathCmb}

    force_coefficients_files_e = {0: pathCDe, 2: pathCLe}
    #Define the files for the moments of the control surface
    moment_coefficients_files_e = {0: pathCme}

    coefficient_settings_0 = environment_setup.aerodynamic_coefficients.tabulated_from_files(
        force_coefficient_files=force_coefficients_files,
        moment_coefficient_files=moment_coefficients_files,
        reference_length=HORUS_ref_length,
        reference_area=HORUS_ref_area,
        independent_variable_names=[environment.angle_of_attack_dependent, environment.mach_number_dependent]
    )

    coefficient_settings_b = environment_setup.aerodynamic_coefficients.tabulated_from_files_control_surface(
        force_coefficient_files=force_coefficients_files_b,
        moment_coefficient_files=moment_coefficients_files_b,
        independent_variable_names=[environment.angle_of_attack_dependent, environment.control_surface_deflection_dependent, environment.mach_number_dependent]
    )

    coefficient_settings_e = environment_setup.aerodynamic_coefficients.tabulated_from_files_control_surface(
        force_coefficient_files=force_coefficients_files_e,
        moment_coefficient_files=moment_coefficients_files_e,
        independent_variable_names=[environment.angle_of_attack_dependent, environment.control_surface_deflection_dependent, environment.mach_number_dependent]
    )

    coefficient_settings_0.add_single_control_surface(control_surface_settings = coefficient_settings_b, control_surface_name = "body_flap")

    coefficient_settings_0.add_single_control_surface(control_surface_settings = coefficient_settings_e, control_surface_name = "elevons")

    if main : 
        environment_setup.add_aerodynamic_coefficient_interface(bodies, "HORUS", coefficient_settings_0)
    else :
        environment_setup.add_aerodynamic_coefficient_interface(bodies, "HORUS_2", coefficient_settings_0)



def compute_Heat_Flux(airspeed : float,
                    density : float,
                    kW : bool = True):
        """
        The function computes the heat flux as a result during the propagation of
        the HORUS vehicle.

        Parameters
        ----------
        airspeed : float
            Current airspeed of the vehicle
        density : float
            Atmospheric density at the current altitude
        
        Returns
        -------
        q_dot : float
            Current incoming heat flux

        """

        # Define the parameters that are required
        HORUS_RN = 0.8 # m Nose Radius
        c_star = 1.1097E8 # Aerodynamic Characteristic
        Vc = 7.9054E3 # Circular Velocity 
        n = 0.5
        m = 3.15
        rho_0 = 1.225 # kg/m^3

        q_dot = float((c_star*(1/((HORUS_RN)**n))*((density/rho_0)**(1-n))*(airspeed/Vc)**m))

        if kW:
            q_dot = float((q_dot / 1000))

        return q_dot

def compute_Heat_Load(q_dot : np.ndarray,
                    time_series : np.ndarray):

    """
    The function returns the value of the Q_load that the HORUS has undergone throughout the propagation

    Parameters
    ----------
    q_dot : ndarray
        Values of the different instantaneous heat fluxes
    time_series : ndarray 
        Time series related to the values of the heat flux

    """

    Q_load = integrate.simpson(q_dot, x=time_series)

    return Q_load

#######################################
####### INTERPOLATION FUNCTIONS #######
#######################################

def OneD_interpolation(value : float,
                    ind_var : np.ndarray,
                    dep_var : np.ndarray,
                    decreasing : bool = False) -> float :

    """
    The function interpolates a list of dependent variables, 
    given a list of independent variables and one value from
    the independent ones to interpolate one from the list of
    the dependent one. If the list is in the decreasing order
    then the decreasing boolean option shall be set to true.

    Parameters
    ----------
    value : float
        value for the interpolation of the ind. variables list
    ind_var : np.ndarray
        list of independent variables
    dep_var : np.ndarray
        list of dependent variables
    
    Return
    ------
    interpolated_value : float
        Interpolated value from the list of dependent variables
    
    """

    assert ind_var.shape[0] == dep_var.shape[0] , (
     f"Error in Utils.OneD_interpolation: dep_var has {dep_var.shape[0]} entries, " 
     f"but list has {ind_var.shape[0]} entries." )

    if decreasing:
        sign = -1
    else:
        sign = 1
    
    # 3 cases: If the index is in the middle of the provided ones, we can interpolate, while if
    # it is outside the bounds, we just take the extreme values (no extrapolation is introduced)
    if sign*value > sign*ind_var[0] and sign*value < sign*ind_var[-1]:
        # Find indexes of the values that "surround" requested value
        idx_1 = np.where(sign*value >= sign*ind_var)[0][-1]
        idx_2 = idx_1 + 1
        interpolated_value = dep_var[ idx_1 ] + ( dep_var[ idx_2 ] - dep_var[ idx_1 ] ) * ( value - ind_var[idx_1] ) / ( ind_var[idx_2] - ind_var[idx_1] )
    elif sign*value <= sign*ind_var[0]:
        interpolated_value = dep_var[0]
    else:
        interpolated_value = dep_var[-1]

    return interpolated_value


def TwoD_interpolation( database : dict,
                   var1 : float,
                   var2 : float ):
    '''
    Receive a 2D database, stored in a dictionary, and interpolates it to a single value.

    Expecting database as follows:
    
    database = { 'var1_lst' : var1_lst,  # where var1_lst is of type np.ndarray
                 'var2_lst' : var2_lst,  # where var2_lst is of type np.ndarray
                 'table_2D' : table_2D } # where table_2D is of type np.ndarray
    '''

    # Extract information from database
    var1_lst = database.get('var1_lst')
    var2_lst = database.get('var2_lst')
    table_2D = database.get('table_2D')

    row = TwoD_2_OneD_interpolation( var1 , var1_lst , table_2D )

    var2_lst_decreasing = False
    if var2_lst[-1] < var2_lst[0]:
        var2_lst_decreasing = True

    output = OneD_interpolation( var2 , var2_lst , row , var2_lst_decreasing )

    return output

def TwoD_2_OneD_interpolation( value : float,
                              ind_var : np.ndarray,
                              table_2D : np.ndarray,
                              T : bool = False,
                              decreasing : bool = False ) -> np.ndarray:
    '''
    Interpolate a 2D aerodynamic database to a 1D row

    Parameters
    ---------
    value : float
        Value (independent variable) where we want to perform the interpolation
    list : np.ndarray
        List of independent variables where 'value' is inserted
    table_2D : np.ndarray
        2D table (dependent variable) that we want to interpolate
    list_decreasing : bool
        Flag to indicate if 'list' is decreasing, which will invert the comparisons that we perform

    Returns
    --------
    row : np.ndarray
        Interpolated row of table_2D corresponding to the first independent variable being 'value'
    '''

    if T: 
        table_2D = np.transpose(table_2D)

    assert table_2D.shape[0] == ind_var.shape[0] , \
     f"Error in simulationUtilities.interpolate_2D_to_1D_row: table_2D has {table_2D.shape[0]} rows, " \
     f"but list has {ind_var.shape[0]} entries."

    # Set sign for comparisons (because we need to invert the comparisons if 'list' is decreasing, i.e., a "greater than" becomes "less than", and vice versa)
    sign = -1 if decreasing else 1
    # If the value is in the middle of the independent variable list, we interpolate the 2 rows that contain the requested variable
    if sign*value > sign*ind_var[0] and sign*value < sign*ind_var[-1]:
        # Find indexes of the rows that "surround" requested value
        idx_1 = np.where(sign*value >= sign*ind_var)[0][-1]
        idx_2 = idx_1 + 1

        # Get corresponding rows
        # NOTE: we use np.array because we want the following behaviour: [1,2] + [3,4] = [4,6], not [1,2,3,4]
        row_1 = np.array( table_2D[ idx_1 , :] )
        row_2 = np.array( table_2D[ idx_2 , :] )

        # Interpolate table_2D
        row = row_1 + ( row_2 - row_1 ) * ( value - ind_var[ idx_1 ] ) / ( ind_var[ idx_2 ] - ind_var[ idx_1 ] )

    # If the variable is at the lower limit of the list, we output the first row (no extrapolation)
    elif sign*value <= sign*ind_var[0]:
        row = np.array( table_2D[0 , :] )

    # Otherwise, the variable is at the upper limit of the list, so we output the last row (also no extrapolation)
    else:
        row = np.array( table_2D[-1 , :] )

    return row


def ThreeD_interpolate(Database,
                    value_to_interpolate : float) -> np.ndarray :

    """
    The function provides a 3D interpolation to bring back the 2D table of 
    C_m given the current Mach number, in dependance of AoA and control 
    surface deflection

    Parameters
    ----------
    Database : dict
        Dictionary with the aerodynamic database for the control surfaces
    value_to_interpolate : float
        Value needed for the interpolation 

    Returns
    -------
    cm_matrix : np.ndarray
        2D database with the interpolated values of the pitch moment coefficient wrt AoA and deflection

    """

    mach_numbers = []

    for key in Database.keys():
        mach_numbers.append(key)

    # We find the array of the mach numbers within the dictionary of the database
    mach_numbers = np.array(mach_numbers)

    # Function finds the previous and the following indices of the 
    previous_index, following_index = find_indices(mach_numbers, value_to_interpolate)

    num_rows, num_cols = Database[mach_numbers[-1]].shape
    cm_matrix = np.zeros((num_rows, num_cols))

    # Three possible events here : the value to interpolate is higher than the highest tabulated value, lower than 
    # the lowest tabulated value or it can be interpolated. If it can be interpolated, then there are two subcases:
    # in the first, the previous and the following indices differ one from another thus the matrix can be interpolated,
    # while in the second the values are the same, meaning that the flight conditions are exactly the same as those that
    # are tabulated.

    if previous_index == None : 
        cm_matrix = Database[mach_numbers[0]]
    elif following_index == None : 
        cm_matrix = Database[mach_numbers[-1]]
    else: 
        if previous_index != following_index: 
            for i in range(num_rows):
                for j in range(num_cols):
                    cm_matrix[i,j] = Database[mach_numbers[previous_index]][i,j] + (value_to_interpolate - mach_numbers[previous_index]) * ((Database[mach_numbers[following_index]][i,j] - Database[mach_numbers[previous_index]][i,j])/(mach_numbers[following_index] - mach_numbers[previous_index]))
        else: 
            cm_matrix = Database[mach_numbers[following_index]]
    return cm_matrix


def find_indices(mach_number : np.ndarray, 
                value_to_interpolate : float):

    """
    The function looks for the indices that anticipate and follow the mach number 
    to interpolate in the 3D interpolation.

    Parameters 
    ----------
    mach_number : np.ndarray
        Array of Mach numbers
    value_to_interpolate : float
        Value for the interpolation
    
    Returns
    -------
    low, high : float
        The required indices

    """ 

    low, high = 0, len(mach_number) - 1
    
    # If the value is out of bounds of the array
    if value_to_interpolate < mach_number[0]:
        return None, 0  # No previous value, first element is the following value
    if value_to_interpolate > mach_number[-1]:
        return len(mach_number) - 1, None  # Last element is the previous value, no following value
    
    # Perform binary search
    while low <= high:
        mid = (low + high) // 2
        if mach_number[mid] < value_to_interpolate:
            low = mid + 1
        elif mach_number[mid] > value_to_interpolate:
            high = mid - 1
        else:
            # If the value is exactly at mid, return mid for both previous and following
            return mid, mid
    
    # At the end of loop, low is the index of the first element greater than the value
    # and high is the index of the last element less than the value
    return high, low


def create_database(var1_lst : np.ndarray,
                    var2_lst : np.ndarray, 
                    table_2D : np.ndarray):
    """
    The function creates a database in the form wanted by
    the Interp_2D function

    """
    var1_lst = np.array(var1_lst)
    var2_lst = np.array(var2_lst)
    table_2D = np.array(table_2D)
    
    # Create the dictionary
    database = {
        'var1_lst': var1_lst,
        'var2_lst': var2_lst,
        'table_2D': table_2D
    }
    
    return database

#######################
######VERIFICATION#####
#######################

def read_gnu_files( validation_path : str , file_name : str ) -> dict :
    '''
    Routine to read the .gnu trajectory files

    Parameters
    --------
    validation_path : str
        Path where the .gnu files are stored
    file_name : str
        Name of the current .gnu file to read
    
    Returns
    --------
    dict
        Dictionary relating the variable name to the np.array containing the corresponding variable history
    '''

    # Initialise dictionary of columns and respective variables
    thisFileFormat = {}

    # Check file names
    if file_name == 'forces' or file_name == 'horus2bus76':

        # Read "readMe" file
        with open( validation_path + "/" + file_name + ".txt" ) as f :

            # Read first line to get number of variables
            n_vars_line = f.readline().strip() # The 'strip' method is used to get rid of the empty spaces before and after the useful text
            n_vars = int( n_vars_line[-2:] )

             # Skip 2 empty lines
            f.readline()
            f.readline()

            # Loop over number of variables
            for _ in range( n_vars ):
                # Skip empty line
                f.readline()
                # Read current line
                current_line = f.readline().strip()

                # Get index of column corresponding to the current variable
                thisColumn = int( current_line[2:4] )
                
                # Get name of current variable
                thisVariable = current_line[5 : -3].strip()

                thisFileFormat[thisColumn] = thisVariable

    else:

        # If requested file does not exist, return None
        print(f"File {file_name}.txt does not exist")
        
        return None

    # Read results file (corresonding to file_name)
    results = np.genfromtxt( validation_path + "/" + file_name + ".gnu" )

    # Initialise a dictionary containing the variable name and the corresponding array
    thisDict = {}

    # Store data corresponding to each column / each variable
    for thisColumn in thisFileFormat.keys():
        # Actual index starts at 0 (but "thisColumn" starts at 1)
        thisColumn_index = thisColumn - 1
        # Fill dictionary: associate current variable name to respective column of "results" matrix
        thisDict[ thisFileFormat.get(thisColumn) ] = results[ : , thisColumn_index ]


    return thisDict

def write_propagation_results_to_file(
    dynamics_simulator : numerical_simulation.SingleArcSimulator,
    output_path : str,
    propagation_time : float,
    max_heat_flux : float,
    max_g_load : float,
    final_mach, 
    final_altitude, 
    final_angular_distance) :

    """

    This function will write the results of a numerical propagation, including the state history, 
    the dependent variables and the time taken to conduct the propagation.

    Parameters
    ----------
    dynamics_simulator : numerical_simulation.SingleArcSimulator
        Object that was used to propagate the dynamics, and which contains the numerical state and dependent
        variable results
    output_path : str
        Directory where to store the filename
    propagation_time : float
        Time taken to perform the simulation

    Files written
    -------------
    <file_output_identifier>_numerical_states.dat
    <file_output_identifier>_dependent_variables.dat

    Return
    ------
    None

    """

    state_history = dynamics_simulator.state_history
    dependent_variable_history = dynamics_simulator.dependent_variable_history

    # Save numerical states
    save2txt(solution=state_history,
            filename=output_path + "\\Numerical_States.dat",
            directory="./")

    # Save dependent variables
    dependent_variables = dependent_variable_history
    if len(dependent_variables.keys()) > 0:
        save2txt(solution=dependent_variables,
                filename=output_path + "\\Dependent_Variables.dat",
                directory="./")

    simulation_results = {
        "Propagation Time [s]": propagation_time,
        "Maximum Heat Flux [kW/m^2]": max_heat_flux,
        "Maximum G-Load [g]": max_g_load,
        "Final Mach Number [-]": final_mach,
        "Final Altitude [m]": final_altitude,
        "Final Angular Distance [deg]": final_angular_distance
    }

    output_results = os.path.join(output_path, "Results.json")

    with open(output_results, 'w') as f:
        json.dump(simulation_results, f, indent=4)

    return