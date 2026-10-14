import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import json

from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, environment, propagation_setup
from tudatpy.math import interpolators
from tudatpy.util import result2array
from tudatpy.kernel import constants

# Obtain the path for the root of this 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import Utils
from MainGuidance import HORUSGuidanceAndTrim
from HelperPlot import HORUS2B_Plotting

Verify_Trajectory = True
Verify_Forces = False


#Start set-up of the basic simulations, including bodies, vehicle model, environment and acceleration
spice_interface.load_standard_kernels()

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

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

bodies_to_create, global_frame_origin, global_frame_orientation = Utils.create_bodies()

verification = True
main = True
guidance_type = "Verification"
wind = False

body_settings = Utils.get_environmental_settings(bodies_to_create, global_frame_origin, global_frame_orientation, simulation_start_epoch, verification, wind)

# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)
Utils.add_HORUS_to_body_system(bodies, main)

# Retrieve dependent variables to save
dependent_variables_to_save = Utils.get_dependent_variable_save_settings( bodies, main, guidance_type, wind, 0, 0, 0, 0, 0, 0 )

# Define the termination settings 
termination_settings = Utils.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     termination_altitude,
                                                     termination_angle,
                                                     bodies,
                                                     main,
                                                     guidance_type,
                                                     wind)
                                                
# Setup interpolator

interpolator_settings = interpolators.lagrange_interpolation(
        8,boundary_interpolation = interpolators.extrapolate_at_boundary)

bodies_to_propagate = ["HORUS"]

# Define central bodies of propagation
central_bodies = ["Earth"]

spherical_initial_state = np.array((bodies.get_body("Earth").shape_model.average_radius + 122e3, np.deg2rad(-22.3), np.deg2rad(-106.7), 7435.5, np.deg2rad(-1.43), np.deg2rad(70.75)))

# We model the Earth here as an oblate spheroid, nonetheless in the .gnu files the Earth is modelled as Spherical
# therefore a good verification requires changes under that aspect

propagator_settings = Utils.get_propagator_settings(bodies,
                            simulation_start_epoch,
                            termination_settings,
                            dependent_variables_to_save,
                            spherical_initial_state,
                            guidance_type,
                            main,
                            wind,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            trim = True,
                            elevons = True)

propagator_settings.integrator_settings = Utils.get_integrator_settings(0.5, simulation_start_epoch, verification)

dynamics_simulator = numerical_simulation.create_dynamics_simulator(
bodies, propagator_settings)

state_history = dynamics_simulator.state_history
dependent_variables = dynamics_simulator.dependent_variable_history

directory = os.path.join(project_root, 'Verification files')

file_name_trajectory = 'horus2bus76'
file_name_forces = 'forces'

if Verify_Trajectory : 
        # Save the dictionary for the trajectory
        Trajectory_dictionary = Utils.read_gnu_files(directory, file_name_trajectory)

        state_history_array = result2array(state_history)
        dependent_variables_array = result2array(dependent_variables)

        plt.figure(figsize=(8, 6))
        plt.plot(dependent_variables_array[:, 0], np.rad2deg(dependent_variables_array[:, 5]), 'r-', linewidth=2.5, label='Simulated $\\tau$', alpha=0.8)
        plt.plot(Trajectory_dictionary['time'], Trajectory_dictionary['longitude'], 'b--', linewidth=2.5, label='Verification files', alpha=0.8)
        plt.xlabel('Time [s]', fontsize=16)
        plt.ylabel('$\\tau$ [deg]', fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.legend(loc='upper left', fontsize=14)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(dependent_variables_array[:, 0], np.rad2deg(dependent_variables_array[:, 4]), 'r-', linewidth=2.5, label='Simulated $\delta$', alpha=0.8)
        plt.plot(Trajectory_dictionary['time'], Trajectory_dictionary['latitude'], 'b--', linewidth=2.5, label='Verification files', alpha=0.8)
        plt.xlabel('Time [s]', fontsize=16)
        plt.ylabel('$\delta$ [deg]', fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)
        plt.legend(loc='upper left', fontsize=14)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(dependent_variables_array[:, 0], np.rad2deg(dependent_variables_array[:, 9]), 'r-', linewidth=2.5, label='Simulated $\sigma$', alpha=0.8)
        plt.plot(Trajectory_dictionary['time'], Trajectory_dictionary['sigma_ground'], 'b--', linewidth=2.5, label='Verification files', alpha=0.8)
        plt.xlabel('Time [s]', fontsize=16)
        plt.ylabel('$\sigma$ [deg]', fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)
        plt.legend(loc='lower left', fontsize=14)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(dependent_variables_array[:, 22], dependent_variables_array[:, 2]/1e3, 'r-', linewidth=2.5, label='Simulated hV graph', alpha=0.8)
        plt.plot(Trajectory_dictionary['velocity'], Trajectory_dictionary['height'], 'b--', linewidth=2.5, label='Verification files', alpha=0.8)
        plt.xlabel('Velocity [m/s]', fontsize=16)
        plt.ylabel('Altitude [km]', fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.legend(loc='upper left', fontsize=14)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(dependent_variables_array[:, 0], dependent_variables_array[:, 2]/1e3, 'r-', linewidth=2.5, label='Simulated h', alpha=0.8)
        plt.plot(Trajectory_dictionary['time'], Trajectory_dictionary['height'], 'b--', linewidth=2.5, label='Verification files', alpha=0.8)
        plt.xlabel('Time [s]', fontsize=16)
        plt.ylabel('Altitude [km]', fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.legend(loc='upper right', fontsize=14)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(dependent_variables_array[:, 0], dependent_variables_array[:, 22], 'r-', linewidth=2.5, label='Simulated V', alpha=0.8)
        plt.plot(Trajectory_dictionary['time'], Trajectory_dictionary['velocity'], 'b--', linewidth=2.5, label='Verification files', alpha=0.8)
        plt.ylabel('Velocity [m/s]', fontsize=16)
        plt.xlabel('Time [s]', fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.legend(loc='lower left', fontsize=14)
        plt.tight_layout()
        plt.show()

        print('Last Mach')
        print(dependent_variables_array[-1, 1])
        print('Last altitude')
        print(dependent_variables_array[-1, 2])

        plotting = HORUS2B_Plotting( bodies, state_history_array, dependent_variables_array)

        plotting.plot_reentry_corridor()
        plotting.plot_ground_track()

        # Let's find the energetic value of the spaceplane  

        value_to_find1 = 924
        value_to_find2 = 1319
        
        ind1 = [index for index, value in enumerate(dependent_variables_array[:,0]) if value == value_to_find1]
        ind2 = [index for index, value in enumerate(dependent_variables_array[:,0]) if value == value_to_find2]

        print('Start maneuver at')
        print(dependent_variables_array[ind1, 37])
        print('End maneuver at')
        print(dependent_variables_array[ind2, 37])

        print('Last energetic value')
        print(dependent_variables_array[-1, 37])

        plt.figure(figsize=(10, 6))
        plt.plot(dependent_variables_array[:, 0], np.rad2deg(dependent_variables_array[:, 36]), 'r-', linewidth=2.5, label='Simulated heading error graph', alpha=0.8)
        plt.plot(Trajectory_dictionary['time'], Trajectory_dictionary['heading error'], 'b--', linewidth=2.5, label='Verification files', alpha=0.8)
        plt.title('Comparison of Simulated and Verification error graphs', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Heading', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.show()

        heat_load = Utils.compute_Heat_Load(np.array(dependent_variables_array[:,24]), np.array(dependent_variables_array[:,0]))
        print('Heat Load')
        print(heat_load/1e3)

        print('Max G-Load')
        print(max(dependent_variables_array[ : , 11 ]))

        print('Max H-Flux')
        print(max(dependent_variables_array[ : , 24 ]))
