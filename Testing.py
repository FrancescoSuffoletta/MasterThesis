"""
Author : Francesco Benigno Suffoletta

The script here developed presents the codes that are needed
to perform one simulation of the re-entry for the HORUS-2B.
It is indeed used for debugging and testing of new implementations
while developing the thesis.

"""
#Import here all the basic packages requested for this Python script
import math
import numpy as np 
from matplotlib import pyplot as plt 
import time
import os

#Import then the libraries from TUDAT
from tudatpy.kernel.interface import spice_interface 
from tudatpy.data import save2txt
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, environment, propagation_setup
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime
from tudatpy.kernel.math import interpolators
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

import Utils
from HelperPlot import HORUS2B_Plotting

run_main = True

#Start set-up of the basic simulations, including bodies, vehicle model, environment and acceleration
spice_interface.load_standard_kernels()

# Choose whether output of the propagation is written to files
write_results_to_file = False
# Get path of current directory
current_dir = os.path.dirname(__file__)

# Set simulation start epoch
simulation_start_epoch = 0.0  # s
# Set termination conditions (Maximum duration of the simulation equals one Julian Day)
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 25.0E3  # m
termination_angle = 0.75 # deg

guidance_type = "PINN"   # "Predictor-Corrector", "PINN", "PINN_baseline" or "Verification"
verification = False
main = True
wind = False

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

bodies_to_create, global_frame_origin, global_frame_orientation = Utils.create_bodies()

# Create body settings
body_settings = Utils.get_environmental_settings(bodies_to_create, global_frame_origin, global_frame_orientation, simulation_start_epoch, verification, wind)

# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create the vehicle with all its parameters and its aerodynamic database
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


output_path = current_dir + '\\SimulationOutput\\Nominal_Run\\' if write_results_to_file else None

# Create the directory if it does not exist
if output_path and not os.path.exists(output_path):
    os.makedirs(output_path)

if run_main:

    bodies_to_propagate = ["HORUS"]

    # Define central bodies of propagation
    central_bodies = ["Earth"]

    spherical_initial_state = np.array((bodies.get_body("Earth").shape_model.average_radius + 122e3, np.deg2rad(-22.3), np.deg2rad(-106.7), 7435.5, np.deg2rad(-1.43), np.deg2rad(70.75)))

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
    
    propagator_settings.integrator_settings = Utils.get_integrator_settings(0.5 , simulation_start_epoch, True)

    start_time = time.time()

    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
    )

    end_time = time.time()

    tot = end_time - start_time 
    print(tot)

    state_history = dynamics_simulator.state_history
    dependent_variables = dynamics_simulator.dependent_variable_history
    # Convert the dependent variables from a dictionary to a numpy array
    state_history_array = result2array(state_history)
    dependent_variables_array = result2array(dependent_variables)

    if write_results_to_file :
        # Save results as .dat files 
        Utils.write_propagation_results_to_file(
            dynamics_simulator,
            output_path,
            tot,
            max(dependent_variables_array[:,24]),
            max(dependent_variables_array[:,11]),
            dependent_variables_array[-1,1],
            dependent_variables_array[-1,2],
            np.rad2deg(dependent_variables_array[-1,33]))

    # Load your data (assuming it's already in dependent_variables_array)
    time = dependent_variables_array[:, 0]  # Time [s]
    Cm = dependent_variables_array[:, 19]   # Pitching moment coefficient [-]
    dynamic_pressure = dependent_variables_array[:, 14]  # Assuming dynamic pressure is column 10

    # Define the threshold for control surface activation (adjust as needed)
    q_activation_threshold = 100  # Example: 100 Pa

    # Find the first time index where dynamic pressure exceeds the threshold
    activation_index = np.argmax(dynamic_pressure > q_activation_threshold)  # First occurrence
    activation_time = time[activation_index]  # Time of activation

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))  # Set figure size

    # Add background color for the untrimmed phase (before activation)
    ax.axvspan(time[0], activation_time, facecolor='lightcoral', alpha=0.3, label="$\delta_{b0}$")

    # Add background color for the trimmed phase (after activation)
    ax.axvspan(activation_time, time[-1], facecolor='lightgreen', alpha=0.3, label="Trimmed Phase")

    # Plot the Cm curve
    ax.plot(time, Cm, 'b', label="$C_m$")

    # Add vertical dashed line at the activation point
    ax.axvline(activation_time, color='black', linestyle='--', label=f"Control Surface Activation\n(q > {q_activation_threshold} Pa)")

    # Labels and title
    ax.set_xlabel("Time since the start of re-entry [s]", fontsize=14)
    ax.set_ylabel("$C_m$ [-]", fontsize=14)

    # Grid and legend
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower right", fontsize=12)

    # Show the plot
    plt.show()

    plt.plot(dependent_variables_array[:,0] , np.rad2deg(dependent_variables_array[:, 8]))
    plt.xlabel('Time since the start of the re-entry [s]')
    plt.ylabel('Angle of Attack [deg]')
    plt.grid()
    plt.show()

    plt.plot(dependent_variables_array[:,0] , np.rad2deg(dependent_variables_array[:, 33]))
    plt.xlabel('Time since the start of the re-entry [s]')
    plt.ylabel('Angular Distance [deg]')
    plt.grid()
    plt.show()

    plt.plot(dependent_variables_array[:,0] , np.rad2deg(dependent_variables_array[:, 4]))
    plt.xlabel('Time since the start of the re-entry [s]')
    plt.ylabel('Latitude [deg]')
    plt.grid()
    plt.show()

    plt.plot(dependent_variables_array[:,0] , np.rad2deg(dependent_variables_array[:, 5]))
    plt.xlabel('Time since the start of the re-entry [s]')
    plt.ylabel('Longitude [deg]')
    plt.grid()
    plt.show()

    plt.plot(dependent_variables_array[:,0] , dependent_variables_array[:, 16])
    plt.xlabel('Time since the start of the re-entry [s]')
    plt.ylabel('CD [-]')
    plt.grid()
    plt.show()

    plt.plot(dependent_variables_array[:,0] , np.rad2deg(dependent_variables_array[:, 7]))
    plt.xlabel('Time since the start of the re-entry [s]')
    plt.ylabel('Flight Path Angle [deg]')
    plt.grid()
    plt.show()

    print('Min Angular Distance reached')
    print(np.rad2deg(min(dependent_variables_array[:,33])))

    bank_rate = np.zeros(len(dependent_variables_array[:,0]))

    aoa_rate = np.zeros(len(dependent_variables_array[:,0]))
    

    for j in range(len(dependent_variables_array[:,0])-1) :

        bank_rate[j+1] = ( dependent_variables_array[j+1, 9] - dependent_variables_array[j , 9] ) / ( dependent_variables_array[j+1 , 0]- dependent_variables_array[j, 0])
        aoa_rate[j+1] = ( dependent_variables_array[j+1, 8] - dependent_variables_array[j , 8] ) / ( dependent_variables_array[j+1 , 0]- dependent_variables_array[j, 0])


    gload_rate = np.zeros(len(dependent_variables_array[:,0]))
    for j in range(len(dependent_variables_array[:,0])-1) :

        gload_rate[j+1] = ( dependent_variables_array[j+1, 11] - dependent_variables_array[j , 11] ) / ( dependent_variables_array[j+1 , 0]- dependent_variables_array[j, 0])

    print('MAX G-LOAD RATE')
    print(max(gload_rate))

    plt.figure(figsize=(10, 6))
    plt.plot(dependent_variables_array[:, 0], np.rad2deg(bank_rate), 'b', linewidth=2.5, label='HORUS $\dot{\sigma}$', alpha=0.8)
    plt.plot(dependent_variables_array[:, 0], np.full(len(dependent_variables_array[:,0]), 15), 'r--', linewidth=2.5, label='Upper limit for $\dot{\sigma}$', alpha=0.8)
    plt.plot(dependent_variables_array[:, 0], np.full(len(dependent_variables_array[:,0]), -15), 'k--', linewidth=2.5, label='Lower limit for $\dot{\sigma}$', alpha=0.8)
    plt.xlabel('Time [s]', fontsize=13)
    plt.ylabel('$\dot{\sigma}$ [deg/s]', fontsize=13)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(loc='best', fontsize=13)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(dependent_variables_array[:, 0], np.rad2deg(aoa_rate), 'b', linewidth=2.5, label='Simulated HORUS $\\dot{\\alpha}$', alpha=0.8)
    plt.plot(dependent_variables_array[:, 0], np.full(len(dependent_variables_array[:,0]), 5), 'r--', linewidth=2.5, label='Upper limit for $\\dot{\\alpha}$', alpha=0.8)
    plt.plot(dependent_variables_array[:, 0], np.full(len(dependent_variables_array[:,0]), -5), 'k--', linewidth=2.5, label='Lower limit for $\\dot{\\alpha}$', alpha=0.8)
    plt.xlabel('Time [s]', fontsize=13)
    plt.ylabel('$\\dot{\\alpha}$ [deg/s]', fontsize=13)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0.1), fontsize=14)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(dependent_variables_array[:, 0], np.rad2deg(dependent_variables_array[:, 9]), 'r', linewidth=2.5, label='Simulated $\sigma$', alpha=0.8)
    plt.plot(dependent_variables_array[:, 0], np.rad2deg(dependent_variables_array[:, 8]), 'b--', linewidth=2.5, label='Simulated $\\alpha$', alpha=0.8)
    plt.xlabel('Time since the start of the propagation [s]', fontsize=14)
    plt.ylabel('Control Angles [deg]', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(loc='lower left', fontsize=14)
    plt.tight_layout()
    plt.show()

    print('Last epoch')
    print(dependent_variables_array[-1,0])

    print('Terminal Altitude')
    print(dependent_variables_array[-1,2])

    print('Last angular distance')
    print(np.rad2deg(dependent_variables_array[-1,33]))

    print('MAX Heat Flux')
    print(max(dependent_variables_array[:,24]))
    print('MAX G-Load')
    print(max(dependent_variables_array[:,11]))

    Q_Load = Utils.compute_Heat_Load( dependent_variables_array[:, 24],  dependent_variables_array[:, 0] )
    print('Integrated Q Load')
    print(Q_Load)

    # Let's find the index for the energy value when we start moving from 0 bank angle

    print('Last Mach number is : ')
    print(dependent_variables_array[-1,1])

    plotting = HORUS2B_Plotting( bodies, state_history_array, dependent_variables_array)

    plotting.plot_hV_trajectory()

    plotting.plot_reentry_corridor()

    plotting.plot_ground_track()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main plot
    ax.plot(dependent_variables_array[:, 0], np.rad2deg(bank_rate), 'b', linewidth=2.5, 
            label='Simulated HORUS $\dot{\sigma}$', alpha=0.8)
    ax.plot(dependent_variables_array[:, 0], np.full(len(dependent_variables_array[:,0]), 15), 'r--', linewidth=2.5, 
            label='Upper limit for $\dot{\sigma}$', alpha=0.8)
    ax.plot(dependent_variables_array[:, 0], np.full(len(dependent_variables_array[:,0]), -15), 'k--', linewidth=2.5, 
            label='Lower limit for $\dot{\sigma}$', alpha=0.8)

    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('$\dot{\sigma}$ [deg/s]', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0.1), fontsize=14)

    # Creating inset axes
    axins = zoomed_inset_axes(ax, zoom=3, loc='upper center',
                          bbox_to_anchor=(0.5, 0.9), 
                          bbox_transform=ax.transAxes)
    # zoom=3 (adjust zoom as needed)

    # Plotting the same data in inset
    axins.plot(dependent_variables_array[:, 0], np.rad2deg(bank_rate), 'b', linewidth=2.5, alpha=0.8)
    axins.plot(dependent_variables_array[:, 0], np.full(len(dependent_variables_array[:,0]), 15), 'r--', linewidth=2.5, alpha=0.8)
    axins.plot(dependent_variables_array[:, 0], np.full(len(dependent_variables_array[:,0]), -15), 'k--', linewidth=2.5, alpha=0.8)

    # Define the zoom area (adjust this region)
    x1, x2 = 970, 1010   # x-limits of zoomed region
    y1, y2 = -1, 2  # y-limits of zoomed region
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Mark the inset region on the main plot
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()
    plt.show()