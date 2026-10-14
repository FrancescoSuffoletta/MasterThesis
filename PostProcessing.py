"""
Author : Francesco Benigno Suffoletta

This python script is used to analyse and 
plot the results from all the simulations that 
have been computed through the development 
of the thesis.

"""

# Import basic packages 
import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
import shutil
import json
import random
import torch
import joblib
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from poliastro.earth.atmosphere.coesa76 import COESA76
from astropy import units as u
from matplotlib.patches import Ellipse
import matplotlib.ticker as ticker

# Import the class for plots
from HelperPlot import HORUS2B_Plotting
from Utils import compute_Heat_Load

plot_lat_lon_results = False
plot_corridor = False
plot_bank_profile = False
select_single_run = False
plot_TAEM_conditions = False
only_failed_simulations = False
plot_Q_Load = False
plot_time_histograms = False
move_results = False
delete_fake_fail = False
train_postprocess = False
model_accuracy = True
radar_chart = False
violin_plot = False
analysis = False
wind_analysis = False

# Possible models : Predictor-Corrector, PINN, PINN_baseline
model = "Predictor-Corrector"
wind = False


# Get the current directory 
current_dir = os.path.dirname(__file__)

# Function to read the last row and extract specified columns
def read_last_row_columns(file_path, column_indices, columns_to_convert):

    """
    The function reads the last rows of the .dat documents from the simulation
    outputs so that it is possible to post process those results.

    Parameters
    ----------
    file_path : str
        The file path where the files are present
    column indices : np.ndarray
        Indices of columns to be read
    columns_to_convert : np.ndarray
        Indices of columns that need to be converted from radians to degrees
    
    Returns
    -------
    data_of_interest : list
        List of data needed for the post processing

    """

    data_of_interest = []
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()  # Read all lines into a list
            if lines:
                last_line = lines[-1].strip()  # Get the last line and strip any surrounding whitespace
                # Split the last line into columns
                columns = last_line.split()
                # Check if the file contains enough columns to match column_indices
                for index in column_indices:
                    if index < len(columns):
                        value = float(columns[index])  # Convert to float if valid column exists
                        # Convert to degrees if this column is in the list of columns to convert
                        if index in columns_to_convert:
                            value = np.rad2deg(value)
                        data_of_interest.append(value)
                    else:
                        print(f"Column index {index} is out of range for last line.")  # Debugging statement
                        data_of_interest.append(None)  # Fill with None if the column index doesn't exist
            else:
                print("The file is empty.")
                return []
            
    except Exception as e:
        print(f"An error occurred: {e}")

    return data_of_interest

# Function to read the maximum value of specified columns across all rows
def read_columns_max_value(file_path, column_indices):

    """
    The function reads the entire file and finds the maximum values for specified columns.

    Parameters
    ----------
    file_path : str
        The file path where the files are present.
    column_indices : list
        Indices of columns to find the maximum values for.

    Returns
    -------
    max_values : list
        List of maximum values for the specified columns.
    """
    
    max_values = [None] * len(column_indices)  # Initialize list to store max values for each column
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.split()  # Split each line into columns
                
                for idx, index in enumerate(column_indices):
                    if index < len(columns):
                        value = float(columns[index])  # Convert to float
                        
                        # Update the maximum value for this column index
                        if max_values[idx] is None or value > max_values[idx]:
                            max_values[idx] = value
                    else:
                        print(f"Column index {index} is out of range for line.")  # Debugging statement

    except Exception as e:
        print(f"An error occurred: {e}")

    return max_values

def read_columns_all_rows(file_path, column_indices):

    """
    Function to read specific columns from all rows in the .dat file.

    Parameters
    ----------
    file_path : str
        Path to the Dependent_Variables.dat file.
    column_indices : list
        Indices of the columns to be extracted (e.g., altitude and airspeed).

    Returns
    -------
    data : dict
        A dictionary where keys are column indices and values are lists of values from each row for that column.
    """

    data = {index: [] for index in column_indices}  # Dictionary to store the data for each column
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.split()  # Split each line into columns
                
                for index in column_indices:
                    if index < len(columns):
                        value = float(columns[index])  # Convert the value to float
                        data[index].append(value)  # Append to the corresponding column list
                    else:
                        data[index].append(None)  # Handle missing values gracefully
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    
    return data

def get_constraints_corridor( density, gravity, drag_coefficient, lift_coefficient ) :

    """
    The function is used to get the constraints for the re-entry corridor

    Parameters
    ----------
    density : np.ndarray
        Density during propagation
    gravity : np.ndarray
        Gravity values during propagation
    drag_coefficient : np.ndarray
        Drag coefficient during propagation
    lift_coefficient : np.ndarray
        Lift coefficient during propagation

    Returns 
    -------
    V_eq : np.ndarray
        Velocity for the equilibrium gliding flight 
    V_g : np.ndarray 
        Velocity for the g-load constraint
    V_heat : np.ndarray
        Velocity for the heat flux constraint

    """

    V_eq = np.zeros((len(density)))
    V_g = np.zeros((len(density)))
    V_heat = np.zeros((len(density)))

    Vc = 7.9054E3
    HORUS_surface = 110.0
    HORUS_mass = 26029.0
    HORUS_Rn = 0.8

    heat_flux_constraint = 530000 # W/m^2
    g_load_constraint = 3 

    n = 0.5
    m = 3.15
    c_star = 1.1097E8
    rho_0 = 1.225

    for i in range(len(density)) : 
        V_eq[i] = math.sqrt( Vc**2 / ( Vc**2 * ( lift_coefficient[i] * HORUS_surface / (gravity[i]*HORUS_mass) ) * 0.5 * density[i] + 1 ))
        V_g[i] = math.sqrt((2 * g_load_constraint * HORUS_mass * gravity[i])/( HORUS_surface * density[i] * math.sqrt(drag_coefficient[i]**2 + lift_coefficient[i]**2)))
        V_heat[i] =  Vc * ( HORUS_Rn**n * heat_flux_constraint * math.sqrt(rho_0/density[i])/c_star)**(1/m)

    return V_eq, V_g, V_heat

def plot_reentry_corridor(results, column_indices, model, categories, max_trajectories= 1):

    """
    Plot the re-entry corridor (Altitude vs Airspeed) for each category (Success, Fail).
    Limits the number of trajectories per category to a maximum of 20.

    Parameters
    ----------
    results : dict
        Dictionary containing the data for each category.
    column_indices : list
        Indices of the columns being plotted (altitude and airspeed)
    categories : list
        List of categories to be plotted (default: Success,  Fail)
    max_trajectories : int
        The maximum number of trajectories to be plotted per category (default: 20)

    Returns 
    -------
    None

    """
    # Set up the figure and axis for the plot
    plt.figure(figsize=(10, 6))

   
    colors = {
        'Success': 'green',
        'Acceptable': 'blue',
        'Fail': 'red'
    }

    j = 0

    # Loop through each category and plot the Altitude vs Airspeed
    for category in categories:
        if category in results:
            # Get the runs for the category
            runs = list(results[category].keys())

            # Randomly sample 20 runs (or fewer if less than 20 are available)
            sampled_runs = random.sample(runs, min(max_trajectories, len(runs)))

            # Plot each sampled run
            for run in sampled_runs:
                data = results[category][run]
                altitude = data[column_indices[0]]  # Altitude values
                airspeed = data[column_indices[1]]  # Airspeed values
                plt.plot(airspeed, altitude, color=colors[category], label=category if run == sampled_runs[0] else "", linewidth = 1.5)

                if j == 0:
                    V_eq, V_gload, V_heat = get_constraints_corridor( data[column_indices[2]], data[column_indices[3]], data[column_indices[4]], data[column_indices[5]] )
                    plt.plot(V_eq, data[column_indices[0]], label='Quasi-equilibrium Glide Condition', linestyle='--', color='blue', linewidth =2.5, alpha = 0.8)
                    plt.plot(V_gload, data[column_indices[0]], label='Maximum g-load', linestyle='--', color='red', linewidth =2.5, alpha = 0.8)
                    plt.plot(V_heat, data[column_indices[0]], label='Maximum Heat Flux', linestyle='--', color='green', linewidth =2.5, alpha = 0.8)

                j += 1
    if wind: 
        # Labels and title
        plt.xlabel('Velocity [m/s]', fontsize = 14)
        plt.ylabel('Altitude [m]', fontsize = 14)
        plt.xlim(0, 8000)
        plt.ylim(0, 130000)
        plt.grid(True)

        # Add legend
        plt.legend(loc='lower right')
        
        # Show the plot
        plt.show()
    else :
        # Labels and title
        plt.xlabel('Velocity [m/s]', fontsize = 14)
        plt.ylabel('Altitude [m]', fontsize = 14)
        plt.xlim(0, 8000)
        plt.ylim(0, 130000)
        plt.grid(True)

        # Add legend
        plt.legend(loc='lower right')
        
        # Show the plot
        plt.show()

def plot_control_angles(results, column_indices, categories, max_trajectories=30):

    """
    Plot the re-entry corridor (Altitude vs Airspeed) for each category (Success,  Fail).
    Limits the number of trajectories per category to a maximum of 20.

    Parameters
    ----------
    results : dict
        Dictionary containing the data for each category.
    column_indices : list
        Indices of the columns being plotted (altitude and airspeed)
    categories : list
        List of categories to be plotted (default: Success, Fail)
    max_trajectories : int
        The maximum number of trajectories to be plotted per category (default: 20)

    Returns 
    -------

    """
    # Set up the figure and axis for the plot
    plt.figure(figsize=(10, 6))

    if model == "Predictor-Corrector " : 
        # Plot colors for each category
        colors = {
            'Success': 'green',
            'Fail': 'red'
        }
    else : 
        colors = {
            'Success': 'green',
            'Acceptable': 'blue',
            'Fail': 'red'
        }

    # Loop through each category and plot the Altitude vs Airspeed
    for category in categories:
        if category in results:
            # Get the runs for the category
            runs = list(results[category].keys())

            # Randomly sample 20 runs (or fewer if less than 20 are available)
            sampled_runs = random.sample(runs, min(max_trajectories, len(runs)))

            # Plot each sampled run
            for run in sampled_runs:
                data = results[category][run]
                aoa = data[column_indices[0]]  # Angle of attack profile
                bank = data[column_indices[1]]  # Bank angle profile
                time = data[column_indices[2]]

                plt.plot(time, np.rad2deg(aoa), linestyle='--', label=category + ' AoA' if run == sampled_runs[0] else "", linewidth = 0.8)
                plt.plot(time, np.rad2deg(bank), color=colors[category], label=category + ' Bank Angle' if run == sampled_runs[0] else "", linewidth = 1)
    if wind:
        # Labels and title
        plt.xlabel('Time since J2000 [s]', fontsize = 14)
        plt.ylabel('Control angles [m]', fontsize = 14)
        plt.title(f'Control Angles profiles using {model} with Wind', fontsize=16, fontweight='bold')
        plt.grid(True)

        # Add legend
        plt.legend(loc='lower left')
        
        # Show the plot
        plt.show()
    else : 
        # Labels and title
        plt.xlabel('Time since J2000 [s]', fontsize = 14)
        plt.ylabel('Control angles [m]', fontsize = 14)
        plt.title(f'Control Angles profiles using {model} ', fontsize=16, fontweight='bold')
        plt.grid(True)

        # Add legend
        plt.legend(loc='lower left')
        
        # Show the plot
        plt.show()

if plot_lat_lon_results :
    # Define the base directory where to look for results
    if wind : 
        first_dir = os.path.join( current_dir, "SimulationOutput", "Wind" )
    else : 
        first_dir = os.path.join( current_dir, "SimulationOutput", "NoWind" )
    # Dictionary to store the data for Success, Fail
    if model == "Predictor-Corrector" :
        base_dir = os.path.join( first_dir, "PredCorr" )
    else : 
        base_dir = os.path.join( first_dir, model )

    valid_categories = ['Success', 'Acceptable', 'Fail']
    results = {
        'Success': {},
        'Acceptable' : {},
        'Fail': {}
    }

    # Columns of interest 
    column_indices = [ 4, 5 ] 
    # Columns to convert from radians to degrees
    columns_to_convert = [ 4, 5 ]
    # Loop between all the valid categories to collect results 
    for category in valid_categories:
        category_dir = os.path.join(base_dir, category)
        # Check if the category directory exists
        if os.path.exists(category_dir):
            # Walk through the directory structure in this category
            for root, dirs, files in os.walk(category_dir):
                if 'Dependent_Variables.dat' in files:
                    # Get the run directory name (e.g., run_14000, run_34829)
                    run_dir = os.path.basename(root)
                    # Full path to the Dependent_Variables.dat file
                    dep_var_file = os.path.join(root, 'Dependent_Variables.dat')
                    # Read the last row's data for the specified columns
                    column_data = read_last_row_columns(dep_var_file, column_indices, columns_to_convert)
                    # Store the data in the results dictionary under the respective category
                    data_dict = {f'column_{i+1}': value for i, value in enumerate(column_data)}
                    results[category][run_dir] = data_dict

    """
    for category, runs in results.items():
        print(f'Category: {category}')
        for run, data in runs.items():
            print(f'  Run: {run}, Data: {data}')
    """

    # Once the results have been collected, plot the final latitudes and longitudes
    target_latitude = 5.0
    target_longitude = -53.07

    colors = {
        'Success': 'green',
        'Acceptable' : 'yellow',
        'Fail': 'red'
    }

    # Set up the figure and axis for the plot
    plt.figure(figsize=(8, 6))

    # Plot the points for Success and Fail
    for category in valid_categories:
        if category in results:
            for run, data in results[category].items():
                if 'column_1' in data and 'column_2' in data:
                    try:
                        lat = data['column_1']  # Assuming column_1 is latitude
                        lon = data['column_2']  # Assuming column_2 is longitude
                        plt.scatter(lon, lat, color=colors[category], label=category if run == list(results[category].keys())[0] else "")
                    except KeyError as e:
                        print(f"Error: Missing key {e} for run: {run} in category: {category}")
                else:
                    # Print a warning if the keys are missing
                    print(f"Warning: 'column_1' or 'column_2' missing for run: {run} in category: {category}")

    # Plot the target point (desired landing point)
    plt.scatter(target_longitude, target_latitude, color='k', label='Target Point')

    # Create circles around the target landing point
    circle_0 = plt.Circle((target_longitude, target_latitude), 0.75, color='c', fill=False, linestyle='--', label='Landing Area Termination Condition')
    circle_1 = plt.Circle((target_longitude, target_latitude), 1, color='m', fill=False, linestyle='--', label='Successful Terminal Landing Area')

    # Get the current axis and add the circles to it
    ax = plt.gca()
    ax.add_artist(circle_0)
    ax.add_artist(circle_1)

    if wind:
        # Set limits and labels
        plt.xlim(target_longitude - 3.51, target_longitude + 0.02)
        plt.ylim(target_latitude - 3, target_latitude + 0.02)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('Longitude [deg]', fontsize = 14)
        plt.ylabel('Latitude [deg]', fontsize = 14)

        # Adding legend and grid
        plt.legend(loc='upper left', fontsize = 12)
        plt.grid()

        # Show the plot
        plt.show()
    else :
        # Set limits and labels
        plt.xlim(target_longitude - 1.51, target_longitude + 0.02)
        plt.ylim(target_latitude - 1, target_latitude + 0.02)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('Longitude [deg]', fontsize = 14)
        plt.ylabel('Latitude [deg]', fontsize = 14)

        # Adding legend and grid
        plt.legend(loc='lower right', fontsize = 10)
        plt.grid()

        # Show the plot
        plt.show()

if plot_corridor : 

    # Define the base directory where to look for results
    if wind : 
        first_dir = os.path.join(current_dir, "SimulationOutput", "Wind")
    else :
        first_dir = os.path.join(current_dir, "SimulationOutput", "NoWind")

    # Keep Defining the directory later
    if model == "Predictor-Corrector" :
        base_dir = os.path.join( first_dir, 'PredCorr' )
    else : 
        base_dir = os.path.join( first_dir, model )

    # Specify the directories within SimulationOutput to be analyzed
    valid_categories = ['Fail']
    results = {
            'Success': {},
            'Acceptable': {},
            'Fail': {}
        }
    # Columns of interest for altitude, airspeed, density, gravity, drag coefficient , lift coefficient
    column_indices = [2, 22, 12, 23, 16, 18]  

    # Loop through the categories to collect results
    for category in valid_categories:
        # Loop between the three categories where results are stored
        category_dir = os.path.join(base_dir, category)
        # Check if the category directory exists
        if os.path.exists(category_dir):
            # Walk through the directory structure in this category
            for root, dirs, files in os.walk(category_dir):
                if 'Dependent_Variables.dat' in files:
                    # Get the run directory name (e.g., run_14000, run_34829)
                    run_dir = os.path.basename(root)

                    # Full path to the Dependent_Variables.dat file
                    dep_var_file = os.path.join(root, 'Dependent_Variables.dat')

                    # Read the specific columns (altitude and airspeed) from all rows
                    column_data = read_columns_all_rows(dep_var_file, column_indices)

                    # Store the data in the results dictionary under the respective category
                    results[category][run_dir] = column_data

    # Once the results have been collected, plot the re-entry corridor
    plot_reentry_corridor(results, column_indices, model, valid_categories)

if plot_bank_profile : 

    if wind == True :
        first_dir = os.path.join(current_dir, "SimulationOutput", "Wind")
    else :
        first_dir = os.path.join(current_dir, "SimulationOutput", "NoWind")

    if model == "Predictor-Corrector" : 
        # Dictionary to store the data for Success, Fail
        results = {
            'Success': {},
            'Fail': {}
        }
        # Specify the directories within SimulationOutput to be analyzed
        valid_categories = ['Success', 'Fail']
        base_dir = os.path.join(first_dir, 'PredCorr')
    else : 
        # Dictionary to store the data for Success, Fail
        results = {
            'Success': {},
            'Acceptable': {},
            'Fail': {}
        }
        # Specify the directories within SimulationOutput to be analyzed
        valid_categories = ['Success', 'Acceptable', 'Fail']
        base_dir = os.path.join(first_dir, model)

    # Columns of interest for AoA, bank angle, time
    column_indices = [ 8, 9, 0]  
    # Loop through the categories to collect results
    for category in valid_categories:
        # Loop between the three categories where results are stored
        category_dir = os.path.join(base_dir, category)
        # Check if the category directory exists
        if os.path.exists(category_dir):
            # Walk through the directory structure in this category
            for root, dirs, files in os.walk(category_dir):
                if 'Dependent_Variables.dat' in files:
                    # Get the run directory name (e.g., run_14000, run_34829)
                    run_dir = os.path.basename(root)
                    # Full path to the Dependent_Variables.dat file
                    dep_var_file = os.path.join(root, 'Dependent_Variables.dat')
                    # Read the specific columns (altitude and airspeed) from all rows
                    column_data = read_columns_all_rows(dep_var_file, column_indices)
                    # Store the data in the results dictionary under the respective category
                    results[category][run_dir] = column_data
    # Once the results have been collected, plot the re-entry corridor
    plot_control_angles(results, column_indices, valid_categories)

if select_single_run : 
    # Define the base directory where to look for results
    if wind : 
        first_dir = os.path.join(current_dir, "SimulationOutput", 'Wind')
    else : 
        first_dir = os.path.join(current_dir, "SimulationOutput", 'NoWind')
    if model == 'Predictor-Corrector' :
        base_dir = os.path.join(first_dir, 'PredCorr')
    else :
        base_dir = os.path.join(first_dir, model)
    # Define the category to be read (To be changed depending on the results to be analysed )
    category = "Success"
    category_dir = os.path.join( base_dir, category )
    # Initialize the dictorionary to store the result
    results = { category : {} 
    }
    # Define the run directory (to be changed depending on the run)
    selected_run = "run_2910"
    run_dir = os.path.join( category_dir, selected_run)
    # Get the file for the dependent_variables  
    dep_var_file = dep_var_file = os.path.join(run_dir, 'Dependent_Variables.dat')
    # Define the column indices for the results to be analysed 
    # Altitude (0), airspeed (1), density (2), gravity (3), drag coefficient (4), lift coefficient (5), AoA (6), bank angle (7), time (8) 
    column_indices = [2, 22, 12, 23, 16, 18, 8, 9, 0]
    # Read the defined pieces of data
    column_data = read_columns_all_rows(dep_var_file, column_indices)
    # Import the processed results
    results[category][selected_run] = column_data
    # Now define the plots that can be useful for improving the understanding of specific runs
    data = results[category][selected_run]

    #########################
    ## PLOT CONTROL ANGLES ##
    #########################

    aoa = data[column_indices[6]]  # Angle of attack profile
    bank = data[column_indices[7]]  # Bank angle profile
    time = data[column_indices[8]]
    plt.plot(time, np.rad2deg(aoa),  linestyle='--', label=category + ' AoA' , linewidth = 1.5)
    plt.plot(time, np.rad2deg(bank), label=category + ' Bank Angle' , linewidth = 1.5)
    # Labels and title
    plt.xlabel('Time since J2000 [s]', fontsize = 14)
    plt.ylabel('Control angles [m]', fontsize = 14)
    plt.title('Control Angles profiles', fontsize=16, fontweight='bold')
    plt.grid(True)
    # Add legend
    plt.legend(loc='lower left')
    # Show the plot
    plt.show()

    #########################
    ## PLOT ENTRY CORRIDOR ##
    #########################

    # Define the quantities to be considered
    altitude = data[column_indices[0]]
    airspeed = data[column_indices[1]]
    density = data[column_indices[2]]
    gravity = data[column_indices[3]]
    drag_coefficient = data[column_indices[4]]
    lift_coefficient = data[column_indices[5]]

    # Compute the constraints for the corridor
    V_eq = np.zeros((len(density)))
    V_g = np.zeros((len(density)))
    V_heat = np.zeros((len(density)))

    Vc = 7.9054E3
    HORUS_surface = 110.0
    HORUS_mass = 26029.0
    HORUS_Rn = 0.8

    heat_flux_constraint = 530000 # W/m^2
    g_load_constraint = 3 

    n = 0.5
    m = 3.15
    c_star = 1.1097E8
    rho_0 = 1.225

    for i in range(len(density)) : 
        V_eq[i] = math.sqrt( Vc**2 / ( Vc**2 * ( lift_coefficient[i] * HORUS_surface / (gravity[i]*HORUS_mass) ) * 0.5 * density[i] + 1 ))
        V_g[i] = math.sqrt((2 * g_load_constraint * HORUS_mass * gravity[i])/( HORUS_surface * density[i] * math.sqrt(drag_coefficient[i]**2 + lift_coefficient[i]**2)))
        V_heat[i] =  Vc * ( HORUS_Rn**n * heat_flux_constraint * math.sqrt(rho_0/density[i])/c_star)**(1/m)

    plt.plot(V_eq, altitude, label='Quasi-equilibrium Glide Condition', linestyle='--', color='blue', linewidth =2.5, alpha = 0.8)
    plt.plot(V_g, altitude, label='Maximum g-load', linestyle='--', color='red', linewidth =2.5, alpha = 0.8)
    plt.plot(V_heat, altitude, label='Maximum Heat Flux', linestyle='--', color='green', linewidth =2.5, alpha = 0.8)
    plt.plot(airspeed, altitude, color= "k", label= "HORUS-2B trajectory", linewidth = 1.5)
    # Labels and title
    plt.xlabel('Velocity [m/s]', fontsize = 14)
    plt.ylabel('Altitude [m]', fontsize = 14)
    plt.xlim(0, 8000)
    plt.ylim(0, 130000)
    plt.title('Re-Entry Corridor: h vs V trajectories (no Tracking)', fontsize=16, fontweight='bold')
    plt.grid(True)
    # Add legend
    plt.legend(loc='lower right')
    # Show the plot
    plt.show()

if plot_TAEM_conditions :

    # Define the base directory where to look for results
    if wind : 
        first_dir = os.path.join( current_dir, "SimulationOutput", "Wind")
    else :
        first_dir = os.path.join( current_dir, "SimulationOutput", "NoWind")

    if model == 'Predictor-Corrector': 
        base_dir = os.path.join(first_dir, 'PredCorr')
    else :
        base_dir = os.path.join(first_dir, model)

    # Dictionary to store the data for Success, Fail
    results1 = {
        'Success': {},
        'Acceptable': {},
        'Fail': {}
    }
    results2 = {
        'Success': {},
        'Acceptable': {},
        'Fail': {}
    }
    # Specify the directories within SimulationOutput to be analysed
    valid_categories = ['Success', 'Acceptable', 'Fail']

    # Columns of interest (Mach, Altitude and Angular Distance) for the last values
    column_indices1 = [ 1, 2, 33 ]  
    # Columns of interest (G-load and Heat-Flux) for the max values
    column_indices2 = [ 11, 24 ]
    # Columns to convert from radians to degrees
    columns_to_convert = [ 33 ]
    # Loop between all the valid categories to collect results 
    for category in valid_categories:
        category_dir = os.path.join(base_dir, category)
        # Check if the category directory exists
        if os.path.exists(category_dir):
            # Walk through the directory structure in this category
            for root, dirs, files in os.walk(category_dir):
                if 'Dependent_Variables.dat' in files:
                    # Get the run directory name (e.g., run_14000, run_34829)
                    run_dir = os.path.basename(root)
                    # Full path to the Dependent_Variables.dat file
                    dep_var_file = os.path.join(root, 'Dependent_Variables.dat')
                    # Read the last row's data for the specified columns
                    column_data1 = read_last_row_columns(dep_var_file, column_indices1, columns_to_convert)
                    column_data2 = read_columns_max_value(dep_var_file, column_indices2)
                    # Store the data in the results dictionary under the respective category
                    data_dict1 = {f'column_{i+1}': value for i, value in enumerate(column_data1)}
                    data_dict2 = {f'column_{i+1}': value for i, value in enumerate(column_data2)}
                    results1[category][run_dir] = data_dict1
                    results2[category][run_dir] = data_dict2

    # Results have been collected, define the threshold for the different quantities
    mach_threshold = 3 # [-]
    altitude_threshold = 30 # [m]
    angular_distance_threshold = 1 # [deg]

    mach_numbers_success = []
    mach_numbers_acc = []
    mach_numbers_fail = []
    altitudes_success = []
    altitudes_acc = []
    altitudes_fail =[]
    angular_distances_success = []
    angular_distances_acc = []
    angular_distances_fail = []
    max_g_load_success = []
    max_g_load_acc = []
    max_g_load_fail = []
    max_heat_flux_success = []
    max_heat_flux_acc = []
    max_heat_flux_fail = []

    for category in valid_categories:
        if category in results1:
            for run, data in results1[category].items():
                if 'column_1' in data and 'column_2' in data and 'column_3' in data :
                    try:
                        if category == 'Success' : 
                            mach_numbers_success.append(data['column_1'])  
                            altitudes_success.append(data['column_2'])  
                            angular_distances_success.append(data['column_3'])
                        elif category == "Fail" : 
                            mach_numbers_fail.append(data['column_1'])
                            altitudes_fail.append(data['column_2'])
                            angular_distances_fail.append(data['column_3'])
                        elif category == "Acceptable" : 
                            mach_numbers_acc.append(data['column_1'])
                            altitudes_acc.append(data['column_2'])
                            angular_distances_acc.append(data['column_3'])
                    except KeyError as e:
                        print(f"Error: Missing key {e} for run: {run} in category: {category}")
                else:
                    # Print a warning if the keys are missing
                    print(f"Warning: 'column_1',  'column_2' or 'column_3' missing for run: {run} in category: {category}")
        if category in results2 :
            for run, data in results2[category].items():
                if  'column_1' in data and 'column_2' in data : 
                    try : 
                        if category == 'Success' : 
                            max_g_load_success.append(data['column_1'])
                            max_heat_flux_success.append(data['column_2'])
                        elif category == "Acceptable" : 
                            max_g_load_acc.append(data['column_1'])
                            max_heat_flux_acc.append(data['column_2'])
                        else : 
                            max_g_load_fail.append(data['column_1'])
                            max_heat_flux_fail.append(data['column_2'])
                    except KeyError as e:
                        print(f"Error: Missing key {e} for run: {run} in category: {category}")
                else:
                    # Print a warning if the keys are missing
                    print(f"Warning: 'column_1' or 'column_2' missing for run: {run} in category: {category}") 

    # Prepare the lists with all the maximum g-loads and heat fluxes
    all_g_loads = []
    all_heat_fluxes = []

    for i in range(len(max_g_load_success)): 
        all_g_loads.append(max_g_load_success[i])
        all_heat_fluxes.append(max_heat_flux_success[i])
    
    for i in range(len(max_g_load_acc)): 
        all_g_loads.append(max_g_load_acc[i])
        all_heat_fluxes.append(max_heat_flux_acc[i])

    for k in range(len(max_g_load_fail)): 
        all_g_loads.append(max_g_load_fail[k])
        all_heat_fluxes.append(max_heat_flux_fail[k])

    print(all_g_loads)
    print(all_heat_fluxes)

    mach_numbers_success = np.array(mach_numbers_success)
    angular_distances_success = np.array(angular_distances_success)
    altitudes_success = np.array(altitudes_success)
    max_heat_flux_success = np.array(max_heat_flux_success)

    mach_numbers_acc = np.array(mach_numbers_acc)
    angular_distances_acc = np.array(angular_distances_acc)
    altitudes_acc = np.array(altitudes_acc)
    max_heat_flux_acc = np.array(max_heat_flux_acc)

    mach_numbers_fail = np.array(mach_numbers_fail)
    angular_distances_fail = np.array(angular_distances_fail)
    altitudes_fail = np.array(altitudes_fail)
    max_heat_flux_fail = np.array(max_heat_flux_fail)

    all_g_loads = np.array(all_g_loads)
    all_heat_fluxes = np.array(all_heat_fluxes)

    # Plot 1: Color based on maximum g-load
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    # Success points
    sc1_success = ax1.scatter(mach_numbers_success, angular_distances_success, altitudes_success/1e3, c=max_g_load_success,
                            cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads), marker='o', s=20, edgecolor=None, alpha = 0.8, label='Success')
        # Acceptable points
    sc1_acceptable = ax1.scatter(mach_numbers_acc, angular_distances_acc, altitudes_acc/1e3, c=max_g_load_acc,
                            cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads), marker='v', s=20, edgecolor=None, alpha = 0.8, label='Acceptable')
    # Fail points
    sc1_fail = ax1.scatter(mach_numbers_fail, angular_distances_fail, altitudes_fail/1e3, c=max_g_load_fail[:],
                        cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads), marker='X', s=20, edgecolor=None,  alpha = 0.8, label='Fail')
    # Setting labels and adding a single color bar
    ax1.set_xlabel("Mach Number [-]", fontsize = 14)
    ax1.set_zlabel("Altitude [km]", fontsize = 14)
    ax1.set_ylabel("Angular Distance [deg]", fontsize = 14)
    ax1.set_xlim(1.9, 3.5)
    ax1.set_zlim(22, 32)
    ax1.set_ylim(0.7, 1.2)
    plt.colorbar(sc1_success, ax=ax1, label="Max G-Load [-]", pad=0.1, shrink=0.9)  # One color bar for all points
    # Plot threshold planes for success criteria
    ax1.plot_surface(np.array([[mach_threshold, mach_threshold], [mach_threshold, mach_threshold]]),
                    np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                    np.array([[20, altitude_threshold], [20, altitude_threshold]]),
                    color='lightgrey', alpha=0.3, rstride=100, cstride=100)
    ax1.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                    np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                    np.array([[altitude_threshold, altitude_threshold], [altitude_threshold, altitude_threshold]]),
                    color='lightgrey', alpha=0.3, rstride=100, cstride=100)
    ax1.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                    np.array([[angular_distance_threshold, angular_distance_threshold], [angular_distance_threshold, angular_distance_threshold]]),
                    np.array([[20, 20], [altitude_threshold, altitude_threshold]]),
                    color='lightgrey', alpha=0.3, rstride=100, cstride=100)

    ax1.legend(fontsize = 12)
    plt.show()

    # Plot 2: Color based on maximum heat flux
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    # Success points
    sc2_success = ax2.scatter(mach_numbers_success,  angular_distances_success, altitudes_success/1e3, c=max_heat_flux_success,
                            cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes), marker='o', s=20, edgecolor=None, alpha = 0.8, label='Success')
    sc2_acc = ax2.scatter(mach_numbers_acc,  angular_distances_acc, altitudes_acc/1e3, c=max_heat_flux_acc,
                            cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes), marker='v', s=20, edgecolor=None, alpha = 0.8, label='Acceptable')
    # Fail points
    sc2_fail = ax2.scatter(mach_numbers_fail, angular_distances_fail, altitudes_fail/1e3, c=max_heat_flux_fail,
                        cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes), marker='X', s=20, edgecolor=None, alpha = 0.8, label='Fail')
    # Setting labels and adding a single color bar
    ax2.set_xlabel("Mach Number [-]", fontsize = 14)
    ax2.set_zlabel("Altitude [km]", fontsize = 14)
    ax2.set_ylabel("Angular Distance [deg]", fontsize = 14)
    ax2.set_xlim(1.9, 3.5)
    ax2.set_zlim(22, 32.5)
    ax2.set_ylim(0.7, 1.2)
    plt.colorbar(sc2_success, ax=ax2, label="Max Heat Flux [kW/m²]", pad=0.1, shrink=0.9)
    # Plot threshold planes for success criteria
    ax2.plot_surface(np.array([[mach_threshold, mach_threshold], [mach_threshold, mach_threshold]]),
                    np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                    np.array([[20, altitude_threshold], [20, altitude_threshold]]),
                    color='lightgrey', alpha=0.3, rstride=100, cstride=100)
    ax2.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                    np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                    np.array([[altitude_threshold, altitude_threshold], [altitude_threshold, altitude_threshold]]),
                    color='lightgrey', alpha=0.3, rstride=100, cstride=100)
    ax2.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                    np.array([[angular_distance_threshold, angular_distance_threshold], [angular_distance_threshold, angular_distance_threshold]]),
                    np.array([[20, 20], [altitude_threshold, altitude_threshold]]),
                    color='lightgrey', alpha=0.3, rstride=100, cstride=100)

    ax2.legend(fontsize = 12)
    plt.show()

    # Thresholds
    mach_threshold = 3
    altitude_threshold = 30000  # Altitude in meters
    angular_distance_threshold = 1

    # Plot
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')

    # Success projections on walls (single label)
    sc1_success = ax1.scatter(mach_numbers_success, [0.5] * len(mach_numbers_success), altitudes_success / 1000,
                            c=max_g_load_success, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                            marker='o', s=20, edgecolor=None, alpha=0.8, label='Success')

    ax1.scatter(mach_numbers_success, angular_distances_success, [20] * len(mach_numbers_success),
                c=max_g_load_success, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                marker='o', s=20, edgecolor=None, alpha=0.8)

    ax1.scatter([2] * len(mach_numbers_success), angular_distances_success, altitudes_success / 1000,
                c=max_g_load_success, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                marker='o', s=20, edgecolor=None, alpha=0.8)

    # Acceptable projections on walls (single label)
    sc1_acceptable = ax1.scatter(mach_numbers_acc, [0.5] * len(mach_numbers_acc), altitudes_acc / 1000,
                            c=max_g_load_acc, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                            marker='v', s=20, edgecolor=None, alpha=0.8, label='Acceptable')

    ax1.scatter(mach_numbers_acc, angular_distances_acc, [20] * len(mach_numbers_acc),
                c=max_g_load_acc, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                marker='o', s=20, edgecolor=None, alpha=0.8)

    ax1.scatter([2] * len(mach_numbers_acc), angular_distances_acc, altitudes_acc / 1000,
                c=max_g_load_acc, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                marker='o', s=20, edgecolor=None, alpha=0.8)

    # Fail projections on walls (single label)
    sc1_fail = ax1.scatter(mach_numbers_fail, [0.5] * len(mach_numbers_fail), altitudes_fail / 1000,
                        c=max_g_load_fail, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                        marker='X', s=20, edgecolor=None, alpha=0.8, label='Fail')

    ax1.scatter(mach_numbers_fail, angular_distances_fail, [20] * len(mach_numbers_fail),
                c=max_g_load_fail, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                marker='X', s=20, edgecolor=None, alpha=0.8)

    ax1.scatter([2] * len(mach_numbers_fail), angular_distances_fail, altitudes_fail / 1000,
                c=max_g_load_fail, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                marker='X', s=20, edgecolor=None, alpha=0.8)

    # Add a single colorbar for g-load
    cbar = fig1.colorbar(sc1_success, ax=ax1, label="Max G-Load [-]", pad=0.1, shrink=0.9)
    cbar.ax.tick_params(labelsize=12) 
    cbar.set_label("Max G-Load [-]", fontsize=12)

    # Add dotted lines for all combinations, including Altitude-Angular Distance
    ax1.plot([mach_threshold, mach_threshold], [0.5, angular_distance_threshold], [20, 20], linestyle='dotted', color='red', label='Terminal Condition Limits')
    ax1.plot([mach_threshold, mach_threshold], [0.5, 0.5], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
    ax1.plot([2, mach_threshold], [angular_distance_threshold, angular_distance_threshold], [20, 20], linestyle='dotted', color='red')
    ax1.plot([2, 2], [angular_distance_threshold, angular_distance_threshold], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
    ax1.plot([2, mach_threshold], [0.5, 0.5], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')
    ax1.plot([2, 2], [0.5, angular_distance_threshold], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')

    # Set labels and title
    ax1.set_xlabel("Mach Number [-]", fontsize = 14)
    ax1.set_zlabel("Altitude [km]", fontsize = 14)
    ax1.set_ylabel("Angular Distance [deg]", fontsize = 14)
    ax1.set_xlim(2, 3.5)
    ax1.set_zlim(20, 32)
    ax1.set_ylim(0.5, 1.3)
    ax1.legend(fontsize = 12)

    plt.show()

    # Plot
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Success projections on walls (single label)
    sc2_success = ax2.scatter(mach_numbers_success, [0.5] * len(mach_numbers_success), altitudes_success / 1000,
                            c=max_heat_flux_success, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                            marker='o', s=20, edgecolor=None, alpha=0.8, label='Success')

    ax2.scatter(mach_numbers_success, angular_distances_success, [20] * len(mach_numbers_success),
                c=max_heat_flux_success, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                marker='o', s=20, edgecolor=None, alpha=0.8)

    ax2.scatter([2] * len(mach_numbers_success), angular_distances_success, altitudes_success / 1000,
                c=max_heat_flux_success, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                marker='o', s=20, edgecolor=None, alpha=0.8)

    # Acceptable projections on walls (single label)
    sc2_acc = ax2.scatter(mach_numbers_acc, [0.5] * len(mach_numbers_acc), altitudes_acc / 1000,
                            c=max_heat_flux_acc, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                            marker='v', s=20, edgecolor=None, alpha=0.8, label='Acceptable')

    ax2.scatter(mach_numbers_acc, angular_distances_acc, [20] * len(mach_numbers_acc),
                c=max_heat_flux_acc, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                marker='o', s=20, edgecolor=None, alpha=0.8)

    ax2.scatter([2] * len(mach_numbers_acc), angular_distances_acc, altitudes_acc / 1000,
                c=max_heat_flux_acc, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                marker='o', s=20, edgecolor=None, alpha=0.8)

    # Fail projections on walls (single label)
    sc2_fail = ax2.scatter(mach_numbers_fail, [0.5] * len(mach_numbers_fail), altitudes_fail / 1000,
                        c=max_heat_flux_fail, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                        marker='X', s=20, edgecolor=None, alpha=0.8, label='Fail')

    ax2.scatter(mach_numbers_fail, angular_distances_fail, [20] * len(mach_numbers_fail),
                c=max_heat_flux_fail, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                marker='X', s=20, edgecolor=None, alpha=0.8)

    ax2.scatter([2] * len(mach_numbers_fail), angular_distances_fail, altitudes_fail / 1000,
                c=max_heat_flux_fail, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                marker='X', s=20, edgecolor=None, alpha=0.8)

    # Add a single colorbar for heat flux
    cbar = fig2.colorbar(sc2_success, ax=ax2, label="Max Heat Flux [kW/m²]", pad=0.1, shrink=0.9)
    cbar.ax.tick_params(labelsize=12) 
    cbar.set_label("Max Heat Flux [kW/m²]", fontsize=12)

    # Add dotted lines for all combinations, including Altitude-Angular Distance
    ax2.plot([mach_threshold, mach_threshold], [0.5, angular_distance_threshold], [20, 20], linestyle='dotted', color='red', label='Terminal Condition Limits')
    ax2.plot([mach_threshold, mach_threshold], [0.5, 0.5], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
    ax2.plot([2, mach_threshold], [angular_distance_threshold, angular_distance_threshold], [20, 20], linestyle='dotted', color='red')
    ax2.plot([2, 2], [angular_distance_threshold, angular_distance_threshold], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
    ax2.plot([2, mach_threshold], [0.5, 0.5], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')

    # New: Altitude-Angular Distance combination
    ax2.plot([2, 2], [0.5, angular_distance_threshold], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')

    # Set labels and title
    ax2.set_xlabel("Mach Number [-]", fontsize = 14)
    ax2.set_zlabel("Altitude [km]", fontsize = 14)
    ax2.set_ylabel("Angular Distance [deg]", fontsize = 14)
    ax2.set_xlim(2, 3.5)
    ax2.set_zlim(20, 32)
    ax2.set_ylim(0.5, 1.3)
    ax2.legend(fontsize = 12)

    plt.show()

if only_failed_simulations : 
    # Define the base directory where to look for results
    if wind:
        first_dir = os.path.join( current_dir, "SimulationOutput", "Wind")
    else :
        first_dir = os.path.join( current_dir, "SimulationOutput", "NoWind")

    # Dictionary to store the data for Success, Fail
    if model == "Predictor-Corrector":
        results1 = {
            'Success': {},
            'Fail': {}
        }
        results2 = {
            'Success': {},
            'Fail': {}
        }
        # Specify the directories within SimulationOutput to be analysed
        valid_categories = ['Success', 'Fail']
        base_dir = os.path.join(first_dir, "PredCorr")
    else : 
        results1 = {
            'Success': {},
            'Acceptable': {},
            'Fail': {}
        }
        results2 = {
            'Success': {},
            'Acceptable':{},
            'Fail': {}
        }
        # Specify the directories within SimulationOutput to be analysed
        valid_categories = ['Success', 'Acceptable','Fail']
        base_dir = os.path.join(first_dir, model)

    # Columns of interest (Mach, Altitude and Angular Distance) for the last values
    column_indices1 = [ 1, 2, 33 ]  
    # Columns of interest (G-load and Heat-Flux) for the max values
    column_indices2 = [ 11, 24 ]
    # Columns to convert from radians to degrees
    columns_to_convert = [ 33 ]
    # Loop between all the valid categories to collect results 
    for category in valid_categories:
        category_dir = os.path.join(base_dir, category)
        # Check if the category directory exists
        if os.path.exists(category_dir):
            # Walk through the directory structure in this category
            for root, dirs, files in os.walk(category_dir):
                if 'Dependent_Variables.dat' in files:
                    # Get the run directory name (e.g., run_14000, run_34829)
                    run_dir = os.path.basename(root)
                    # Full path to the Dependent_Variables.dat file
                    dep_var_file = os.path.join(root, 'Dependent_Variables.dat')
                    # Read the last row's data for the specified columns
                    column_data1 = read_last_row_columns(dep_var_file, column_indices1, columns_to_convert)
                    column_data2 = read_columns_max_value(dep_var_file, column_indices2)
                    # Store the data in the results dictionary under the respective category
                    data_dict1 = {f'column_{i+1}': value for i, value in enumerate(column_data1)}
                    data_dict2 = {f'column_{i+1}': value for i, value in enumerate(column_data2)}
                    results1[category][run_dir] = data_dict1
                    results2[category][run_dir] = data_dict2

    if model == "Predictor-Corrector" :
        # Results have been collected, define the threshold for the different quantities
        mach_threshold = 3 # [-]
        altitude_threshold = 30 # [m]
        angular_distance_threshold = 1 # [deg]

        mach_numbers_fail = []
        altitudes_fail =[]
        angular_distances_fail = []
        max_g_load_fail = []
        max_heat_flux_fail = []

        # Plot the points for Success and Fail
        for category in ['Success', 'Fail']:
            if category in results1:
                for run, data in results1[category].items():
                    if 'column_1' in data and 'column_2' in data and 'column_3' in data :
                        try:
                            if category == 'Fail' : 
                                mach_numbers_fail.append(data['column_1'])
                                altitudes_fail.append(data['column_2'])
                                angular_distances_fail.append(data['column_3'])
                        except KeyError as e:
                            print(f"Error: Missing key {e} for run: {run} in category: {category}")
                    else:
                        # Print a warning if the keys are missing
                        print(f"Warning: 'column_1',  'column_2' or 'column_3' missing for run: {run} in category: {category}")
            if category in results2 :
                for run, data in results2[category].items():
                    if  'column_1' in data and 'column_2' in data : 
                        try : 
                            if category == 'Fail' :
                                max_g_load_fail.append(data['column_1'])
                                max_heat_flux_fail.append(data['column_2'])
                        except KeyError as e:
                            print(f"Error: Missing key {e} for run: {run} in category: {category}")
                    else:
                        # Print a warning if the keys are missing
                        print(f"Warning: 'column_1' or 'column_2' missing for run: {run} in category: {category}") 

        # Prepare the lists with all the maximum g-loads and heat fluxes
        all_g_loads = []
        all_heat_fluxes = []

        for k in range(len(max_g_load_fail)): 
            all_g_loads.append(max_g_load_fail[k])
            all_heat_fluxes.append(max_heat_flux_fail[k])

        mach_numbers_fail = np.array(mach_numbers_fail)
        angular_distances_fail = np.array(angular_distances_fail)
        altitudes_fail = np.array(altitudes_fail)
        max_heat_flux_fail = np.array(max_heat_flux_fail)

        all_g_loads = np.array(all_g_loads)
        all_heat_fluxes = np.array(all_heat_fluxes)

        # Plot 1: Color based on maximum g-load
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        # Fail points
        sc1_fail = ax1.scatter(mach_numbers_fail, angular_distances_fail, altitudes_fail/1e3, c=max_g_load_fail[:],
                            cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads), marker='X', s=20, alpha = 0.8, edgecolor=None)
        # Setting labels and adding a single color bar
        ax1.set_xlabel("Mach Number [-]", fontsize = 14)
        ax1.set_zlabel("Altitude [km]", fontsize = 14)
        ax1.set_ylabel("Angular Distance [deg]", fontsize = 14)
        ax1.set_xlim(2, 4.5)
        ax1.set_zlim(20, 34)
        ax1.set_ylim(0.5, 2)
        plt.colorbar(sc1_fail, ax=ax1, label="Max G-Load [-]", pad=0.1, shrink=0.9)  # One color bar for all points
        # Plot threshold planes for success criteria
        ax1.plot_surface(np.array([[mach_threshold, mach_threshold], [mach_threshold, mach_threshold]]),
                        np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                        np.array([[20, altitude_threshold], [20, altitude_threshold]]),
                        color='lightgrey', alpha=0.3, rstride=100, cstride=100)
        ax1.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                        np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                        np.array([[altitude_threshold, altitude_threshold], [altitude_threshold, altitude_threshold]]),
                        color='lightgrey', alpha=0.3, rstride=100, cstride=100)
        ax1.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                        np.array([[angular_distance_threshold, angular_distance_threshold], [angular_distance_threshold, angular_distance_threshold]]),
                        np.array([[20, 20], [altitude_threshold, altitude_threshold]]),
                        color='lightgrey', alpha=0.3, rstride=100, cstride=100)
        if wind :
            ax1.set_title("Failed Re-Entry simulations terminal conditions with Wind", fontsize=16, fontweight='bold')
        else :
            ax1.set_title("Failed Re-Entry simulations terminal conditions", fontsize=16, fontweight='bold')
        ax1.legend()
        plt.show()

        # Plot 2: Color based on maximum heat flux
        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111, projection='3d')
        # Fail points
        sc2_fail = ax2.scatter(mach_numbers_fail, angular_distances_fail, altitudes_fail/1e3, c=max_heat_flux_fail,
                            cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes), marker='X', s=20, alpha = 0.8, edgecolor=None)
        # Setting labels and adding a single color bar
        ax2.set_xlabel("Mach Number [-]", fontsize = 14)
        ax2.set_zlabel("Altitude [km]", fontsize = 14)
        ax2.set_ylabel("Angular Distance [deg]", fontsize = 14)
        ax2.set_xlim(2, 4)
        ax2.set_zlim(20, 32)
        ax2.set_ylim(0.5, 1.5)
        plt.colorbar(sc2_fail, ax=ax2, label="Max Heat Flux [kW/m²]", pad=0.1, shrink=0.9)
        # Plot threshold planes for success criteria
        ax2.plot_surface(np.array([[mach_threshold, mach_threshold], [mach_threshold, mach_threshold]]),
                        np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                        np.array([[20, altitude_threshold], [20, altitude_threshold]]),
                        color='lightgrey', alpha=0.3, rstride=100, cstride=100)
        ax2.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                        np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                        np.array([[altitude_threshold, altitude_threshold], [altitude_threshold, altitude_threshold]]),
                        color='lightgrey', alpha=0.3, rstride=100, cstride=100)
        ax2.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                        np.array([[angular_distance_threshold, angular_distance_threshold], [angular_distance_threshold, angular_distance_threshold]]),
                        np.array([[20, 20], [altitude_threshold, altitude_threshold]]),
                        color='lightgrey', alpha=0.3, rstride=100, cstride=100)
        if wind:
            ax2.set_title("Failed Re-Entry simulations terminal conditions with Wind", fontsize=16, fontweight='bold')
        else :
            ax2.set_title("Failed Re-Entry simulations terminal conditions", fontsize=16, fontweight='bold')
        ax2.legend(fontsize = 12)
        plt.show()

        # Thresholds
        mach_threshold = 3
        altitude_threshold = 30000  # Altitude in meters
        angular_distance_threshold = 1 #[deg]

        # Plot
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111, projection='3d')

        # Fail projections on walls (single label)
        sc1_fail = ax1.scatter(mach_numbers_fail, [0.5] * len(mach_numbers_fail), altitudes_fail / 1000,
                            c=max_g_load_fail, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                            marker='X', s=20, edgecolor=None, alpha=0.8)

        ax1.scatter(mach_numbers_fail, angular_distances_fail, [20] * len(mach_numbers_fail),
                    c=max_g_load_fail, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                    marker='X', s=20, edgecolor=None, alpha=0.8)

        ax1.scatter([2] * len(mach_numbers_fail), angular_distances_fail, altitudes_fail / 1000,
                    c=max_g_load_fail, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                    marker='X', s=20, edgecolor=None, alpha=0.8)

        # Add a single colorbar for g-load
        cbar = fig1.colorbar(sc1_fail, ax=ax1, label="Max G-Load [-]", pad=0.1, shrink=0.9)

        # Add dotted lines for all combinations, including Altitude-Angular Distance
        ax1.plot([mach_threshold, mach_threshold], [0.5, angular_distance_threshold], [20, 20], linestyle='dotted', color='red', label='Terminal Condition Limits')
        ax1.plot([mach_threshold, mach_threshold], [0.5, 0.5], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
        ax1.plot([2, mach_threshold], [angular_distance_threshold, angular_distance_threshold], [20, 20], linestyle='dotted', color='red')
        ax1.plot([2, 2], [angular_distance_threshold, angular_distance_threshold], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
        ax1.plot([2, mach_threshold], [0.5, 0.5], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')
        ax1.plot([2, 2], [0.5, angular_distance_threshold], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')

        # Set labels and title
        ax1.set_xlabel("Mach Number [-]", fontsize = 14)
        ax1.set_zlabel("Altitude [km]", fontsize = 14)
        ax1.set_ylabel("Angular Distance [deg]", fontsize = 14)
        ax1.set_xlim(2, 4.5)
        ax1.set_zlim(20, 34)
        ax1.set_ylim(0.5, 2)
        if wind :
            ax1.set_title("Failed Re-Entry Simulations projected terminal conditions with Wind", fontsize=16, fontweight='bold')
        else :
            ax1.set_title("Failed Re-Entry Simulations projected terminal conditions", fontsize=16, fontweight='bold')
        ax1.legend(loc="upper left", fontsize = 12)

        plt.show()

        # Plot
        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111, projection='3d')

        # Fail projections on walls (single label)
        sc2_fail = ax2.scatter(mach_numbers_fail, [0.5] * len(mach_numbers_fail), altitudes_fail / 1000,
                            c=max_heat_flux_fail, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                            marker='X', s=20, edgecolor=None, alpha=0.8, label='Fail')

        ax2.scatter(mach_numbers_fail, angular_distances_fail, [20] * len(mach_numbers_fail),
                    c=max_heat_flux_fail, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                    marker='X', s=20, edgecolor=None, alpha=0.8)

        ax2.scatter([2] * len(mach_numbers_fail), angular_distances_fail, altitudes_fail / 1000,
                    c=max_heat_flux_fail, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                    marker='X', s=20, edgecolor=None, alpha=0.8)

        # Add a single colorbar for heat flux
        cbar = fig2.colorbar(sc2_fail, ax=ax2, label="Max Heat Flux [kW/m²]", pad=0.1, shrink=0.9)

        # Add dotted lines for all combinations, including Altitude-Angular Distance
        ax2.plot([mach_threshold, mach_threshold], [0.5, angular_distance_threshold], [20, 20], linestyle='dotted', color='red', label='Terminal Condition Limits')
        ax2.plot([mach_threshold, mach_threshold], [0.5, 0.5], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
        ax2.plot([2, mach_threshold], [angular_distance_threshold, angular_distance_threshold], [20, 20], linestyle='dotted', color='red')
        ax2.plot([2, 2], [angular_distance_threshold, angular_distance_threshold], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
        ax2.plot([2, mach_threshold], [0.5, 0.5], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')

        # New: Altitude-Angular Distance combination
        ax2.plot([2, 2], [0.5, angular_distance_threshold], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')

        # Set labels and title
        ax2.set_xlabel("Mach Number [-]", fontsize = 14)
        ax2.set_zlabel("Altitude [km]", fontsize = 14)
        ax2.set_ylabel("Angular Distance [deg]", fontsize = 14)
        ax2.set_xlim(2, 4.5)
        ax2.set_zlim(20, 34)
        ax2.set_ylim(0.5, 2)
        if wind :
            ax2.set_title("Failed Re-Entry Simulations projected terminal conditions with Wind", fontsize=16, fontweight='bold')
        else :
            ax2.set_title("Failed Re-Entry Simulations projected terminal conditions", fontsize=16, fontweight='bold')
        ax2.legend(loc="upper left", fontsize = 12)

        plt.show()
    else :
        # Results have been collected, define the threshold for the different quantities
        mach_threshold = 3 # [-]
        altitude_threshold = 30 # [m]
        angular_distance_threshold = 1 # [deg]

        mach_numbers_fail = []
        altitudes_fail =[]
        angular_distances_fail = []
        max_g_load_fail = []
        max_heat_flux_fail = []

        # Plot the points for Success and Fail
        for category in valid_categories:
            if category in results1:
                for run, data in results1[category].items():
                    if 'column_1' in data and 'column_2' in data and 'column_3' in data :
                        try:
                            if category == 'Fail' : 
                                mach_numbers_fail.append(data['column_1'])
                                altitudes_fail.append(data['column_2'])
                                angular_distances_fail.append(data['column_3'])
                        except KeyError as e:
                            print(f"Error: Missing key {e} for run: {run} in category: {category}")
                    else:
                        # Print a warning if the keys are missing
                        print(f"Warning: 'column_1',  'column_2' or 'column_3' missing for run: {run} in category: {category}")
            if category in results2 :
                for run, data in results2[category].items():
                    if  'column_1' in data and 'column_2' in data : 
                        try : 
                            if category == 'Fail' :
                                max_g_load_fail.append(data['column_1'])
                                max_heat_flux_fail.append(data['column_2'])
                        except KeyError as e:
                            print(f"Error: Missing key {e} for run: {run} in category: {category}")
                    else:
                        # Print a warning if the keys are missing
                        print(f"Warning: 'column_1' or 'column_2' missing for run: {run} in category: {category}") 

        # Prepare the lists with all the maximum g-loads and heat fluxes
        all_g_loads = []
        all_heat_fluxes = []

        for k in range(len(max_g_load_fail)): 
            all_g_loads.append(max_g_load_fail[k])
            all_heat_fluxes.append(max_heat_flux_fail[k])

        mach_numbers_fail = np.array(mach_numbers_fail)
        angular_distances_fail = np.array(angular_distances_fail)
        altitudes_fail = np.array(altitudes_fail)
        max_heat_flux_fail = np.array(max_heat_flux_fail)

        all_g_loads = np.array(all_g_loads)
        all_heat_fluxes = np.array(all_heat_fluxes)

        # Plot 1: Color based on maximum g-load
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        # Fail points
        sc1_fail = ax1.scatter(mach_numbers_fail, angular_distances_fail, altitudes_fail/1e3, c=max_g_load_fail[:],
                            cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads), marker='X', s=20, alpha = 0.8, edgecolor=None)
        # Setting labels and adding a single color bar
        ax1.set_xlabel("Mach Number [-]", fontsize = 14)
        ax1.set_zlabel("Altitude [km]", fontsize = 14)
        ax1.set_ylabel("Angular Distance [deg]", fontsize = 14)
        ax1.set_xlim(2, 4.5)
        ax1.set_zlim(20, 34)
        ax1.set_ylim(0.5, 2)
        plt.colorbar(sc1_fail, ax=ax1, label="Max G-Load [-]", pad=0.1, shrink=0.9)  # One color bar for all points
        # Plot threshold planes for success criteria
        ax1.plot_surface(np.array([[mach_threshold, mach_threshold], [mach_threshold, mach_threshold]]),
                        np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                        np.array([[20, altitude_threshold], [20, altitude_threshold]]),
                        color='lightgrey', alpha=0.3, rstride=100, cstride=100)
        ax1.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                        np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                        np.array([[altitude_threshold, altitude_threshold], [altitude_threshold, altitude_threshold]]),
                        color='lightgrey', alpha=0.3, rstride=100, cstride=100)
        ax1.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                        np.array([[angular_distance_threshold, angular_distance_threshold], [angular_distance_threshold, angular_distance_threshold]]),
                        np.array([[20, 20], [altitude_threshold, altitude_threshold]]),
                        color='lightgrey', alpha=0.3, rstride=100, cstride=100)
        if wind :
            ax1.set_title(f"Failed Re-Entry simulations terminal conditions with Wind using {model}", fontsize=16, fontweight='bold')
        else :
            ax1.set_title(f"Failed Re-Entry simulations terminal conditions using {model}", fontsize=16, fontweight='bold')
        ax1.legend(fontsize = 12)
        plt.show()

        # Plot 2: Color based on maximum heat flux
        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111, projection='3d')
        # Fail points
        sc2_fail = ax2.scatter(mach_numbers_fail, angular_distances_fail, altitudes_fail/1e3, c=max_heat_flux_fail,
                            cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes), marker='X', s=20, alpha = 0.8, edgecolor=None)
        # Setting labels and adding a single color bar
        ax2.set_xlabel("Mach Number [-]", fontsize = 14)
        ax2.set_zlabel("Altitude [km]", fontsize = 14)
        ax2.set_ylabel("Angular Distance [deg]", fontsize = 14)
        ax2.set_xlim(2, 4)
        ax2.set_zlim(20, 32)
        ax2.set_ylim(0.5, 1.5)
        plt.colorbar(sc2_fail, ax=ax2, label="Max Heat Flux [kW/m²]", pad=0.1, shrink=0.9)
        # Plot threshold planes for success criteria
        ax2.plot_surface(np.array([[mach_threshold, mach_threshold], [mach_threshold, mach_threshold]]),
                        np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                        np.array([[20, altitude_threshold], [20, altitude_threshold]]),
                        color='lightgrey', alpha=0.3, rstride=100, cstride=100)
        ax2.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                        np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                        np.array([[altitude_threshold, altitude_threshold], [altitude_threshold, altitude_threshold]]),
                        color='lightgrey', alpha=0.3, rstride=100, cstride=100)
        ax2.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                        np.array([[angular_distance_threshold, angular_distance_threshold], [angular_distance_threshold, angular_distance_threshold]]),
                        np.array([[20, 20], [altitude_threshold, altitude_threshold]]),
                        color='lightgrey', alpha=0.3, rstride=100, cstride=100)
        if wind:
            ax2.set_title(f"Failed Re-Entry simulations terminal conditions with Wind using {model}", fontsize=16, fontweight='bold')
        else :
            ax2.set_title(f"Failed Re-Entry simulations terminal conditions using {model}", fontsize=16, fontweight='bold')
        ax2.legend(fontsize = 12)
        plt.show()

        # Thresholds
        mach_threshold = 3
        altitude_threshold = 30000  # Altitude in meters
        angular_distance_threshold = 1 #[deg]

        # Plot
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111, projection='3d')

        # Fail projections on walls (single label)
        sc1_fail = ax1.scatter(mach_numbers_fail, [0.5] * len(mach_numbers_fail), altitudes_fail / 1000,
                            c=max_g_load_fail, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                            marker='X', s=20, edgecolor=None, alpha=0.8)

        ax1.scatter(mach_numbers_fail, angular_distances_fail, [20] * len(mach_numbers_fail),
                    c=max_g_load_fail, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                    marker='X', s=20, edgecolor=None, alpha=0.8)

        ax1.scatter([2] * len(mach_numbers_fail), angular_distances_fail, altitudes_fail / 1000,
                    c=max_g_load_fail, cmap='plasma', vmin=min(all_g_loads), vmax=max(all_g_loads),
                    marker='X', s=20, edgecolor=None, alpha=0.8)

        # Add a single colorbar for g-load
        cbar = fig1.colorbar(sc1_fail, ax=ax1, label="Max G-Load [-]", pad=0.1, shrink=0.9)

        # Add dotted lines for all combinations, including Altitude-Angular Distance
        ax1.plot([mach_threshold, mach_threshold], [0.5, angular_distance_threshold], [20, 20], linestyle='dotted', color='red', label='Terminal Condition Limits')
        ax1.plot([mach_threshold, mach_threshold], [0.5, 0.5], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
        ax1.plot([2, mach_threshold], [angular_distance_threshold, angular_distance_threshold], [20, 20], linestyle='dotted', color='red')
        ax1.plot([2, 2], [angular_distance_threshold, angular_distance_threshold], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
        ax1.plot([2, mach_threshold], [0.5, 0.5], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')
        ax1.plot([2, 2], [0.5, angular_distance_threshold], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')

        # Set labels and title
        ax1.set_xlabel("Mach Number [-]", fontsize = 14)
        ax1.set_zlabel("Altitude [km]", fontsize = 14)
        ax1.set_ylabel("Angular Distance [deg]", fontsize = 14)
        ax1.set_xlim(2, 4.5)
        ax1.set_zlim(20, 34)
        ax1.set_ylim(0.5, 2)
        if wind :
            ax1.set_title(f"Failed Re-Entry Simulations projected terminal conditions with Wind using {model}", fontsize=16, fontweight='bold')
        else :
            ax1.set_title(f"Failed Re-Entry Simulations projected terminal conditions using {model}", fontsize=16, fontweight='bold')
        ax1.legend(loc="upper left", fontsize = 12)

        plt.show()

        # Plot
        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111, projection='3d')

        # Fail projections on walls (single label)
        sc2_fail = ax2.scatter(mach_numbers_fail, [0.5] * len(mach_numbers_fail), altitudes_fail / 1000,
                            c=max_heat_flux_fail, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                            marker='X', s=20, edgecolor=None, alpha=0.8, label='Fail')

        ax2.scatter(mach_numbers_fail, angular_distances_fail, [20] * len(mach_numbers_fail),
                    c=max_heat_flux_fail, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                    marker='X', s=20, edgecolor=None, alpha=0.8)

        ax2.scatter([2] * len(mach_numbers_fail), angular_distances_fail, altitudes_fail / 1000,
                    c=max_heat_flux_fail, cmap='viridis', vmin=min(all_heat_fluxes), vmax=max(all_heat_fluxes),
                    marker='X', s=20, edgecolor=None, alpha=0.8)

        # Add a single colorbar for heat flux
        cbar = fig2.colorbar(sc2_fail, ax=ax2, label="Max Heat Flux [kW/m²]", pad=0.1, shrink=0.9)

        # Add dotted lines for all combinations, including Altitude-Angular Distance
        ax2.plot([mach_threshold, mach_threshold], [0.5, angular_distance_threshold], [20, 20], linestyle='dotted', color='red', label='Terminal Condition Limits')
        ax2.plot([mach_threshold, mach_threshold], [0.5, 0.5], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
        ax2.plot([2, mach_threshold], [angular_distance_threshold, angular_distance_threshold], [20, 20], linestyle='dotted', color='red')
        ax2.plot([2, 2], [angular_distance_threshold, angular_distance_threshold], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
        ax2.plot([2, mach_threshold], [0.5, 0.5], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')

        # New: Altitude-Angular Distance combination
        ax2.plot([2, 2], [0.5, angular_distance_threshold], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')

        # Set labels and title
        ax2.set_xlabel("Mach Number [-]", fontsize = 14)
        ax2.set_zlabel("Altitude [km]", fontsize = 14)
        ax2.set_ylabel("Angular Distance [deg]", fontsize = 14)
        ax2.set_xlim(2, 4.5)
        ax2.set_zlim(20, 34)
        ax2.set_ylim(0.5, 2)
        if wind :
            ax2.set_title(f"Failed Re-Entry Simulations projected terminal conditions with Wind using {model} ", fontsize=16, fontweight='bold')
        else :
            ax2.set_title(f"Failed Re-Entry Simulations projected terminal conditions using {model} ", fontsize=16, fontweight='bold')
        ax2.legend(loc="upper left", fontsize = 12)

        plt.show()

if plot_Q_Load : 

    if wind:
        # Define the base directory where to look for results
        first_dir = os.path.join( current_dir, "SimulationOutput", "Wind")  
    else :
        # Define the base directory where to look for results
        first_dir = os.path.join( current_dir, "SimulationOutput", "NoWind")  
    
    if model == 'Predictor-Corrector': 
        base_dir = os.path.join(first_dir, 'PredCorr')
    else :
        base_dir = os.path.join(first_dir, model)

    results1 = {
        'Success': {},
        'Acceptable': {},
        'Fail': {}
    }
    results2 = {
        'Success': {},
        'Acceptable': {},
        'Fail': {}
    }
    # Specify the directories within SimulationOutput to be analysed
    valid_categories = ['Success', 'Acceptable', 'Fail']

    # Columns of interest (Mach, Altitude and Angular Distance) for the last values
    column_indices1 = [ 1, 2, 33 ]  
    # Columns of interest (Time series, Heat-Flux) to be integrated 
    column_indices2 = [ 0, 24 ]
    # Columns to convert from radians to degrees
    columns_to_convert = [ 33 ]
    # Loop between all the valid categories to collect results 
    for category in valid_categories:
        category_dir = os.path.join(base_dir, category)
        # Check if the category directory exists
        if os.path.exists(category_dir):
            # Walk through the directory structure in this category
            for root, dirs, files in os.walk(category_dir):
                if 'Dependent_Variables.dat' in files:
                    # Get the run directory name (e.g., run_14000, run_34829)
                    run_dir = os.path.basename(root)
                    # Full path to the Dependent_Variables.dat file
                    dep_var_file = os.path.join(root, 'Dependent_Variables.dat')
                    # Read the last row's data for the specified columns
                    column_data1 = read_last_row_columns(dep_var_file, column_indices1, columns_to_convert)
                    column_data2 = read_columns_all_rows(dep_var_file, column_indices2)
                    # Store the data in the results dictionary under the respective category
                    data_dict1 = {f'column_{i+1}': value for i, value in enumerate(column_data1)}
                    data_dict2 = {f'column_{i+1}': value for i, value in enumerate(column_data2)}
                    results1[category][run_dir] = data_dict1
                    results2[category][run_dir] = compute_Heat_Load(np.array(column_data2[24]), np.array(column_data2[0]))

    mach_numbers_success = []
    mach_numbers_acc = []
    mach_numbers_fail = []
    altitudes_success = []
    altitudes_acc = []
    altitudes_fail =[]
    angular_distances_success = []
    angular_distances_acc = []
    angular_distances_fail = []
    heat_load_success = []
    heat_load_acc = []
    heat_load_fail = []

    # Results have been collected, define the threshold for the different quantities
    mach_threshold = 3 # [-]
    altitude_threshold = 30 # [km]
    angular_distance_threshold = 1 # [deg]

    for category in valid_categories:
        if category in results1:
            for run, data in results1[category].items():
                if 'column_1' in data and 'column_2' in data and 'column_3' in data :
                    try:
                        if category == 'Success' : 
                            mach_numbers_success.append(data['column_1'])  
                            altitudes_success.append(data['column_2'])  
                            angular_distances_success.append(data['column_3'])
                        elif category == 'Fail' : 
                            mach_numbers_fail.append(data['column_1'])
                            altitudes_fail.append(data['column_2'])
                            angular_distances_fail.append(data['column_3'])
                        else :
                            mach_numbers_acc.append(data['column_1'])
                            altitudes_acc.append(data['column_2'])
                            angular_distances_acc.append(data['column_3'])
                    except KeyError as e:
                        print(f"Error: Missing key {e} for run: {run} in category: {category}")
                else:
                    # Print a warning if the keys are missing
                    print(f"Warning: 'column_1',  'column_2' or 'column_3' missing for run: {run} in category: {category}")
        if category in results2 :
            for run, data in results2[category].items():
                try : 
                    if category == 'Success' :
                        heat_load_success.append(data)
                    elif category == "Fail" : 
                        heat_load_fail.append(data)
                    else:
                        heat_load_acc.append(data)
                except KeyError as e:
                    print(f"Error: Missing key {e} for run: {run} in category: {category}")
            else:
                # Print a warning if the keys are missing
                print(f"Warning: 'column_1' or 'column_2' missing for run: {run} in category: {category}")

    all_heat_loads = []

    for i in range(len(heat_load_success)): 
        all_heat_loads.append(heat_load_success[i])

    for i in range(len(heat_load_acc)): 
        all_heat_loads.append(heat_load_acc[i])

    for k in range(len(heat_load_fail)): 
        all_heat_loads.append(heat_load_fail[k])

    altitude_max = 35000  # 36 km in meters
    mach_max = 4.5
    angular_distance_max = 2  # degrees

    # Function to filter data
    def filter_data(mach, altitude, angular_distance, heat_load):
        mask = (altitude <= altitude_max) & (mach <= mach_max) & (angular_distance <= angular_distance_max)
        return mach[mask], altitude[mask], angular_distance[mask], heat_load[mask]

    mach_numbers_success = np.array(mach_numbers_success)
    angular_distances_success = np.array(angular_distances_success)
    altitudes_success = np.array(altitudes_success)
    heat_load_success = np.array(heat_load_success)/1e3

    mach_numbers_acc = np.array(mach_numbers_acc)
    angular_distances_acc = np.array(angular_distances_acc)
    altitudes_acc = np.array(altitudes_acc)
    heat_load_acc = np.array(heat_load_acc)/1e3

    mach_numbers_fail = np.array(mach_numbers_fail)
    angular_distances_fail = np.array(angular_distances_fail)
    altitudes_fail = np.array(altitudes_fail)
    heat_load_fail = np.array(heat_load_fail)/1e3

    # Apply filtering
    mach_numbers_success, altitudes_success, angular_distances_success, heat_load_success = filter_data(
        mach_numbers_success, altitudes_success, angular_distances_success, heat_load_success
    )

    mach_numbers_acc, altitudes_acc, angular_distances_acc, heat_load_acc = filter_data(
        mach_numbers_acc, altitudes_acc, angular_distances_acc, heat_load_acc
    )

    mach_numbers_fail, altitudes_fail, angular_distances_fail, heat_load_fail = filter_data(
        mach_numbers_fail, altitudes_fail, angular_distances_fail, heat_load_fail
    )

    all_heat_loads = np.array(all_heat_loads)/1e3

    # Plot 2: Color based on heat-load
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    # Success points
    sc1_success = ax1.scatter(mach_numbers_success, angular_distances_success, altitudes_success/1e3, c=heat_load_success,
                            cmap='jet', vmin = 350, vmax=max(all_heat_loads), marker='o', s=20, edgecolor=None, alpha = 0.8, label='Success')
    # Success points
    sc1_acceptable = ax1.scatter(mach_numbers_acc, angular_distances_acc, altitudes_acc/1e3, c=heat_load_acc,
                            cmap='jet', vmin = 350, vmax=max(all_heat_loads), marker='v', s=20, edgecolor=None, alpha = 0.8, label='Acceptable')
    # Fail points
    sc1_fail = ax1.scatter(mach_numbers_fail, angular_distances_fail, altitudes_fail/1e3, c=heat_load_fail,
                        cmap='jet', vmin = 350, vmax=max(all_heat_loads), marker='X', s=20, edgecolor=None, alpha = 0.8, label='Fail')
    # Setting labels and adding a single color bar
    ax1.set_xlabel("Mach Number [-]", fontsize = 14)
    ax1.set_zlabel("Altitude [km]", fontsize = 14)
    ax1.set_ylabel("Angular Distance [deg]", fontsize = 14)
    ax1.set_xlim(2, 4)
    ax1.set_zlim(20, 34)
    ax1.set_ylim(0.5, 2)
    plt.colorbar(sc1_success, ax=ax1, label="Integrated Heat Load [MJ/m²]", )  # One color bar for all points
    # Plot threshold planes for success criteria
    ax1.plot_surface(np.array([[mach_threshold, mach_threshold], [mach_threshold, mach_threshold]]),
                    np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                    np.array([[20, altitude_threshold], [20, altitude_threshold]]),
                    color='lightgrey', alpha=0.3, rstride=100, cstride=100)
    ax1.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                    np.array([[0.5, 0.5], [angular_distance_threshold, angular_distance_threshold]]),
                    np.array([[altitude_threshold, altitude_threshold], [altitude_threshold, altitude_threshold]]),
                    color='lightgrey', alpha=0.3, rstride=100, cstride=100)
    ax1.plot_surface(np.array([[2, mach_threshold], [2, mach_threshold]]),
                    np.array([[angular_distance_threshold, angular_distance_threshold], [angular_distance_threshold, angular_distance_threshold]]),
                    np.array([[20, 20], [altitude_threshold, altitude_threshold]]),
                    color='lightgrey', alpha=0.3, rstride=100, cstride=100)

    ax1.legend(fontsize = 12)
    plt.show()

    # Plot
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')

    # Success projections on walls (single label)
    sc1_success = ax1.scatter(mach_numbers_success, [0.5] * len(mach_numbers_success), altitudes_success / 1000,
                            c=heat_load_success, cmap='jet', vmin = 350, vmax=max(all_heat_loads),
                            marker='o', s=20, edgecolor=None, alpha=0.8, label='Success')

    ax1.scatter(mach_numbers_success, angular_distances_success, [20] * len(mach_numbers_success),
                c=heat_load_success, cmap='jet', vmin = 350, vmax=max(all_heat_loads),
                marker='o', s=20, edgecolor=None, alpha=0.8)

    ax1.scatter([2] * len(mach_numbers_success), angular_distances_success, altitudes_success / 1000,
                c=heat_load_success, cmap='jet', vmin = 350, vmax=max(all_heat_loads),
                marker='o', s=20, edgecolor=None, alpha=0.8)

    # Success projections on walls (single label)
    sc1_acceptable = ax1.scatter(mach_numbers_acc, [0.5] * len(mach_numbers_acc), altitudes_acc / 1000,
                            c=heat_load_acc, cmap='jet', vmin = 350, vmax=max(all_heat_loads),
                            marker='v', s=20, edgecolor=None, alpha=0.8, label='Acceptable')

    ax1.scatter(mach_numbers_acc, angular_distances_acc, [20] * len(mach_numbers_acc),
                c=heat_load_acc, cmap='jet', vmin = 350, vmax=max(all_heat_loads),
                marker='v', s=20, edgecolor=None, alpha=0.8)

    ax1.scatter([2] * len(mach_numbers_acc), angular_distances_acc, altitudes_acc / 1000,
                c=heat_load_acc, cmap='jet', vmin = 350, vmax=max(all_heat_loads),
                marker='v', s=20, edgecolor=None, alpha=0.8)

    # Fail projections on walls (single label)
    sc1_fail = ax1.scatter(mach_numbers_fail, [0.5] * len(mach_numbers_fail), altitudes_fail / 1000,
                        c=heat_load_fail, cmap='jet', vmin= 350, vmax=max(all_heat_loads),
                        marker='X', s=20, edgecolor=None, alpha=0.8, label='Fail')

    ax1.scatter(mach_numbers_fail, angular_distances_fail, [20] * len(mach_numbers_fail),
                c=heat_load_fail, cmap='jet', vmin=350, vmax=max(all_heat_loads),
                marker='X', s=20, edgecolor=None, alpha=0.8)

    ax1.scatter([2] * len(mach_numbers_fail), angular_distances_fail, altitudes_fail / 1000,
                c=heat_load_fail, cmap='jet', vmin=350, vmax=max(all_heat_loads),
                marker='X', s=20, edgecolor=None, alpha=0.8)

    # Add a single colorbar for g-load
    cbar = fig1.colorbar(sc1_success, ax=ax1, label="Integrated Heat Load [MJ/m²]", pad=0.1, shrink=0.9)
    cbar.ax.tick_params(labelsize=12) 
    cbar.set_label("Integrated Heat Load [MJ/m²]", fontsize=12)

    altitude_threshold = 30000

    # Add dotted lines for all combinations, including Altitude-Angular Distance
    ax1.plot([mach_threshold, mach_threshold], [0.5, angular_distance_threshold], [20, 20], linestyle='dotted', color='red', label='Terminal Condition Limits')
    ax1.plot([mach_threshold, mach_threshold], [0.5, 0.5], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
    ax1.plot([2, mach_threshold], [angular_distance_threshold, angular_distance_threshold], [20, 20], linestyle='dotted', color='red')
    ax1.plot([2, 2], [angular_distance_threshold, angular_distance_threshold], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
    ax1.plot([2, mach_threshold], [0.5, 0.5], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')
    ax1.plot([2, 2], [0.5, angular_distance_threshold], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')

    # Set labels and title
    ax1.set_xlabel("Mach Number [-]", fontsize = 14)
    ax1.set_zlabel("Altitude [km]", fontsize = 14)
    ax1.set_ylabel("Angular Distance [deg]", fontsize = 14)
    ax1.set_xlim(2, 3.5)
    ax1.set_zlim(20, 32)
    ax1.set_ylim(0.5, 1.3)

    ax1.legend(loc="upper left", fontsize = 12)

    plt.show()

    # Plot 3
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Fail projections on walls (single label)
    sc2_fail = ax2.scatter(mach_numbers_fail, [0.5] * len(mach_numbers_fail), altitudes_fail / 1000,
                        c=heat_load_fail, cmap='jet', vmin=350, vmax=max(all_heat_loads),
                        marker='X', s=20, edgecolor=None, alpha=0.8, label='Fail')

    ax2.scatter(mach_numbers_fail, angular_distances_fail, [20] * len(mach_numbers_fail),
                c=heat_load_fail, cmap='jet', vmin=350, vmax=max(all_heat_loads),
                marker='X', s=20, edgecolor=None, alpha=0.8)

    ax2.scatter([2] * len(mach_numbers_fail), angular_distances_fail, altitudes_fail / 1000,
                c=heat_load_fail, cmap='jet', vmin=350, vmax=max(all_heat_loads),
                marker='X', s=20, edgecolor=None, alpha=0.8)

    # Add a single colorbar for heat flux
    cbar = fig2.colorbar(sc2_fail, ax=ax2, label="Integrated Heat Load [MJ/m²]", pad=0.1, shrink=0.9)
    cbar.ax.tick_params(labelsize=12) 
    cbar.set_label("Integrated Heat Load [MJ/m²]", fontsize=12)
    

    # Add dotted lines for all combinations, including Altitude-Angular Distance
    ax2.plot([mach_threshold, mach_threshold], [0.5, angular_distance_threshold], [20, 20], linestyle='dotted', color='red', label='Terminal Condition Limits')
    ax2.plot([mach_threshold, mach_threshold], [0.5, 0.5], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
    ax2.plot([2, mach_threshold], [angular_distance_threshold, angular_distance_threshold], [20, 20], linestyle='dotted', color='red')
    ax2.plot([2, 2], [angular_distance_threshold, angular_distance_threshold], [20, altitude_threshold / 1000], linestyle='dotted', color='red')
    ax2.plot([2, mach_threshold], [0.5, 0.5], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')

    # New: Altitude-Angular Distance combination
    ax2.plot([2, 2], [0.5, angular_distance_threshold], [altitude_threshold / 1000, altitude_threshold / 1000], linestyle='dotted', color='red')

    # Set labels and title
    ax2.set_xlabel("Mach Number [-]", fontsize = 14)
    ax2.set_zlabel("Altitude [km]", fontsize = 14)
    ax2.set_ylabel("Angular Distance [deg]", fontsize = 14)
    ax2.set_xlim(2, 3.5)
    ax2.set_zlim(20, 32)
    ax2.set_ylim(0.5, 1.3)
    ax2.legend(loc="upper left", fontsize = 12)

    print('QLoad Max')
    print(max(all_heat_loads))

    tot_heat = sum(all_heat_loads)
    av = tot_heat / (500e3)

    print('Av QLoad')
    print(av)

    plt.show()
        

if plot_time_histograms :

    CONDITION = lambda x: x > 1.0
    if wind:
        # Define the base directory where to look for results
        first_dir = os.path.join(current_dir, "SimulationOutput", "Wind")
    else :
        # Define the base directory where to look for results
        first_dir = os.path.join(current_dir, "SimulationOutput", "NoWind")
    if model == "Predictor-Corrector":
        valid_categories = ['Success', 'Fail']
        base_dir = os.path.join(first_dir, "PredCorr")
    else :
        valid_categories = ['Success', 'Acceptable', 'Fail']
        base_dir = os.path.join(first_dir, model)
    # Open the lists for the cycle and propagation times
    propagation_times = []
    average_cycle_times = []
    for category in valid_categories:
        category_dir = os.path.join(base_dir, category)
        # Check if the category directory exists
        if os.path.exists(category_dir):
            # Walk through the directory structure in this category
            for root, dirs, files in os.walk(category_dir):
                if 'Results.json' in files:
                    # Get the run directory name (e.g., run_14000, run_34829)
                    run_dir = os.path.basename(root)
                    # Full path to the Dependent_Variables.dat file
                    results_file = os.path.join(root, 'Results.json')
                    train_file = os.path.join(root, 'Train.json')
                    with open(results_file, 'r') as file : 
                        data = json.load(file)
                        with open(train_file, 'r') as file : 
                            train = json.load(file)
                            last_key = max(train, key=float)  # Convert keys to float for sorting
                            last_entry = train[last_key]
                            commanded_bank = last_entry.get("commanded bank [rad]", None)
                            if data['Propagation Time [s]'] < 90 and commanded_bank > 1.43:
                                propagation_times.append(data['Propagation Time [s]'])
                                cycle_times = [
                                    entry["cycle time [s]"]
                                    for key, entry in train.items()
                                    if isinstance(entry, dict) and "cycle time [s]" in entry
                                ]
                                average_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0
                                average_cycle_times.append(average_cycle_time)
                            else :
                                pass

    if model == "Predictor-Corrector":
        # Create horizontal subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Plot Propagation Times Histogram
        axes[0].hist(propagation_times, bins=15, alpha=0.8, color='blue', edgecolor='black')
        axes[0].set_xlabel('CPU Times [s]', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Histogram of CPU times ', fontsize=14)

        # Plot Average Cycle Times Histogram
        axes[1].hist(average_cycle_times, bins=15, alpha=0.8, color='green', edgecolor='black')
        axes[1].set_xlabel('Average Cycle Times [s]', fontsize=12)
        axes[1].set_title('Histogram of Average Cycle Times', fontsize=14)

        # Adjust layout for better appearance
        plt.tight_layout()
        plt.show()
    else :
        # Create horizontal subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Plot Propagation Times Histogram
        axes[0].hist(propagation_times, bins=15, alpha=0.8, color='blue', edgecolor='black')
        axes[0].set_xlabel('CPU Times [s]', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Histogram of CPU times using {model}', fontsize=14)

        # Plot Average Cycle Times Histogram
        axes[1].hist(average_cycle_times, bins=15, alpha=0.8, color='green', edgecolor='black')
        axes[1].set_xlabel('Average Cycle Times [s]', fontsize=12)
        axes[1].set_title(f'Histogram of Average Cycle Times using {model}', fontsize=14)

        # Adjust layout for better appearance
        plt.tight_layout()
        plt.show()


if move_results : 
    current_dir = os.getcwd()  # or set this explicitly if needed
    fail_dir = os.path.join(current_dir, "SimulationOutput", "Wind", "PredCorr", "Fail")
    acceptable_dir = os.path.join(current_dir, "SimulationOutput", "Wind", "PredCorr", "Acceptable")

    # Create the Acceptable directory if it doesn't exist
    if not os.path.exists(acceptable_dir):
        os.makedirs(acceptable_dir)
        print(f"Created directory: {acceptable_dir}")

    # Iterate over each run folder in the Fail directory
    for run_folder in os.listdir(fail_dir):
        run_path = os.path.join(fail_dir, run_folder)
        # Ensure we process only directories starting with "run_"
        if os.path.isdir(run_path) and run_folder.startswith("run_"):
            results_file = os.path.join(run_path, "Results.json")
            # Check if the Results.json file exists in the run folder
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    data = json.load(f)
                # Extract the required metrics
                heat_flux = data.get("Maximum Heat Flux [kW/m^2]", None)
                g_load = data.get("Maximum G-Load [g]", None)
                # Verify that both values exist and check the constraints
                if heat_flux is not None and g_load is not None:
                    if heat_flux < 530.0 and g_load < 3.0:
                        # Construct the destination path in the Acceptable folder
                        dest_path = os.path.join(acceptable_dir, run_folder)
                        # Move the entire run folder from Fail to Acceptable
                        shutil.move(run_path, dest_path)
                        print(f"Moved {run_folder} to Acceptable folder.")

if delete_fake_fail : 
    # Define the requirements
    REQUIREMENTS = {
        "Final Mach Number [-]": lambda x: x < 3,
        "Final Altitude [m]": lambda x: x < 30000,
        "Final Angular Distance [deg]": lambda x: x < 1,
        "Maximum G-Load [g]": lambda x: x < 3
    }
    # Paths
    base_dir = "SimulationOutput"
    fail_dir = os.path.join(base_dir, "Fail")
    # Process each run in Fail
    for run_dir in os.listdir(fail_dir):
        run_path = os.path.join(fail_dir, run_dir)
        if not os.path.isdir(run_path):  # Skip non-directory files
            continue
        results_file = os.path.join(run_path, "Results.json")
        if not os.path.isfile(results_file):  # Skip if Results.json doesn't exist
            print(f"Skipping {run_path}, Results.json not found.")
            continue
        # Read and evaluate the Results.json file
        with open(results_file, "r") as file:
            try:
                results = json.load(file)
            except json.JSONDecodeError:
                print(f"Error reading {results_file}, skipping.")
                continue
        # Check requirements
        meets_requirements = all(req_func(results.get(key, None)) for key, req_func in REQUIREMENTS.items() if key in results)
        # Delete the run folder if all requirements are respected
        if meets_requirements:
            shutil.rmtree(run_path)
            print(f"Deleted {run_path} as it meets all requirements.")

if train_postprocess : 

    base_dir = os.path.join( current_dir, "PINN_Model")

    widths = [ 32, 64 ]
    learning_rates = [ 0.001, 0.01, 0.1 ]

    losses_data = {}

    for w in widths:
        for lr in learning_rates:
            model_dir = os.path.join(base_dir, f'width_{w}', f'lr_{lr}')

            # Construct paths to CSV files
            data_loss_path = os.path.join(model_dir, 'train_data_loss.csv')
            physics_loss_path = os.path.join(model_dir, 'train_physics_loss.csv')
            total_loss_path = os.path.join(model_dir, 'train_total_loss.csv')
            
            # Check if they exist (in case some combination is missing)
            if (not os.path.exists(data_loss_path) or 
                not os.path.exists(physics_loss_path) or 
                not os.path.exists(total_loss_path)):
                print(f"Skipping (width={w}, lr={lr}) - missing CSV files.")
                continue
            
            # Load the CSVs into NumPy arrays
            data_loss = np.loadtxt(data_loss_path, delimiter=',')
            physics_loss = np.loadtxt(physics_loss_path, delimiter=',')
            total_loss = np.loadtxt(total_loss_path, delimiter=',')
            
            # Store in the dictionary
            losses_data[(w, lr)] = {
                'data': data_loss,
                'physics': physics_loss,
                'total': total_loss
            }

    # 2. PLOT ALL TOTAL LOSSES ON THE SAME FIGURE
    plt.figure(figsize=(8, 6))
    for (w, lr), loss_dict in losses_data.items():
        plt.plot(loss_dict['total'], label=f'width={w}, lr={lr}')
    plt.xlabel("Epoch [-]", fontsize = 14)
    plt.ylabel("Total Loss [-]", fontsize = 14)
    plt.yscale("log")
    plt.legend(fontsize = 12, loc = "upper right", frameon=True, shadow=True)
    plt.grid(True)
    plt.show()

    # Generate improved plots for each model configuration using the already loaded losses_data
    for (w, lr), loss_dict in losses_data.items():
        plt.figure(figsize=(8, 6))
        
        # Plot each loss type with smoother styling
        plt.plot(loss_dict['total'], label="Total Loss", color='#1f77b4', linewidth=2)
        plt.plot(loss_dict['data'], label="Data Loss", color='#ff7f0e', linestyle="dashed", linewidth=2)
        plt.plot(loss_dict['physics'], label="Physics Loss", color='#2ca02c', linestyle="dotted", linewidth=2)
        
        # Enhance plot appearance
        plt.xlabel("Epoch [-]", fontsize=16)
        plt.ylabel("Loss [-]", fontsize=16)
        plt.yscale("log")  # Log scale for better visualization
        plt.legend(fontsize=14, loc='upper right', frameon=True, shadow=True)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Show the plot
        plt.tight_layout()
        plt.show()

    # 3. PLOT ALL DATA/PHYSICS/TOTAL LOSS FOR EACH MODEL SEPARATELY (OPTIONAL)
    for (w, lr), loss_dict in losses_data.items():
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
        fig.suptitle(f"Losses for width={w}, lr={lr}")
        
        # Data Loss
        ax[0].plot(loss_dict['data'], label="Data Loss", color='orange')
        ax[0].set_title("Data Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].grid(True)
        
        # Physics Loss
        ax[1].plot(loss_dict['physics'], label="Physics Loss", color='green')
        ax[1].set_title("Physics Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].grid(True)
        
        # Total Loss
        ax[2].plot(loss_dict['total'], label="Total Loss", color='blue')
        ax[2].set_title("Total Loss")
        ax[2].set_xlabel("Epoch")
        ax[2].grid(True)
        
        plt.tight_layout()
        plt.show()

    # 4. COMPARE FINAL (LAST EPOCH) TOTAL LOSSES IN A BAR CHART
    model_keys = []
    final_losses = []
    for (w, lr), loss_dict in losses_data.items():
        model_keys.append(f"w={w}, lr={lr}")
        final_losses.append(loss_dict['total'][-1])  # last epoch's total loss

    plt.figure(figsize=(8, 6))
    plt.bar(model_keys, final_losses, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Final Total Loss")
    plt.yscale("log")
    plt.title("Comparison of Final Total Loss Across Models")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    model_keys = []
    final_losses = []
    # Extract final losses from the dictionary
    for (w, lr), loss_dict in losses_data.items():
        model_keys.append(f"w={w}, lr={lr}")
        final_losses.append(loss_dict['total'][-1])  # last epoch's total loss

    # Filter out non-positive values for log-scale plotting
    valid_indices = [i for i, loss in enumerate(final_losses) if loss > 0]
    filtered_model_keys = [model_keys[i] for i in valid_indices]
    filtered_final_losses = [final_losses[i] for i in valid_indices]

    # Generate improved histogram with log scale
    plt.figure(figsize=(8, 6))

    # Create a bar chart with improved aesthetics
    bars = plt.bar(filtered_model_keys, filtered_final_losses, color='skyblue', edgecolor='black', alpha=0.8)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2e}',
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')

    # Improve labels and styling
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.ylabel("Final Total Loss [-]", fontsize=14)
    plt.yscale("log")  # Log scale for better contrast
    plt.grid(True, axis='y')

    # Improve layout and display the chart
    plt.tight_layout()
    plt.show()

    # 5. DECIDE THE BEST MODEL
    # For example, pick the model with the lowest final total loss
    if final_losses:
        best_index = np.argmin(final_losses)
        best_model_key = model_keys[best_index]
        best_loss_val = final_losses[best_index]
        print(f"Best model based on final total loss: {best_model_key}, Loss = {best_loss_val:.6f}")
    else:
        print("No models found to compare.")

if model_accuracy : 

    """
    Here the analysis regarding the performance of the two 
    best trained models can be found. The postprocessed 
    images represent the differences between the 
    actual integrated results from simulated runs and the 
    ones that are achieved from the PINN models

    """

    # ------------------ Functions ------------------

    def load_and_combine_run_data(run_path):

        """
        Loads the Dependent_Variables.dat and Numerical_States.dat for a run,
        computes the distance from the center of mass (r) from numerical states,
        and returns a combined DataFrame.
        
        Assumes:
        - Dependent_Variables.dat can be read with pandas.read_csv using whitespace as delimiter.
        - Numerical_States.dat has columns where indices 1,2,3 correspond to x, y, z.
        """
        dv_path = os.path.join(run_path, 'Dependent_Variables.dat')
        ns_path = os.path.join(run_path, 'Numerical_States.dat')
        
        try:
            dependent_vars = pd.read_csv(dv_path, sep='\s+', header=None)
            numerical_states = pd.read_csv(ns_path, sep='\s+', header=None)
        except Exception as e:
            print(f"Error loading data from {run_path}: {e}")
            return None

        # --- Compute distance from center of mass (r) ---
        try:
            x = numerical_states[1]
            y = numerical_states[2]
            z = numerical_states[3]
        except KeyError:
            print(f"Unexpected column indices in {ns_path}")
            return None
        r = np.sqrt(x**2 + y**2 + z**2)

        # --- Select relevant columns from dependent variables ---
        # (Update the list of column indices as needed.)
        relevant_columns = [0, 1, 4, 5, 6, 7, 8, 9, 12, 16, 18, 22, 23]
        try:
            relevant_dependent_vars = dependent_vars[relevant_columns].copy()
        except Exception as e:
            print(f"Error selecting columns from {dv_path}: {e}")
            return None

        # Assume that the file provides these basic columns (without r and heat_flux):
        basic_columns = [
            'time', 'mach', 'latitude', 'longitude', 'heading', 
            'flight_path', 'aoa', 'bank', 'density', 'CD', 'CL', 'airspeed', 'g_norm'
        ]
        try:
            relevant_dependent_vars.columns = basic_columns
        except Exception as e:
            print(f"Error assigning column names: {e}")
            return None

        relevant_dependent_vars.insert(1, 'r', r)

        heat_flux = dependent_vars[24].copy()
        g_load = dependent_vars[11].copy()

        # Insert the computed distance ('r') as the second column:
        relevant_dependent_vars['g-load'] = g_load
        relevant_dependent_vars['heat_flux'] = heat_flux

        # Reorder the columns to match DEPENDENT_VAR_NAMES.
        try:
            combined_df = relevant_dependent_vars[DEPENDENT_VAR_NAMES].copy()
        except Exception as e:
            print(f"Error reordering columns in {run_path}: {e}")
            return None

        return combined_df

    def select_random_start_index(df, min_start_time=200.0, max_start_time=1000.0):

        """
        Randomly select an index in the DataFrame where 'time' is within [min_start_time, max_start_time).
        Returns the index (integer) if available; otherwise returns None.

        """

        valid_rows = df[(df['time'] >= min_start_time) & (df['time'] < max_start_time)]
        if valid_rows.empty:
            return None
        return valid_rows.sample(1).index[0]

    def pinn_integrate(current_time, initial_state, duration, model, scalers, features_in, features_out):

        """
        Integrates the current state for a given duration using the PINN model.
        
        This function mimics your guidance routine by iteratively updating the model state 
        (every 0.5 seconds, for example) until the total integrated time reaches the given 
        duration. It uses the same scaling/inverse-scaling procedure as your guidance code.

        Parameters
        ----------
        current_time : float
            The current simulation time.
        initial_state : np.ndarray
            An array of shape (1, n_features) representing the current state.
            (For example: model_state in your guidance code.)
        duration : float
            The time interval (in seconds) to integrate forward.
        model : torch.nn.Module
            The PINN model.
        scalers : dict
            A dictionary mapping feature names (strings) to scaling objects
            (with .transform and .inverse_transform methods) for normalization.
        features_in : list of str
            List of names corresponding to the input features used by the model.
        features_out : list of str
            List of names corresponding to the output features from the model.
            
        Returns
        -------
        final_time : float
            The integration (propagation) time after integration.
        final_state : np.ndarray
            The final state as predicted by the model after the integration.
        """
        dt = 0.5  # integration time step (seconds)

        # Define useful quantities to compute the new model_state elements
        atmosphere = COESA76()
        semimajor = 6371000
        f = 1 / 298.257223563
        gamma = 1.4
        R = 287.05 
        J2 =  1.082626E-3
        mu_earth =  3.9859383624e14

        # Define the parameters that are required for heat flux computations
        HORUS_RN = 0.8 # m Nose Radius
        c_star = 1.1097E8 # Aerodynamic Characteristic
        Vc = 7.9054E3 # Circular Velocity 
        n = 0.5
        m = 3.15
        rho_0 = 1.225 # kg/m^3

        # HORUS parameters
        mass = 26029.0 # kg
        ref_area = 110 # m^2

        # Parameters for the angle of attack  
        #Define the maximum angle of attack
        max_AoA = np.deg2rad(40)
        final_AoA = np.deg2rad(11.5) # To be changed
        aoa_energy = np.array([-1600000.00, -200000.00])

        # Copy the initial state to the working variable.
        model_state = initial_state.copy()

        # Loop until the integrated time reaches the specified duration.
        while (model_state[0, 0] - current_time) < duration:
            # First, normalize each column of model_state using its corresponding scaler.
            input_normalized = np.zeros_like(model_state)
            for i, feat in enumerate(features_in):
                # Reshape the column vector so it can be transformed.
                input_normalized[:, i] = scalers[feat].transform(model_state[:, i].reshape(-1, 1)).ravel()
            
            # Create a PyTorch tensor from the normalized input.
            input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
            
            # Run the model prediction.
            with torch.no_grad():
                y_pred_normalized = model(input_tensor)
            
            # Move prediction to CPU and convert to numpy.
            y_pred_normalized = y_pred_normalized.cpu().numpy()
            
            # Inverse transform the outputs.
            y_pred_physical = np.zeros_like(y_pred_normalized)
            for i, feat in enumerate(features_out):
                y_pred_physical[:, i] = scalers[feat].inverse_transform(y_pred_normalized[:, i].reshape(-1, 1)).ravel()
            
            # --- Update the model state ---ù
            # First, increment the time by dt.
            model_state[0, 0] = model_state[0, 0] + dt

            # Compute useful quantities to complete the model state elements
            altitude = y_pred_physical[0,0] - semimajor * (1 - f*np.sin(y_pred_physical[0,1])**2)
            altitude_atmo = altitude * u.m
            density = atmosphere.density(altitude_atmo) 
            temperature = atmosphere.temperature(altitude_atmo)
            sound = math.sqrt(gamma*R*temperature.value)

            # Then, update other parts of model_state with the predicted values.
            model_state[0,1]  = y_pred_physical[0,0]
            model_state[0,3]  = y_pred_physical[0,1]
            model_state[0,4]  = y_pred_physical[0,2]
            model_state[0,5]  = y_pred_physical[0,3]
            model_state[0,6]  = y_pred_physical[0,4]
            model_state[0,10] = y_pred_physical[0,5]
            model_state[0,11] = y_pred_physical[0,6]
            model_state[0,12] = y_pred_physical[0,7]
            model_state[0,2] = y_pred_physical[0,7] / sound                                                                                 
            model_state[0,8] = model_state[0,8]                                                                                     
            model_state[0,9] = density.value 

            current_energy = (1/y_pred_physical[0,0]) - (y_pred_physical[0,7]**2/2)
            if current_energy <= aoa_energy[0]: 
                model_state[0,7] = max_AoA
            elif aoa_energy[0] < current_energy <= aoa_energy[1]:
                model_state[0,7] = ((final_AoA  - max_AoA)/(aoa_energy[1] - aoa_energy[0]))*(current_energy - aoa_energy[0]) + max_AoA
            else:
                model_state[0,7] = final_AoA

            gr = (mu_earth/y_pred_physical[0,0]**2)*(1 - 1.5*J2*((6371e3/y_pred_physical[0,0])**2)*(3*np.sin(y_pred_physical[0,1])**2 - 1))
            gphi = (mu_earth/y_pred_physical[0,0]**2)*(-3*((6371e3/y_pred_physical[0,0])**2)*np.sin(y_pred_physical[0,1])*np.cos(y_pred_physical[0,1])*J2)
            model_state[0,13] = math.sqrt(gr**2 + gphi**2)  

        # Now we compute the necessary quantities for the constraints
        L = 0.5*model_state[0,9]*model_state[0,12]**2*ref_area*model_state[0,11]
        D = 0.5*model_state[0,9]*model_state[0,12]**2*ref_area*model_state[0,10]
        W = mass * model_state[0,13]    
        q_dot = float((c_star*(1/((HORUS_RN)**n))*((model_state[0,9]/rho_0)**(1-n))*(model_state[0,12]/Vc)**m))

        g_load = math.sqrt(L**2 + D**2)/W
        heat_flux = q_dot / 1000
    
        # Once the integration loop is complete, return the final time and the final state.
        final_time = model_state[0, 0]
        final_state = model_state.copy()

        # model_state is shape (1, 14); create an array with the extra two values
        extra = np.array([[g_load, heat_flux]])
        # Concatenate along the second axis so that the final state becomes shape (1, 16)
        final_state = np.concatenate([model_state, extra], axis=1)

        return final_state

    def compute_error_metrics_signed(errors_array):

        """
        Given a 2D array (n_runs x n_variables) of signed percentage errors (i.e.,
        (Predicted - True)/True * 100) for each variable, compute:
            - Mean signed error (percentage),
            - RMSE (percentage),
            - Maximum error (percentage),
            - Minimum error (percentage).
        
        Returns a dictionary with these metrics.

        """

        mean_error = np.mean(errors_array, axis=0)  # signed mean error (in %)
        rmse = np.sqrt(np.mean(errors_array**2, axis=0))
        max_error = np.max(errors_array, axis=0)
        min_error = np.min(errors_array, axis=0)
        
        return {
            'Mean Error': mean_error,
            'RMSE': rmse,
            'Max Error': max_error,
            'Min Error': min_error
        }
    
    def safe_percentage_error(pred, true, tol=1e-12):

        """
        Compute the percentage error as 100 * (pred - true) / true, but if the true value
        is below tol in absolute magnitude, return np.nan to indicate that the percentage error
        is not reliable.
        
        """

        if np.abs(true) < tol:
            return np.nan
        else:
            return 100.0 * (pred - true) / true

    def plot_histogram(data, title, xlabel):

        """
        Plot histograms for the frequency regarding the error
        percentage of the constraints g-load and heat flux with 
        respect to the actual integrator used for the simulations 

        Parameters
        ----------
        data : np.ndarray
            Data to be plotted
        title : str
            Title of the plot
        xlabel : str
            Name to be assigned at the x-label of the plot

        """

        plt.figure()
        plt.hist(data, bins=50, alpha=0.7)
        plt.xlabel(xlabel, fontsize = 14)
        plt.ylabel("Frequency", fontsize = 14)
        plt.title(title)
        plt.show()

    def plot_error_ellipse_percent(
        error_data, 
        title="",
        x_label="G-load Error (%)", 
        y_label="Heat Flux Error (%)", 
        conf_level=0.95, 
        max_abs_percent=30.0
    ) :
        """
        Plot a 2D scatter of (x_error%, y_error%) along with a confidence ellipse.
        We assume error_data is already in PERCENTAGES.
        
        Outliers with |error| > max_abs_percent on either axis are excluded.

        Parameters
        ----------
        error_data : np.ndarray
            Shape (N, 2). Each row is [x_error%, y_error%].
        title : str
            Title for the plot.
        x_label : str
            Label for the X axis.
        y_label : str
            Label for the Y axis.
        conf_level : float
            Confidence level for the ellipse (e.g., 0.95 => ~95%).
        max_abs_percent : float
            Threshold for outlier removal in absolute percentage. Points with
            |error| > max_abs_percent on either axis are dropped.
        """

        # 1. Remove NaNs
        mask_nonan = ~np.isnan(error_data[:, 0]) & ~np.isnan(error_data[:, 1])
        data_no_nan = error_data[mask_nonan]

        # 2. Remove outliers: keep only points within ±max_abs_percent
        mask_inliers = (
            (np.abs(data_no_nan[:, 0]) <= max_abs_percent) &
            (np.abs(data_no_nan[:, 1]) <= max_abs_percent)
        )
        data = data_no_nan[mask_inliers]

        if data.shape[0] < 2:
            print("Not enough points after outlier removal to plot an ellipse.")
            return

        # 3. Compute mean (center) and covariance
        mean = np.mean(data, axis=0)  # [mean_x, mean_y]
        cov = np.cov(data, rowvar=False)  # shape (2, 2)

        # 4. Eigen-decomposition of covariance => ellipse axes + orientation
        evals, evecs = np.linalg.eigh(cov)
        # Sort eigenvalues descending
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:, idx]

        # 5. Confidence level => chi-square(2) ~ {0.90:4.605, 0.95:5.991, 0.99:9.210}
        chi2_dict = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210}
        chi2_val = chi2_dict.get(conf_level, 5.991)
        radius = np.sqrt(chi2_val)  

        # Width, height of ellipse => 2 * radius * sqrt(evals)
        width, height = 2 * radius * np.sqrt(evals)

        # Orientation angle in degrees
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))

        # 6. Plot
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(data[:, 0], data[:, 1], alpha=0.4, label='Data points')
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.axvline(0, color='gray', linewidth=0.8)

        # Ellipse patch
        ellipse = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            edgecolor='red', facecolor='none', linewidth=2,
            label=f'{int(conf_level*100)}% Conf. Ellipse'
        )
        ax.add_patch(ellipse)

        # Plot center (mean)
        ax.scatter(mean[0], mean[1], color='red', s=80, marker='x', label='Mean')

        ax.set_xlabel(x_label, fontsize = 16)
        ax.set_ylabel(y_label, fontsize = 16)
        ax.legend(fontsize = 14)
        plt.show()

    
    # Function to plot histogram with KDE line
    def plot_histogram_with_kde(data, bins, xlabel, title, color, alpha=0.6):
        plt.figure(figsize=(8, 6))  # Set figure size
        sns.histplot(data, bins=bins, kde=True, color=color, alpha=alpha, edgecolor="black", linewidth=0.8)
        
        # Labels and title
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel("Frequency", fontsize=16)
        
        # Adjust grid
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Show plot
        plt.show()


    def filtered_hist(
        data_array, 
        variable_name="Variable Error (%)", 
        model_label="Model",
        max_abs_err=200.0, 
        bins=50
    ):
        """
        Exclude outliers (abs(value) > max_abs_err) from data_array and plot a histogram.

        Parameters
        ----------
        data_array : np.ndarray
            1D array of error values (percentage).
        variable_name : str
            Label used for the x-axis.
        model_label : str
            A short descriptor for the title (e.g. 'Wider Model').
        max_abs_err : float
            Threshold for outlier removal in absolute value.
        bins : int
            Number of histogram bins.
        """
        # Remove NaNs
        mask_nonan = ~np.isnan(data_array)
        data_no_nan = data_array[mask_nonan]

        # Remove large outliers
        mask_inliers = np.abs(data_no_nan) <= max_abs_err
        data_inliers = data_no_nan[mask_inliers]

        plt.figure()
        plt.hist(data_inliers, bins=bins, alpha=0.7)
        plt.xlabel(variable_name)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {variable_name} ({model_label}, ±{max_abs_err}%)")
        plt.show()

    # ------------------ Configuration ------------------

    TIME_STEP = 0.5                 # seconds per simulation time step
    INTEGRATION_DURATION = 0.5     # seconds to integrate forward with the PINN
    MAX_START_TIME = 1100.0         # select start times less than 1100 s (so that t0+25 is within simulation)
    SAMPLE_SIZE_PER_CAT = 100       # number of runs to sample from each category

    # Achieve the scalers
    scalers_path = os.path.join(current_dir, "PINN_Scalers")
    features_in = [
        'time', 'r', 'mach', 'latitude', 'longitude', 
        'heading', 'flight_path', 'aoa', 'bank', 
        'density', 'CD', 'CL', 'airspeed', 'g_norm'
    ]
    features_out = [
        'r', 'latitude', 'longitude', 'heading', 
        'flight_path', 'CD', 'CL', 'airspeed'
    ]
    scalers = {}
    for f in features_in :
        scaler_file = os.path.join(scalers_path, f"{f}_scaler.pkl")
        scalers[f] = joblib.load(scaler_file)

    # Define directories (adjust as needed)
    CURRENT_DIR = os.path.dirname(__file__)
    BASE_DIR = os.path.join(CURRENT_DIR, "SimulationOutput", "NoWind", "PredCorr")
    VALID_CATEGORIES = ['Success', 'Fail']

    # --- Define the column names for the combined dependent variables ---
    DEPENDENT_VAR_NAMES = [
        'time', 'r', 'mach', 'latitude', 'longitude', 'heading',
        'flight_path', 'aoa', 'bank', 'density', 'CD', 'CL',
        'airspeed', 'g_norm', 'g-load', 'heat_flux'
    ]

    all_run_paths = []
    for category in VALID_CATEGORIES:
        cat_folder = os.path.join(BASE_DIR, category)
        if not os.path.exists(cat_folder):
            continue
        for run_folder in os.listdir(cat_folder):
            run_path = os.path.join(cat_folder, run_folder)
            if os.path.isdir(run_path):
                dv_path = os.path.join(run_path, 'Dependent_Variables.dat')
                ns_path = os.path.join(run_path, 'Numerical_States.dat')
                if os.path.exists(dv_path) and os.path.exists(ns_path):
                    all_run_paths.append(run_path)

    # Sample a subset of runs per category.
    sampled_run_paths = []
    for category in VALID_CATEGORIES:
        category_runs = [rp for rp in all_run_paths if category in rp]
        if len(category_runs) > SAMPLE_SIZE_PER_CAT:
            sampled_run_paths.extend(random.sample(category_runs, SAMPLE_SIZE_PER_CAT))
        else:
            sampled_run_paths.extend(category_runs)


    # Containers for error metrics (percentage error).
    # We accumulate signed percentage errors for each integrated dependent variable.
    tighter_model_signed_errors = []  # each element is an array of errors (one row per run)
    wider_model_signed_errors = []  # each element is an array of errors (one row per run)

    # Define the directories of the two models
    wider_dir = os.path.join( current_dir, "PINN_Model", "width_64", "lr_0.005", "model.pkl" )
    tighter_dir = os.path.join( current_dir, "Archive_Superseded", "PINN_Model_old", "width3rdModel_64", "lr_0.005", "model.pkl" )  # archived legacy model

    wider_model = joblib.load(wider_dir)
    tighter_model = joblib.load(tighter_dir)

    wider_model.eval()
    tighter_model.eval()

    for run_path in sampled_run_paths:
        combined_df = load_and_combine_run_data(run_path)
        if combined_df is None or combined_df.empty:
            continue

        # Select a random start index (ensuring t0 < MAX_START_TIME).
        start_idx = select_random_start_index(combined_df)
        if start_idx is None:
            continue

        start_row = combined_df.loc[start_idx]
        t0 = start_row['time']

        # Extract the initial condition for the PINN integration:
        # Use the values corresponding to features_in.
        initial_state = start_row[features_in].values.reshape(1, -1)

        # Define target final time (t0 + integration duration)
        target_time = t0 + INTEGRATION_DURATION
        # Find the row index where 'time' is closest to target_time.
        time_diffs = np.abs(combined_df['time'] - target_time)
        final_idx = time_diffs.idxmin()

        # Retrieve the final state (ground truth) from simulation data
        final_row = combined_df.loc[final_idx]

        # Define a tolerance for bank angle differences (adjust as needed)
        tolerance = 1e-3  # For example, if bank angles are in radians

        # Check if the bank angle at t0 is approximately equal to the bank angle at final time.
        if abs(start_row['bank'] - final_row['bank']) > tolerance:
            # If not, skip this run.
            continue


        # --- PINN integration ---
        pred_state_wide = pinn_integrate(t0, initial_state, INTEGRATION_DURATION, wider_model, scalers, features_in, features_out)
        pred_state_tight = pinn_integrate(t0, initial_state, INTEGRATION_DURATION, tighter_model, scalers, features_in, features_out)

        # --- Retrieve ground-truth final state (from the same run) ---
        true_dep = combined_df.loc[final_idx].values.astype(float).flatten()

        # Flatten the predicted states (they originally have shape (1, 16))
        pred_state_wide = np.array(pred_state_wide).flatten()
        pred_state_tight = np.array(pred_state_tight).flatten()

        # Compute the signed percentage error for each variable elementwise.
        percentage_errors_wide = np.array([safe_percentage_error(p, t) for p, t in zip(pred_state_wide, true_dep)])
        percentage_errors_tight = np.array([safe_percentage_error(p, t) for p, t in zip(pred_state_tight, true_dep)])

        wider_model_signed_errors.append(percentage_errors_wide)
        tighter_model_signed_errors.append(percentage_errors_tight)

    # Convert error lists to numpy arrays.
    wider_errors_array = np.array(wider_model_signed_errors)  # shape: (n_runs, n_variables)
    tighter_errors_array = np.array(tighter_model_signed_errors)

    # Compute error metrics.
    wider_metrics = compute_error_metrics_signed(wider_errors_array)
    tighter_metrics = compute_error_metrics_signed(tighter_errors_array)

    # Print summary of metrics (skipping 'time' and 'r' if desired).
    print("\nData-Heavy Model Error Metrics (Signed Percentage Errors):")
    for i, col in enumerate(DEPENDENT_VAR_NAMES):
        if col in ['time', 'aoa', 'bank']:
            continue
        print(f"  {col}: Mean = {wider_metrics['Mean Error'][i]:+.2f}%, RMSE = {wider_metrics['RMSE'][i]:.2f}%, Max = {wider_metrics['Max Error'][i]:+.2f}%, Min = {wider_metrics['Min Error'][i]:+.2f}%")

    print("\nBalanced Model Error Metrics (Signed Percentage Errors):")
    for i, col in enumerate(DEPENDENT_VAR_NAMES):
        if col in ['time', 'aoa', 'bank']:
            continue
        print(f"  {col}: Mean = {tighter_metrics['Mean Error'][i]:+.2f}%, RMSE = {tighter_metrics['RMSE'][i]:.2f}%, Max = {tighter_metrics['Max Error'][i]:+.2f}%, Min = {tighter_metrics['Min Error'][i]:+.2f}%")

    g_load_idx = DEPENDENT_VAR_NAMES.index('g-load')
    heat_flux_idx = DEPENDENT_VAR_NAMES.index('heat_flux')

    # For the wider model:
    plt.figure()
    plt.hist(wider_errors_array[:, g_load_idx][~np.isnan(wider_errors_array[:, g_load_idx])], bins=50, alpha=0.7)
    plt.xlabel("G-load Error (%)")
    plt.ylabel("Frequency")
    plt.title("Histogram of G-load Percentage Errors (Data-Heavy Model)")
    plt.show()

    plt.figure()
    plt.hist(wider_errors_array[:, heat_flux_idx][~np.isnan(wider_errors_array[:, heat_flux_idx])], bins=50, alpha=0.7)
    plt.xlabel("Heat Flux Error (%)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Heat Flux Percentage Errors (Data-Heavy Model)")
    plt.show()

    # For the tighter model:
    plt.figure()
    plt.hist(tighter_errors_array[:, g_load_idx][~np.isnan(tighter_errors_array[:, g_load_idx])], bins=50, alpha=0.7)
    plt.xlabel("G-load Error (%)")
    plt.ylabel("Frequency")
    plt.title("Histogram of G-load Percentage Errors (Balanced Model)")
    plt.show()

    plt.figure()
    plt.hist(tighter_errors_array[:, heat_flux_idx][~np.isnan(tighter_errors_array[:, heat_flux_idx])], bins=50, alpha=0.7)
    plt.xlabel("Heat Flux Error (%)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Heat Flux Percentage Errors (Balanced Model)")
    plt.show()

    num_bins = 50

    # Colors for different models
    color_64 = "royalblue"
    color_32 = "darkorange"

    # Plot histograms for the wider model (64 neurons)
    plot_histogram_with_kde(
        wider_errors_array[:, g_load_idx][~np.isnan(wider_errors_array[:, g_load_idx])], 
        bins=num_bins, xlabel="G-load Error (%)", 
        title="Histogram of G-load Percentage Errors (Data-Heavy Model)", 
        color=color_64
    )

    plot_histogram_with_kde(
        wider_errors_array[:, heat_flux_idx][~np.isnan(wider_errors_array[:, heat_flux_idx])], 
        bins=num_bins, xlabel="Heat Flux Error (%)", 
        title="Histogram of Heat Flux Percentage Errors (Data-Heavy Model)", 
        color=color_64
    )

    # Plot histograms for the tighter model (32 neurons)
    plot_histogram_with_kde(
        tighter_errors_array[:, g_load_idx][~np.isnan(tighter_errors_array[:, g_load_idx])], 
        bins=num_bins, xlabel="G-load Error (%)", 
        title="Histogram of G-load Percentage Errors (Balanced Model)", 
        color=color_32
    )

    plot_histogram_with_kde(
        tighter_errors_array[:, heat_flux_idx][~np.isnan(tighter_errors_array[:, heat_flux_idx])], 
        bins=num_bins, xlabel="Heat Flux Error (%)", 
        title="Histogram of Heat Flux Percentage Errors (Balanced Model)", 
        color=color_32
    )

    g_load_idx = DEPENDENT_VAR_NAMES.index('g-load')
    heat_flux_idx = DEPENDENT_VAR_NAMES.index('heat_flux')

    # WIDER model 2D
    wider_g_data = wider_errors_array[:, g_load_idx]
    wider_hf_data = wider_errors_array[:, heat_flux_idx]
    wider_2D = np.column_stack((wider_g_data, wider_hf_data))

    # TIGHTER model 2D
    tighter_g_data = tighter_errors_array[:, g_load_idx]
    tighter_hf_data = tighter_errors_array[:, heat_flux_idx]
    tighter_2D = np.column_stack((tighter_g_data, tighter_hf_data))

    # Plot for the WIDER model
    plot_error_ellipse_percent(
        error_data=wider_2D,
        title="G-load vs Heat Flux Error (Data-Heavy Model)",
        x_label="G-load Error (%)",
        y_label="Heat Flux Error (%)",
        conf_level=0.9,       # or 0.90, 0.99, etc.
        max_abs_percent=80.0  # exclude points >200% in absolute error
    )

    # Plot for the TIGHTER model
    plot_error_ellipse_percent(
        error_data=tighter_2D,
        title="G-load vs Heat Flux Error (Balanced Model)",
        x_label="G-load Error (%)",
        y_label="Heat Flux Error (%)",
        conf_level=0.9,
        max_abs_percent=70.0
    )

    # Plot G-load error histogram, removing outliers
    filtered_hist(
        data_array=wider_g_data,
        variable_name="G-load Error (%)",
        model_label="Data-Heavy Model",
        max_abs_err=70.0,  # or set your desired threshold
        bins=50
    )

    # Plot Heat Flux error histogram, removing outliers
    filtered_hist(
        data_array=wider_hf_data,
        variable_name="Heat Flux Error (%)",
        model_label="Data-Heavy Model",
        max_abs_err=70.0,
        bins=50
    )


    filtered_hist(
        data_array=tighter_g_data,
        variable_name="G-load Error (%)",
        model_label="Balanced Model",
        max_abs_err=70.0
    )

    filtered_hist(
        data_array=tighter_hf_data,
        variable_name="Heat Flux Error (%)",
        model_label="Balanced Model",
        max_abs_err=70.0
    )

if radar_chart : 
    MAX_G_LOAD     = 3.0         # [g]
    MAX_HEAT_FLUX  = 530.0       # [kW/m^2]
    FINAL_MACH     = 3.0         # [-]
    FINAL_ALTITUDE = 30000.0     # [m]
    FINAL_ANG_DIST = 1.0         # [deg]

    # -------------------------
    # Set up Base Directory
    # -------------------------
    if wind:
        first_dir = os.path.join(current_dir, "SimulationOutput", "Wind")
    else:
        first_dir = os.path.join(current_dir, "SimulationOutput", "NoWind")

    if model == "Predictor-Corrector":
        base_dir = os.path.join(first_dir, "PredCorr")
    else:
        base_dir = os.path.join(first_dir, model)

    valid_categories = ['Success', 'Acceptable', 'Fail']

    # -------------------------
    # Initialize lists for each category (storing ratios)
    # -------------------------
    mach_ratio_success, alt_ratio_success = [], []
    gload_ratio_success, flux_ratio_success = [], []
    ang_ratio_success = []

    mach_ratio_fail, alt_ratio_fail = [], []
    gload_ratio_fail, flux_ratio_fail = [], []
    ang_ratio_fail = []

    mach_ratio_accept, alt_ratio_accept = [], []
    gload_ratio_accept, flux_ratio_accept = [], []
    ang_ratio_accept = []

    # -------------------------
    # Classification function
    # -------------------------
    def classify_run(mach_val, alt_val, g_val, heat_val, ang_val):
        # Fail if g-load or heat flux exceed limits
        if g_val >= MAX_G_LOAD or heat_val >= MAX_HEAT_FLUX:
            return 'Fail'
        # Success if all terminal conditions are within limits
        if (mach_val < FINAL_MACH and alt_val < FINAL_ALTITUDE and ang_val < FINAL_ANG_DIST):
            return 'Success'
        # Otherwise Acceptable
        return 'Acceptable'

    # -------------------------
    # Collect data from Results.json
    # -------------------------
    for category in valid_categories:
        category_dir = os.path.join(base_dir, category)
        if os.path.exists(category_dir):
            for root, dirs, files in os.walk(category_dir):
                if 'Results.json' in files:
                    results_file = os.path.join(root, 'Results.json')
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                    # Extract raw metrics
                    mach_val = data.get("Final Mach Number [-]", None)
                    alt_val  = data.get("Final Altitude [m]", None)
                    g_val    = data.get("Maximum G-Load [g]", None)
                    heat_val = data.get("Maximum Heat Flux [kW/m^2]", None)
                    ang_val  = data.get("Final Angular Distance [deg]", None)

                    # Skip if any value is missing
                    if None in [mach_val, alt_val, g_val, heat_val, ang_val]:
                        continue

                    # Classify
                    final_category = classify_run(mach_val, alt_val, g_val, heat_val, ang_val)

                    # Compute ratios
                    mach_ratio = mach_val / FINAL_MACH
                    alt_ratio  = alt_val / FINAL_ALTITUDE
                    gload_ratio = g_val / MAX_G_LOAD
                    flux_ratio  = heat_val / MAX_HEAT_FLUX
                    ang_ratio   = ang_val / FINAL_ANG_DIST

                    # Store ratios in the correct list
                    if final_category == 'Success':
                        mach_ratio_success.append(mach_ratio)
                        alt_ratio_success.append(alt_ratio)
                        gload_ratio_success.append(gload_ratio)
                        flux_ratio_success.append(flux_ratio)
                        ang_ratio_success.append(ang_ratio)
                    elif final_category == 'Fail':
                        mach_ratio_fail.append(mach_ratio)
                        alt_ratio_fail.append(alt_ratio)
                        gload_ratio_fail.append(gload_ratio)
                        flux_ratio_fail.append(flux_ratio)
                        ang_ratio_fail.append(ang_ratio)
                    else:  # Acceptable
                        mach_ratio_accept.append(mach_ratio)
                        alt_ratio_accept.append(alt_ratio)
                        gload_ratio_accept.append(gload_ratio)
                        flux_ratio_accept.append(flux_ratio)
                        ang_ratio_accept.append(ang_ratio)

    # Convert lists to NumPy arrays
    mach_ratio_success = np.array(mach_ratio_success)
    alt_ratio_success = np.array(alt_ratio_success)
    gload_ratio_success = np.array(gload_ratio_success)
    flux_ratio_success = np.array(flux_ratio_success)
    ang_ratio_success = np.array(ang_ratio_success)

    mach_ratio_fail = np.array(mach_ratio_fail)
    alt_ratio_fail = np.array(alt_ratio_fail)
    gload_ratio_fail = np.array(gload_ratio_fail)
    flux_ratio_fail = np.array(flux_ratio_fail)
    ang_ratio_fail = np.array(ang_ratio_fail)

    mach_ratio_accept = np.array(mach_ratio_accept)
    alt_ratio_accept = np.array(alt_ratio_accept)
    gload_ratio_accept = np.array(gload_ratio_accept)
    flux_ratio_accept = np.array(flux_ratio_accept)
    ang_ratio_accept = np.array(ang_ratio_accept)

    # -------------------------
    # Compute Aggregated (Mean) Ratios
    # -------------------------
    def safe_mean(arr):
        return np.mean(arr) if arr.size > 0 else 0

    success_means = np.array([
        safe_mean(mach_ratio_success),
        safe_mean(alt_ratio_success),
        safe_mean(gload_ratio_success),
        safe_mean(flux_ratio_success),
        safe_mean(ang_ratio_success)
    ])

    fail_means = np.array([
        safe_mean(mach_ratio_fail),
        safe_mean(alt_ratio_fail),
        safe_mean(gload_ratio_fail),
        safe_mean(flux_ratio_fail),
        safe_mean(ang_ratio_fail)
    ])

    accept_means = np.array([
        safe_mean(mach_ratio_accept),
        safe_mean(alt_ratio_accept),
        safe_mean(gload_ratio_accept),
        safe_mean(flux_ratio_accept),
        safe_mean(ang_ratio_accept)
    ])

    # -------------------------
    # Radar Plot with Ratios
    # -------------------------
    # Each metric is dimensionless here, but let's annotate them with their real meaning.

    metrics_names = [
        'Mach Ratio ', 
        'Altitude Ratio ', 
        'G-Load Ratio ', 
        'Heat Flux Ratio ', 
        'Ang. Dist Ratio '
    ]
    N = len(metrics_names)

    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    success_means = np.concatenate((success_means, [success_means[0]]))
    fail_means = np.concatenate((fail_means, [fail_means[0]]))
    accept_means = np.concatenate((accept_means, [accept_means[0]]))

    # Create a larger figure to accommodate legend and labels
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)

    # Build the plot title
    title_str = "Radar Chart of Normalized (Ratio) Re-Entry Metrics"
    if wind:
        title_str += " (Wind)"
    else:
        title_str += " (No Wind)"

    # Plot each category if data is available
    if mach_ratio_success.size > 0:
        ax.plot(angles, success_means, label='Success', marker='o', color='green')
        ax.fill(angles, success_means, alpha=0.15, color='green')
    if mach_ratio_accept.size > 0:
        ax.plot(angles, accept_means, label='Acceptable', marker='o', color='orange')
        ax.fill(angles, accept_means, alpha=0.15, color='orange')
    if mach_ratio_fail.size > 0:
        ax.plot(angles, fail_means, label='Fail', marker='o', color='red')
        ax.fill(angles, fail_means, alpha=0.15, color='red')

    # Set the category labels around the circle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=12)

    # Shift radial labels outward so they don't overlap polygons
    ax.tick_params(axis='x', pad=20)

    # Set radial range: from 0 to 2 (200% of limit, can adjust if needed)
    ax.set_ylim(0, 3.0)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(['50%', '100%', '150%', '200%', '250%', '300%'], fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Position radial labels
    ax.set_rlabel_position(35.5)

    # Add grid lines for clarity
    ax.grid(True, linestyle='--', alpha=0.7)

    # Place the legend above the plot, spanning multiple columns if needed
    plt.legend(loc='center left', bbox_to_anchor=(-0.4, 0.5), fontsize = 13)

    #bbox_to_anchor=(-0.35, 1.14),

    plt.show()

if violin_plot : 
    MAX_G_LOAD     = 3.0         # [g]
    MAX_HEAT_FLUX  = 530.0       # [kW/m^2]
    FINAL_MACH     = 3.0         # [-]
    FINAL_ALTITUDE = 30000.0     # [m]
    FINAL_ANG_DIST = 1.0         # [deg]

    # -------------------------
    # Set up Base Directory
    # -------------------------
    if wind:
        first_dir = os.path.join(current_dir, "SimulationOutput", "Wind")
    else:
        first_dir = os.path.join(current_dir, "SimulationOutput", "NoWind")

    if model == "Predictor-Corrector":
        valid_categories = ['Success', 'Acceptable', 'Fail']
        base_dir = os.path.join(first_dir, "PredCorr")
    else:
        valid_categories = ['Success', 'Acceptable', 'Fail']
        base_dir = os.path.join(first_dir, model)

    # -------------------------
    # Initialize lists for each metric per category
    # -------------------------
    # We'll store the raw values as provided in Results.json
    final_mach_success, final_alt_success = [], []
    gload_success, heatflux_success = [], []
    ang_success = []

    final_mach_fail, final_alt_fail = [], []
    gload_fail, heatflux_fail = [], []
    ang_fail = []

    final_mach_accept, final_alt_accept = [], []
    gload_accept, heatflux_accept = [], []
    ang_accept = []

    # -------------------------
    # Classification function
    # -------------------------
    def classify_run(mach_val, alt_val, g_val, heat_val, ang_val):
        # Fail: if either maximum G‑Load or heat flux exceed limits
        if g_val >= MAX_G_LOAD or heat_val >= MAX_HEAT_FLUX:
            return 'Fail'
        # Success: if all termination conditions are within limits
        if (mach_val < FINAL_MACH and alt_val < FINAL_ALTITUDE and ang_val < FINAL_ANG_DIST):
            return 'Success'
        # Otherwise, Acceptable
        return 'Acceptable'

    # -------------------------
    # Walk through directories and collect data
    # -------------------------
    for category in valid_categories:
        category_dir = os.path.join(base_dir, category)
        if os.path.exists(category_dir):
            for root, dirs, files in os.walk(category_dir):
                if 'Results.json' in files:
                    results_file = os.path.join(root, 'Results.json')
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                    # Extract metrics from Results.json
                    mach_val = data.get("Final Mach Number [-]", None)
                    alt_val  = data.get("Final Altitude [m]", None)
                    g_val    = data.get("Maximum G-Load [g]", None)
                    heat_val = data.get("Maximum Heat Flux [kW/m^2]", None)
                    ang_val  = data.get("Final Angular Distance [deg]", None)
                    # Skip run if any value is missing
                    if None in [mach_val, alt_val, g_val, heat_val, ang_val]:
                        continue

                    # Classify the run
                    final_category = classify_run(mach_val, alt_val, g_val, heat_val, ang_val)

                    # Append values to corresponding lists
                    if final_category == 'Success':
                        final_mach_success.append(mach_val)
                        final_alt_success.append(alt_val)
                        gload_success.append(g_val)
                        heatflux_success.append(heat_val)
                        ang_success.append(ang_val)
                    elif final_category == 'Fail':
                        final_mach_fail.append(mach_val)
                        final_alt_fail.append(alt_val)
                        gload_fail.append(g_val)
                        heatflux_fail.append(heat_val)
                        ang_fail.append(ang_val)
                    else:  # Acceptable
                        final_mach_accept.append(mach_val)
                        final_alt_accept.append(alt_val)
                        gload_accept.append(g_val)
                        heatflux_accept.append(heat_val)
                        ang_accept.append(ang_val)

    # Convert lists to NumPy arrays
    final_mach_success = np.array(final_mach_success)
    final_alt_success = np.array(final_alt_success)
    gload_success = np.array(gload_success)
    heatflux_success = np.array(heatflux_success)
    ang_success = np.array(ang_success)

    final_mach_fail = np.array(final_mach_fail)
    final_alt_fail = np.array(final_alt_fail)
    gload_fail = np.array(gload_fail)
    heatflux_fail = np.array(heatflux_fail)
    ang_fail = np.array(ang_fail)

    final_mach_accept = np.array(final_mach_accept)
    final_alt_accept = np.array(final_alt_accept)
    gload_accept = np.array(gload_accept)
    heatflux_accept = np.array(heatflux_accept)
    ang_accept = np.array(ang_accept)

    # -------------------------
    # Create Violin Plots for Each Metric
    # -------------------------
    # We produce a subplot for each of the 5 metrics.
    # Note: For altitude, we convert from meters to kilometers.
    fig, axs = plt.subplots(1, 5, figsize=(10, 5))

    # Define categories (order matters)
    cat_labels = ['S', 'A', 'F']

    # Prepare data for each metric:
    data_mach = [final_mach_success, final_mach_accept, final_mach_fail]
    data_alt = [final_alt_success / 1000.0, final_alt_accept / 1000.0, final_alt_fail / 1000.0]  # in km
    data_gload = [gload_success, gload_accept, gload_fail]
    data_heat = [heatflux_success, heatflux_accept, heatflux_fail]
    data_ang = [ang_success, ang_accept, ang_fail]

    # Plot Final Mach Number
    axs[0].violinplot(data_mach, showmeans=True, widths=0.8)
    axs[0].set_title('Final Mach [-]', fontsize=12)
    axs[0].set_xticks([1, 2, 3])
    axs[0].set_xticklabels(cat_labels, fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Plot Final Altitude (km)
    axs[1].violinplot(data_alt, showmeans=True, widths=0.8)
    axs[1].set_title('Final Altitude [km]', fontsize=12)
    axs[1].set_xticks([1, 2, 3])
    axs[1].set_xticklabels(cat_labels, fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # Plot Maximum G-Load
    axs[2].violinplot(data_gload, showmeans=True, widths=0.8)
    axs[2].set_title('Max G-Load [g]', fontsize=12)
    axs[2].set_xticks([1, 2, 3])
    axs[2].set_xticklabels(cat_labels, fontsize=10)
    axs[2].grid(True, linestyle='--', alpha=0.5)

    # Plot Maximum Heat Flux
    axs[3].violinplot(data_heat, showmeans=True, widths=0.8)
    axs[3].set_title('Max Heat Flux [kW/m²]', fontsize=12)
    axs[3].set_xticks([1, 2, 3])
    axs[3].set_xticklabels(cat_labels, fontsize=10)
    axs[3].grid(True, linestyle='--', alpha=0.5)

    # Plot Final Angular Distance
    axs[4].violinplot(data_ang, showmeans=True, widths=0.8)
    axs[4].set_title('Final Distance [deg]', fontsize=12)
    axs[4].set_xticks([1, 2, 3])
    axs[4].set_xticklabels(cat_labels, fontsize=10)
    axs[4].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

if analysis:
    # Constraint limits
    MAX_G_LOAD     = 3.0         # [g]
    MAX_HEAT_FLUX  = 530.0       # [kW/m^2]
    FINAL_MACH     = 3.0         # [-]
    FINAL_ALTITUDE = 30000.0     # [m]
    FINAL_ANG_DIST = 1.0         # [deg]

    # ---------------------------
    # Set up Base Directory
    # ---------------------------
    if wind:
        first_dir = os.path.join(current_dir, "SimulationOutput", "Wind")
    else:
        first_dir = os.path.join(current_dir, "SimulationOutput", "NoWind")

    if model == "Predictor-Corrector":
        base_dir = os.path.join(first_dir, "PredCorr")
    else:
        base_dir = os.path.join(first_dir, model)

    valid_categories = ['Success', 'Acceptable', 'Fail']

    # ---------------------------
    # Initialize counters and lists for statistics
    # ---------------------------
    total_success = 0
    total_acceptable = 0
    total_fail = 0

    prop_times_success = []
    prop_times_acceptable = []
    prop_times_fail = []
    prop_times_all = []

    fail_due_to_gload = 0
    fail_due_to_heatflux = 0
    fail_due_to_both = 0

    # Lists to collect constraint values for overall statistics
    g_load_all = []
    heat_flux_all = []

    # ---------------------------
    # Classification Function
    # ---------------------------
    def classify_run(mach_val, alt_val, g_val, heat_val, ang_val):
        # Fail if either maximum G‑Load or heat flux exceed limits
        if g_val >= MAX_G_LOAD or heat_val >= MAX_HEAT_FLUX:
            return 'Fail'
        # Success if all terminal conditions are within limits
        if (mach_val < FINAL_MACH and alt_val < FINAL_ALTITUDE and ang_val < FINAL_ANG_DIST):
            return 'Success'
        # Otherwise, Acceptable
        return 'Acceptable'

    # ---------------------------
    # Walk through directories and gather data from Results.json
    # ---------------------------
    for category in valid_categories:
        category_dir = os.path.join(base_dir, category)
        if os.path.exists(category_dir):
            for root, dirs, files in os.walk(category_dir):
                if 'Results.json' in files:
                    results_file = os.path.join(root, 'Results.json')
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                    # Extract metrics
                    prop_time = data.get("Propagation Time [s]", None)
                    mach_val = data.get("Final Mach Number [-]", None)
                    alt_val  = data.get("Final Altitude [m]", None)
                    g_val    = data.get("Maximum G-Load [g]", None)
                    heat_val = data.get("Maximum Heat Flux [kW/m^2]", None)
                    ang_val  = data.get("Final Angular Distance [deg]", None)

                    # Skip run if any value is missing
                    if None in [prop_time, mach_val, alt_val, g_val, heat_val, ang_val]:
                        continue

                    # Add propagation time to overall list
                    prop_times_all.append(prop_time)

                    # Also accumulate constraint values
                    g_load_all.append(g_val)
                    heat_flux_all.append(heat_val)

                    # Classify the run
                    classification = classify_run(mach_val, alt_val, g_val, heat_val, ang_val)
                    if classification == 'Success':
                        total_success += 1
                        prop_times_success.append(prop_time)
                    elif classification == 'Fail':
                        total_fail += 1
                        prop_times_fail.append(prop_time)
                        # Count failure reasons
                        g_fail = (g_val >= MAX_G_LOAD)
                        heat_fail = (heat_val >= MAX_HEAT_FLUX)
                        if g_fail and heat_fail:
                            fail_due_to_both += 1
                        elif g_fail:
                            fail_due_to_gload += 1
                        elif heat_fail:
                            fail_due_to_heatflux += 1
                    else:
                        total_acceptable += 1
                        prop_times_acceptable.append(prop_time)

    # ---------------------------
    # Convert propagation times and constraints to numpy arrays for statistics
    # ---------------------------
    prop_times_all = np.array(prop_times_all)
    prop_times_success = np.array(prop_times_success)
    prop_times_acceptable = np.array(prop_times_acceptable)
    prop_times_fail = np.array(prop_times_fail)
    g_load_all = np.array(g_load_all)
    heat_flux_all = np.array(heat_flux_all)

    # ---------------------------
    # Print Statistical Insights for Propagation Times and Category Counts
    # ---------------------------

    total_runs = total_success + total_acceptable + total_fail
    print("Statistical Insights on Re-Entry Runs:")
    print("--------------------------------------")
    print(f"Total Runs: {total_runs}")
    print(f"  Success: {total_success} runs")
    print(f"  Acceptable: {total_acceptable} runs")
    print(f"  Fail: {total_fail} runs\n")

    if total_fail > 0:
        print("Among Failures:")
        print(f"  Fail due to G-Load only: {fail_due_to_gload}")
        print(f"  Fail due to Heat Flux only: {fail_due_to_heatflux}")
        print(f"  Fail due to both G-Load and Heat Flux: {fail_due_to_both}\n")
    else:
        print("No Failures.\n")

    if prop_times_all.size > 0:
        print("Propagation Time [s] Statistics (Overall):")
        print(f"  Mean: {np.mean(prop_times_all):.2f} s")
        print(f"  Min: {np.min(prop_times_all):.2f} s")
        print(f"  Max: {np.max(prop_times_all):.2f} s\n")

    if prop_times_success.size > 0:
        print("Propagation Time [s] for Success Runs:")
        print(f"  Mean: {np.mean(prop_times_success):.2f} s, Min: {np.min(prop_times_success):.2f} s, Max: {np.max(prop_times_success):.2f} s")
    if prop_times_acceptable.size > 0:
        print("Propagation Time [s] for Acceptable Runs:")
        print(f"  Mean: {np.mean(prop_times_acceptable):.2f} s, Min: {np.min(prop_times_acceptable):.2f} s, Max: {np.max(prop_times_acceptable):.2f} s")
    if prop_times_fail.size > 0:
        print("Propagation Time [s] for Fail Runs:")
        print(f"  Mean: {np.mean(prop_times_fail):.2f} s, Min: {np.min(prop_times_fail):.2f} s, Max: {np.max(prop_times_fail):.2f} s")
    
    # ---------------------------
    # Print Mean Values for the Constraints (Overall)
    # ---------------------------
    if g_load_all.size > 0 and heat_flux_all.size > 0:
        print("\nConstraint Averages (Overall):")
        print(f"  Mean Maximum G-Load: {np.mean(g_load_all):.2f} g")
        print(f"  Mean Maximum Heat Flux: {np.mean(heat_flux_all):.2f} kW/m^2")
    
    # ---------------------------
    # Analysis on Cycle Times from Train.json Files
    # ---------------------------
    average_cycle_times = []

    # Loop over valid categories to gather cycle time information from Train.json
    for category in valid_categories:
        category_dir = os.path.join(base_dir, category)
        if os.path.exists(category_dir):
            for root, dirs, files in os.walk(category_dir):
                if 'Train.json' in files:
                    print(root)
                    train_file = os.path.join(root, 'Train.json')
                    with open(train_file, 'r') as f:
                        train_data = json.load(f)
                    # Extract cycle times from each epoch in the train file
                    cycle_times = [entry["cycle time [s]"] 
                                   for key, entry in train_data.items() 
                                   if isinstance(entry, dict) and "cycle time [s]" in entry]
                    if cycle_times:
                        avg_cycle = np.mean(cycle_times)
                        average_cycle_times.append(avg_cycle)
    
    average_cycle_times = np.array(average_cycle_times)
    if average_cycle_times.size > 0:
        mean_cycle = np.mean(average_cycle_times)
        rmse_cycle = np.sqrt(np.mean((average_cycle_times - mean_cycle)**2))
        print("\nCycle Time [s] Statistics:")
        print(f"  Mean Cycle Time: {mean_cycle:.2f} s")
        print(f"  RMSE of Cycle Time: {rmse_cycle:.2f} s")
    else:
        print("\nNo Cycle Time data available.")


if wind_analysis : 
    # ==============================
    # Parameters and Constraint Limits
    # ==============================
    # (Adjust these if needed)
    MAX_G_LOAD     = 3.0         # [g]
    MAX_HEAT_FLUX  = 530.0       # [kW/m^2]
    FINAL_MACH     = 3.0         # [-]
    FINAL_ALTITUDE = 30000.0     # [m]
    FINAL_ANG_DIST = 1.0         # [deg]

    # ==============================
    # Function to Collect Simulation Data
    # ==============================
    def collect_simulation_data(wind, model):
        """
        Collects key metrics from simulation runs (Results.json)
        for a given environmental condition (wind or no wind)
        and returns a dictionary with arrays of:
        - propagation times,
        - final Mach, final altitude, max G-Load, max Heat Flux,
        - final angular distance,
        - outcome counts (Success, Acceptable, Fail)
        """
        # Set base directory depending on wind flag
        current_dir = os.getcwd()  # or set explicitly
        if wind:
            first_dir = os.path.join(current_dir, "SimulationOutput", "Wind")
        else:
            first_dir = os.path.join(current_dir, "SimulationOutput", "NoWind")
        
        # For simplicity, assume model determines valid categories
        if model == "Predictor-Corrector":
            base_dir = os.path.join(first_dir, "PredCorr")
        else:
            base_dir = os.path.join(first_dir, model)

        valid_categories = ['Success', 'Acceptable', 'Fail']
        
        # Initialize lists to store data
        prop_times = []
        final_mach_list = []
        final_alt_list = []
        g_load_list = []
        heat_flux_list = []
        final_ang_list = []
        outcomes = {'Success': 0, 'Acceptable': 0, 'Fail': 0}
        
        # Walk through directories and read each Results.json
        for category in valid_categories:
            category_dir = os.path.join(base_dir, category)
            if os.path.exists(category_dir):
                for root, dirs, files in os.walk(category_dir):
                    if 'Results.json' in files:
                        results_file = os.path.join(root, 'Results.json')
                        with open(results_file, 'r') as f:
                            data = json.load(f)
                        # Extract metrics
                        prop_time = data.get("Propagation Time [s]", None)
                        mach_val = data.get("Final Mach Number [-]", None)
                        alt_val  = data.get("Final Altitude [m]", None)
                        g_val    = data.get("Maximum G-Load [g]", None)
                        heat_val = data.get("Maximum Heat Flux [kW/m^2]", None)
                        ang_val  = data.get("Final Angular Distance [deg]", None)
                        
                        # Skip run if any value is missing
                        if None in [prop_time, mach_val, alt_val, g_val, heat_val, ang_val]:
                            continue
                        
                        # Append data to lists
                        prop_times.append(prop_time)
                        final_mach_list.append(mach_val)
                        final_alt_list.append(alt_val)
                        g_load_list.append(g_val)
                        heat_flux_list.append(heat_val)
                        final_ang_list.append(ang_val)
                        
                        # Classify the run:
                        # Failed if either mission constraint is exceeded
                        if g_val >= MAX_G_LOAD or heat_val >= MAX_HEAT_FLUX:
                            classification = 'Fail'
                        # Success if all terminal conditions are met
                        elif mach_val < FINAL_MACH and alt_val < FINAL_ALTITUDE and ang_val < FINAL_ANG_DIST:
                            classification = 'Success'
                        else:
                            classification = 'Acceptable'
                        
                        outcomes[classification] += 1
        return {
            'outcomes': outcomes,
            'prop_times': np.array(prop_times),
            'final_mach': np.array(final_mach_list),
            'final_alt': np.array(final_alt_list),  # in m
            'g_load': np.array(g_load_list),
            'heat_flux': np.array(heat_flux_list),
            'final_ang': np.array(final_ang_list)
        }

    # ==============================
    # Collect Data for Wind and No-Wind Cases
    # ==============================
    data_wind = collect_simulation_data(wind=True, model=model)
    data_no_wind = collect_simulation_data(wind=False, model=model)

    # ==============================
    # Plot 1: Grouped Bar Chart for Outcome Counts
    # ==============================
    categories = ['Success', 'Acceptable', 'Fail']
    counts_wind = [data_wind['outcomes'].get(cat, 0) for cat in categories]
    counts_no_wind = [data_no_wind['outcomes'].get(cat, 0) for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,6))
    rects1 = ax.bar(x - width/2, counts_wind, width, label='Wind', color='skyblue')
    rects2 = ax.bar(x + width/2, counts_no_wind, width, label='No Wind', color='salmon')

    ax.set_ylabel('Number of Simulations', fontsize = 14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize = 14)
    ax.legend(fontsize = 16)

    plt.tight_layout()
    plt.grid()
    plt.show()

    # ==============================
    # Plot 2: Violin Plots for Key Metrics (Example: Maximum Heat Flux)
    # ==============================
    df_heat = pd.DataFrame({
        'Maximum Heat Flux (kW/m$^2$)': np.concatenate([data_wind['heat_flux'], data_no_wind['heat_flux']]),
        'Environment': ['Wind'] * len(data_wind['heat_flux']) + ['No Wind'] * len(data_no_wind['heat_flux'])
    })

    df_gload = pd.DataFrame({
        'Maximum G-Load (-)': np.concatenate([data_wind['g_load'], data_no_wind['g_load']]),
        'Environment': ['Wind'] * len(data_wind['g_load']) + ['No Wind'] * len(data_no_wind['g_load'])
    })

    plt.figure(figsize=(8,6))
    ax = sns.violinplot(x='Environment', 
                        y='Maximum Heat Flux (kW/m$^2$)', 
                        data=df_heat, 
                        hue='Environment', 
                        palette="Set2")
    ax.legend(loc = 'best', fontsize = 16)
    ax.set_xlabel("Environment", fontsize=16)  # X-axis label size
    ax.set_ylabel("Maximum Heat Flux (kW/m$^2$)", fontsize=16)  # Y-axis label size 
    plt.grid()
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8,6))
    ax = sns.violinplot(x='Environment', 
                        y='Maximum G-Load (-)', 
                        data=df_gload, 
                        hue='Environment', 
                        palette="Set2")
    ax.legend(loc = 'best', fontsize = 16) 
    ax.set_xlabel("Environment", fontsize=16)  # X-axis label size
    ax.set_ylabel("Maximum G-Load (-)", fontsize=16)  # Y-axis label size
    plt.grid()
    plt.tight_layout()
    plt.show()

    # (Similarly, you could produce violin plots for Maximum G-Load, Final Mach, Final Altitude, Final Angular Distance)

    # ==============================
    # Plot 3: Radar Chart for Aggregated (Normalized) Metrics
    # ==============================
    # Compute aggregated (mean) values for each environment
    def compute_aggregated_metrics(data):
        return np.array([
            np.mean(data['final_mach']),
            np.mean(data['final_alt']) / 1000.0,  # convert altitude to km
            np.mean(data['g_load']),
            np.mean(data['heat_flux']),
            np.mean(data['final_ang'])
        ])

    wind_means = compute_aggregated_metrics(data_wind)
    no_wind_means = compute_aggregated_metrics(data_no_wind)

    # Normalize each metric with respect to its constraint
    wind_norm = np.array([
        wind_means[0] / FINAL_MACH,
        wind_means[1] / (FINAL_ALTITUDE / 1000.0),  # altitude normalized by 30 km
        wind_means[2] / MAX_G_LOAD,
        wind_means[3] / MAX_HEAT_FLUX,
        wind_means[4] / FINAL_ANG_DIST
    ])
    no_wind_norm = np.array([
        no_wind_means[0] / FINAL_MACH,
        no_wind_means[1] / (FINAL_ALTITUDE / 1000.0),
        no_wind_means[2] / MAX_G_LOAD,
        no_wind_means[3] / MAX_HEAT_FLUX,
        no_wind_means[4] / FINAL_ANG_DIST
    ])

    metrics_names = [
        'Final Mach\n(M/3)', 
        'Final Alt (km)\n(h/30)', 
        'Max G-Load\n(g/3)', 
        'Max Heat Flux\n(HF/530)', 
        'Final Ang Dist\n(deg/1)'
    ]
    N = len(metrics_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    wind_norm = np.concatenate((wind_norm, [wind_norm[0]]))
    no_wind_norm = np.concatenate((no_wind_norm, [no_wind_norm[0]]))

    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    ax.set_title("Normalized Aggregated Metrics: Wind vs No Wind", fontsize=14, fontweight='bold', y=1.1)

    ax.plot(angles, wind_norm, label="Wind", marker='o', color='blue')
    ax.fill(angles, wind_norm, alpha=0.2, color='blue')
    ax.plot(angles, no_wind_norm, label="No Wind", marker='o', color='red')
    ax.fill(angles, no_wind_norm, alpha=0.2, color='red')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.tick_params(axis='x', pad=15)  # shift labels outward

    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(['50%', '100%', '150%', '200%'], fontsize=10, color='gray')
    ax.set_ylim(0, 2.0)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.show()

    # ==============================
    # Plot 4: Scatter Matrix (Pairplot) Comparing Key Parameters
    # ==============================
    # Combine data from wind and no wind cases into one DataFrame
    df_all = pd.DataFrame({
        'Final $Ma$ [-]': np.concatenate([data_wind['final_mach'], data_no_wind['final_mach']]),
        'Final $h$ [km]': np.concatenate([data_wind['final_alt'], data_no_wind['final_alt']]) / 1000.0,
        'Max $n$ [-]': np.concatenate([data_wind['g_load'], data_no_wind['g_load']]),
        'Max $q_c$': np.concatenate([data_wind['heat_flux'], data_no_wind['heat_flux']]),
        'Final $d$ [deg]': np.concatenate([data_wind['final_ang'], data_no_wind['final_ang']]),
        'Environment': ['Wind'] * len(data_wind['final_mach']) + ['No Wind'] * len(data_no_wind['final_mach'])
    })


    sns.set(style="whitegrid", context="notebook")

    pair_plot = sns.pairplot(
        df_all,
        vars=['Final $Ma$ [-]', 'Final $h$ [km]', 'Max $n$ [-]',
            'Max $q_c$', 'Final $d$ [deg]'],
        hue='Environment',
        diag_kind='kde',
        corner=True,  # Only show lower triangle
        plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'w'}
    )

    # Increase the figure size
    pair_plot.fig.set_size_inches(10, 8)

    # Increase space around subplots to avoid label cutoff
    pair_plot.fig.subplots_adjust(
        left=0.2,   # more space on the left
        bottom=0.25, 
        right=0.95,
        top=0.95
    )

    def format_func(value, _):
        return f'{value:.2f}'

    # Adjust label font sizes and formatting
    for i, ax in enumerate(pair_plot.axes.flatten()):
        if ax is not None:
            ax.tick_params(axis='both', labelsize=12)

            # Apply number formatting
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))

            # Limit number of ticks for readability
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

            # Rotate x-axis labels slightly less to prevent overlap
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_horizontalalignment('right')

            # Increase padding for y-axis labels
            ax.yaxis.labelpad = 12

    # Dynamically find the first non-None axis in the first column
    for i in range(len(pair_plot.axes)):  
        if pair_plot.axes[i, 0] is not None:  # Find the first existing y-axis
            pair_plot.axes[i, 0].set_ylabel("Final $Ma$ [-]")  # Set label
            break  # Stop after setting the first one

    pair_plot._legend.set_bbox_to_anchor((0.9, 0.9))  # Move legend to top-right
    pair_plot._legend.set_title("Environment", prop={'size': 18})  # Set title size
    for text in pair_plot._legend.texts:
        text.set_fontsize(14)  # Increase legend text size

    # Adjust layout to prevent cutoff
    plt.tight_layout()

    plt.show()