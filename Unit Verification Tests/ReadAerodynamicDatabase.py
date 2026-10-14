import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from poliastro.earth.atmosphere.coesa76 import COESA76
from astropy import units as u
import math
import joblib
import torch

# The unit test here prints the 2D and 3D aerodynamic database that is extracteed using the functions in Utils.py, 
# and it checks whether the results are accurate and working.

# Obtain the path for the root of this 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import Utils
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, environment, propagation_setup
from tudatpy.math import interpolators
from tudatpy.util import result2array
from MainGuidance import HORUSGuidanceAndTrim

# We now define the paths for the different database to be compared with what gets printed out.

Verify1 = False
Verify_Interp1 = False
Verify_qflux = False
Verify_3D_interp = False
Verify_atmo = False
Verify_GLoad = False                    
Verify_Scalers = False

if Verify1:

    # 2D databases (CD, CL, Cm)
    CD_file = os.path.join(project_root, 'body_coefficients', 'CD0.txt')
    CL_file = os.path.join(project_root, 'body_coefficients', 'CL0.txt')
    Cm_file = os.path.join(project_root, 'body_coefficients', 'Cm0.txt')

    #3D databases (As an example only a few of them)
    dCDb_file = os.path.join(project_root, 'ctrl_surf_coefficients', 'dCDb.txt')
    dCLb_file = os.path.join(project_root, 'ctrl_surf_coefficients', 'dCLb.txt')

    CD0, AoA_CD0, MaCD = Utils.extract_2D_database_HORUS(CD_file)
    CL0, AoA_CL0, MaCL = Utils.extract_2D_database_HORUS(CL_file)
    Cm0, AoA_Cm0, MaCm = Utils.extract_2D_database_HORUS(Cm_file)
    
    DatabasedCDb , IndVar1, IndVar2 = Utils.extract_3D_database_HORUS(dCDb_file)
    DatabasedCLb , CLVar1, CLVar2 = Utils.extract_3D_database_HORUS(dCLb_file)

    #print('AoA for CD0:')
    #print(CD0)

    #print('Mach numbers for CL0:')
    #print(AoA_CL0)

    #print('Array of Cm:')
    #print(Cm0)



    #print('Deflections of dCLb:')
    #print(CLVar2)

    mach = []

    for key in DatabasedCDb.keys():
        mach.append(key)

    mach = np.array(mach)

    print(mach)

    print('previous matrix')
    print(DatabasedCDb[2])

    print('following matrix')
    print(DatabasedCDb[3])

    cm_matrix = Utils.ThreeD_interpolate(DatabasedCDb, 2.2)
    print('Interpolated matrix')
    print(cm_matrix)

if Verify_Interp1:

    spice_interface.load_standard_kernels()

    bodies_to_create, global_frame_origin, global_frame_orientation = Utils.create_bodies()

    simulation_start_epoch = 0

    # Create body settings
    body_settings = Utils.get_environmental_settings(bodies_to_create, global_frame_origin, global_frame_orientation, simulation_start_epoch)

    # Create bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    bodies.create_empty_body('HORUS')
    
    HORUS_mass = 26029.0 #kg

    bodies.get_body('HORUS').set_constant_mass(HORUS_mass)

    guidance = HORUSGuidanceAndTrim(bodies)

    interpolator_bf, interpolator_elevons = guidance.create_interpolator_for_pitch_trim()

    # --> Does not work!

    current_mach = 3.2

    aoa_vs_body_flap = interpolator_bf.interpolate( current_mach )

    print(aoa_vs_body_flap)

if Verify_qflux:
    airspeed = 4500
    density = 1E-4
    q_dot = Utils.compute_Heat_Flux(airspeed, density)
    print(q_dot)

    HORUS_RN = 0.8 # m Nose Radius
    c_star = 1.1097E8 # Aerodynamic Characteristic
    Vc = 7.9054E3 # Circular Velocity 
    n = 0.5
    m = 3.15
    rho_0 = 1.225 # kg/m^3

    q_dot = c_star*((HORUS_RN)**n)*((density/rho_0)**(1-n))*(airspeed/Vc)**m/(1000)
    print(q_dot)

if Verify_3D_interp:

    dCmb_file = os.path.join(project_root, 'ctrl_surf_coefficients', 'dCmb.txt')

    Cm = Utils.ThreeD_interpolate(dCmb_file, 11)

    print(Cm)

if Verify_atmo : 
    # Let's just verify that this implementation is coherent with the TUDAT one!
    # Altitude 2, density 12

    # Couple of examples, alt = 45743.45760873798; density = 0.0017752871568270283
    altitude = 45743.45760873798 * u.m
    # Create an Atmosphere object
    atmosphere = COESA76()
    density = atmosphere.density(altitude)  # Returns density in kg/m^3
    temperature = atmosphere.temperature(altitude)
    print('Density is ')
    print(density.value)
    print('Speed of Sound is')
    print(temperature)

    # Let's say acceptable 
    # Let's evaluate speed of sound with the formula tho !
    # Same altitude
    gamma = 1.4
    R = 287.05 
    tudat_speedofsound = 1866.2444055079463 / 5.7061811192318315
    print('TUDAT')
    print(tudat_speedofsound)
    sound = math.sqrt(gamma*R*temperature.value)
    print('Check')
    print(sound)
    print(1866.2444055079463/sound)
    # We can use sound too

    # KEEP THIS 

if Verify_GLoad : 
    # Let's verify how 'hard' is to compute the g-load in our way! and compare it to the actual
    # G-Load is the 11th column in the dependent variables
    # We need mass, density, velocity, surface, Cl, Cd. 
    # Real G-Load : 1.433186528384503

    altitude = 40622.63058418408 * u.m
    atmosphere = COESA76()
    density = atmosphere.density(altitude)  # Returns density in kg/m^3 
    velocity = 1469.8603
    M = 26029.0 
    Surface = 110.0
    earth_mass = 5.972e24
    G = 6.6743e-11
    r = altitude.value + 6371000
    g = G*earth_mass * 1.001 / r**2
    CD = 0.467068
    CL = 0.70232319
    print(g)
    # real g is 9.711872981499793
    # Lets try another g!
    J2 =  1.082626E-3
    mu_earth =  3.9859383624e14
    latitude = 0.06756309583537812
    gr = (mu_earth/r**2)*(1 - 1.5*J2*((6371e3/r)**2)*(3*np.sin(latitude)**2 - 1))
    gphi = (mu_earth/r**2)*(-3*((6371e3/r)**2)*np.sin(latitude)*np.cos(latitude)*J2)
    gnew = math.sqrt(gr**2 + gphi**2)
    print('G with spherical harmonics')
    print(gnew)
    L = 0.5*density.value*velocity**2*Surface*CL
    D = 0.5*density.value*velocity**2*Surface*CD
    #cd 16; cl 18
    g_load = math.sqrt(L**2 + D**2)/ (M*g)
    print(g_load)

    # Last check
    semimajor = 6371000
    f = 1 / 298.257223563
    latitude = -0.3898254580107936
    realR = math.sqrt( 1728756.9416571525**2 + 5744803.099245918**2 + 2464809.6562738353**2)
    print('Real distance is')
    print(realR)
    altitude = 117978.73760735802
    r_earth = semimajor * (1 - f*np.sin(latitude)**2) + altitude
    print('Computed')
    print(r_earth)

if Verify_Scalers : 
    scalers_path = os.path.join(project_root, 'PINN_Scalers')
    scalers = {}
    for feature in ['time', 'r', 'mach', 'latitude', 'longitude', 'heading', 'flight_path','aoa', 'bank', 'density', 'CD', 'CL',  'airspeed', 'g_norm']:
        scalers[feature] = joblib.load(f'{scalers_path}\\{feature}_scaler.pkl')

    # Let's see if the scalers work properly
    r = 110000 + 6370000
    r_array = np.array([r])
    r_norm = scalers['r'].transform(r_array.reshape(-1, 1))
    print('r_norm is')
    print(r_norm)
    r_inverse = scalers['r'].inverse_transform(r_norm.reshape(-1, 1))
    print('Reinversed')
    print(r_inverse)

    time = 0.5
    t_array = np.array([time])
    t_norm = scalers['time'].transform(t_array.reshape(-1, 1))
    print('t_norm is')
    print(t_norm)
    t_inverse = scalers['time'].inverse_transform(t_norm.reshape(-1, 1))
    print('Reinversed')
    print(t_inverse)
