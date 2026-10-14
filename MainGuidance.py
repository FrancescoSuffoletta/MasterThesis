"""
Author : Francesco Benigno Suffoletta

Main Guidance of the HORUS-2B re-entry vehicle.

The class produces the control angles of the vehicle:
  - the ANGLE OF ATTACK follows an energy-based law that transitions from the
    maximum AoA (40 deg) to the near-maximum-efficiency AoA (11.5 deg);
  - the BANK ANGLE magnitude is produced by the numerical PREDICTOR-CORRECTOR
    algorithm (longitudinal guidance), while its sign is handled by the LATERAL
    guidance through a heading-error deadband with bank reversals;
  - the script also trims the pitching moment (Cm = 0) at every step through
    the body flap, with the elevons as a secondary surface.

Supported guidance types:
  - 'Predictor-Corrector' : numerical predictor-corrector guidance only.
  - 'PINN'              : predictor-corrector + the PINN constraint corrector
                            (physics-informed flow-map surrogate, 50 s look-ahead,
                            batched-grid bisection bank correction with skip guard).
  - 'PINN_baseline'     : identical code path with the corrector DISABLED,
                            used as the paired baseline for A/B comparisons.
  - 'Verification'        : prescribed control-node profile for software
                            verification runs.

"""
# Basic Packages
import numpy as np 
import math
import os
import time
import pandas as pd
import json
import torch 
import joblib
from poliastro.earth.atmosphere.coesa76 import COESA76
from astropy import units as u

# TUDAT imports
from tudatpy.kernel.interface import spice_interface 
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, environment, propagation_setup
from tudatpy.math import interpolators
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime
from tudatpy.kernel.math import interpolators

# Auxiliary Functions
import Utils
from pinn_corrector import PINNCorrector   # PINN flow-map constraint corrector (50 s look-ahead)

current_dir = os.path.dirname(__file__)

class HORUSGuidanceAndTrim:

    def __init__(self, 
                bodies : environment.SystemOfBodies,
                Guidance : str, 
                wind,
                arrival_latitude : float = np.deg2rad(5.0),
                arrival_longitude : float = np.deg2rad(-53.07),
                pitch_trim : bool = True,
                use_elevons : bool = True,
                use_corrector : bool = True):

        # Master ON/OFF switch for the AI (PINN) constraint corrector.
        # Lets you A/B the SAME guidance with vs without the AI interface.
        # (Also forced OFF by the "PINN_baseline" guidance type below.)
        self._use_corrector_kwarg = use_corrector

        # Extract HORUS and Earth bodies
        self.HORUS = bodies.get_body("HORUS")
        self.earth = bodies.get_body("Earth")

        # HORUS dimensions
        self.mass = 26029.0 # kg 
        self.ref_area = 110.0 # m^2

        # Initialize bank angle and angle of attack as zero
        self.bank_angle = 0.0
        self.angle_of_attack = 0.0

        # Extract the STS flight conditions, angle calculator, and aerodynamic coefficient interface
        environment_setup.add_flight_conditions( bodies, 'HORUS', 'Earth' )
        self.vehicle_flight_conditions = bodies.get_body("HORUS").flight_conditions
        self.aerodynamic_angle_calculator = self.vehicle_flight_conditions.aerodynamic_angle_calculator
        self.system_models = bodies.get_body('HORUS').system_models
        self.aerodynamic_coefficient_interface = self.vehicle_flight_conditions.aerodynamic_coefficient_interface

        # Define limits on aerodynamic angle changes and angles themselves
        self.max_alpha_dot = np.deg2rad( 5 )  # rad/s
        self.max_sigma_dot = np.deg2rad( 15 ) # rad/s 
        self.max_sigma = np.deg2rad(87)

        # Define whether wind is present or not 
        self.wind = wind

        # Define all the different paths that might be needed 
        self.pathCD0 = current_dir + "/body_coefficients/CD0.txt"
        self.pathCL0 = current_dir + "/body_coefficients/CL0.txt"
        self.pathCm0 = current_dir + "/body_coefficients/Cm0.txt"
        self.pathCmb = current_dir + "/ctrl_surf_coefficients/dCmb.txt"
        self.pathCme = current_dir + "/ctrl_surf_coefficients/dCmw.txt"

        #Define the maximum angle of attack
        self.max_AoA = np.deg2rad(40)
        self.final_AoA = np.deg2rad(11.5) # To be changed

        #Define the maximum Mach number 
        self.maximum_mach = 20
        
        #Include boolean statements for the use of pitch trim and elevons
        self.pitch_trim = pitch_trim
        self.use_elevons = use_elevons

        self.useless1 = 0.0
        self.useless2 = 0.0
        self.useless3 = 0.0

        #Check whether we want to perform verification runs 
        self.guidance_type = Guidance

        # We have the current forces from the database but we might need the moment database for the interpolations.
        self.Cm0, self.AoAList, self.MachList = Utils.extract_2D_database_HORUS(self.pathCm0)
        self.CD0, self.AoAList, self.MachList = Utils.extract_2D_database_HORUS(self.pathCD0)
        self.CL0, self.AoAList, self.MachList = Utils.extract_2D_database_HORUS(self.pathCL0)

        #Define outside the pitch trim condition the only interpolating function we need without the trim 
        self.interpolate_2D_to_1D = Utils.TwoD_2_OneD_interpolation

        self.current_time = float("NaN")

        if self.pitch_trim: 
            # Define the required interpolating functions
            self.interpolate_3D_to_2D = Utils.ThreeD_interpolate
            self.interpolate_2D = Utils.TwoD_interpolation
            self.interpolate_1D = Utils.OneD_interpolation
            self.create_database = Utils.create_database

            #Import the aerodynamic database relative to the body-flap
            self.dCm, self.AoA, self.deflections = Utils.extract_3D_database_HORUS( self.pathCmb )

            # Define the minimum and maximum deflections of the bodyflaps to achieve the desired trim
            self.min_deflection_bodyflap = np.deg2rad(-20)
            self.max_deflection_bodyflap = np.deg2rad(+30)
            #Define the minimum dynamic pressure for the control surfaces to be activated and produce the desired moments
            self.minimum_CS_dynamic_pressure = 100 # N/m^2
            #Define control surfaces names
            self.surface_name1 = 'body_flap'

            # Define initial deflections (The second ones that relate to the elevons will be included later)
            self.initial_constant_deflection1 = np.deg2rad( 17 ) # rad
            self.system_models.set_control_surface_deflection( self.surface_name1 , self.initial_constant_deflection1 )

        if self.use_elevons:
            # Get the elevons database
            self.dCm_e, self.AoA_e, self.deflections_e = Utils.extract_3D_database_HORUS( self.pathCme )
            # Initialize also the elevons
            self.surface_name2 = 'elevons'
            self.initial_constant_deflection2 = np.deg2rad(0) # rad
            self.system_models.set_control_surface_deflection( self.surface_name2 , self.initial_constant_deflection2 )
            # Provide maximum and minimum deflections
            self.min_deflection_elevons = np.deg2rad(-40)
            self.max_deflection_elevons = np.deg2rad(+40)

        # We don't need the nodes in the validation runs
        if self.guidance_type == 'Predictor-Corrector' :

            print('Inizialization of Guidance variables')

            # Call arrival latitude and longitude
            self.latitude_f = arrival_latitude # rad
            self.longitude_f = arrival_longitude # rad

            # Initize the deadband distances and the heading errors to be linearly interpolated
            self.deadband_distance = np.deg2rad(np.array([60 , 30 , 10 , 0.94331 , 0.91452 , 0.90739 , 0.86476 , 0.85776 , 0]))  # rad
            self.heading_error = np.deg2rad(np.array([15 , 15 , 23 , 23 , 20.36779 , 19.62079 , 15.20725 , 14.35438 , 14.35438]))  # rad

            # Initialize the previous commanded angle
            self.previous_commanded_angle = 0.0

            # Initialize the flag for the command to bee generated by the predictor-corrector cycle 
            self.command =  False

            # Only activate the heat flux check once
            self.heat_check = 0

            # These variables will be deleted once we code the predictor-corrector
            self.bank_nodes = np.deg2rad(np.array([ 0 , 0 , 79.6 , 56 , 59.8 , 59.8 , 54 ]))
            self.aoa_nodes = np.deg2rad(np.array([ 40 , 40 , 40 , 40 , 40 , 40 , 11.5]))
            self.guidance_times = np.array([ 0 , 264 , 290 , 554 , 686 , 924 , 1319])

            # Initialize the sign of the bank angle
            self.sign_bank = +1
            
            # Initialize a flag to indicate that we are performing a bank reversal or maneuver
            self.currently_reversing = False
            self.currently_maneuvering = False

            # Initialize a starting reversal time for the modelling of the bank reversal
            self.start_reversal_time = 0.0

            # Maneuver time
            self.period = 0.0

            # Initialize cycle time
            self.cycle_time = 0.0

            # Auxiliar variables to compute the derivative of the heading error
            self.previous_heading_error = 0.0
            self.previous_heading_time = 0.0

            # Initialize the current heading error as 0 
            self.current_heading_error = 0.0

            # Initialize max deadband error 
            self.max_deadband_error = 0.0

            # Initialize the time epoch when the previous reversal ended
            self.previous_reversal_end = 0.0

            # We define two different energetic levels : The first one initializes the guidance cycles, 
            # the second one defines the approach of the TAEM, meaning that the frequency with which the 
            # guidance cycles are called needs to be higher 
            self.energy_phases = np.array([ -27000000.0, -2000000.0 ])
            self.aoa_energy = np.array([-1600000.00, -200000.00])

            # Initialize the variable taking count of the activation time of the previous guidance cycle
            self.previous_cycle_time = 0.0 

            # Define the periods over which the guidance  cycle shall be intialized depending on the phase we are in
            self.activation_period = np.array([ 100 , 20 ])

            # Initialize the variable for the previous bank angle and angle of attack
            self.previous_bank_angle = 0.0
            self.previous_angle_of_attack = 0.0

            # Initialize the time of the bank change
            self.start_maneuver_time = 0.0
            self.initial_bank_maneuver = 0.0

            self.bodies = bodies

        elif self.guidance_type in ("PINN", "PINN_baseline") :
            # ── PINN flow-map surrogate (256 wide, 6 residual blocks) + 50 s constraint
            #    look-ahead + batched-grid bisection bank correction ──────────────
            #  guidance_type "PINN"          -> corrector active (subject to kwarg)
            #  guidance_type "PINN_baseline" -> corrector OFF (pure predictor-
            #                                     corrector, for A/B comparison)
            print(f'Inizialization of Guidance variables ({self.guidance_type})')

            # Resolve the corrector ON/OFF switch
            self.use_corrector = (self.guidance_type == "PINN") and self._use_corrector_kwarg

            # Arrival target
            self.latitude_f = arrival_latitude
            self.longitude_f = arrival_longitude

            # Lateral guidance deadband / heading-error schedule
            self.deadband_distance = np.deg2rad(np.array([60 , 30 , 10 , 0.94331 , 0.91452 , 0.90739 , 0.86476 , 0.85776 , 0]))
            self.heading_error = np.deg2rad(np.array([15 , 15 , 23 , 23 , 20.36779 , 19.62079 , 15.20725 , 14.35438 , 14.35438]))

            # Predictor-corrector bookkeeping (identical to the other PINN modes)
            self.previous_commanded_angle = 0.0
            self.command = False
            self.heat_check = 0
            self.bank_nodes = np.deg2rad(np.array([ 0 , 0 , 79.6 , 56 , 59.8 , 59.8 , 54 ]))
            self.aoa_nodes = np.deg2rad(np.array([ 40 , 40 , 40 , 40 , 40 , 40 , 11.5]))
            self.guidance_times = np.array([ 0 , 264 , 290 , 554 , 686 , 924 , 1319])
            self.sign_bank = +1
            self.currently_reversing = False
            self.currently_maneuvering = False
            self.start_reversal_time = 0.0
            self.period = 0.0
            self.cycle_time = 0.0
            self.previous_heading_error = 0.0
            self.previous_heading_time = 0.0
            self.current_heading_error = 0.0
            self.max_deadband_error = 0.0
            self.previous_reversal_end = 0.0
            self.energy_phases = np.array([ -27000000.0, -2000000.0 ])
            self.aoa_energy = np.array([-1600000.00, -200000.00])
            self.previous_cycle_time = 0.0
            self.activation_period = np.array([ 100 , 20 ])
            self.previous_bank_angle = 0.0
            self.previous_angle_of_attack = 0.0
            self.start_maneuver_time = 0.0
            self.initial_bank_maneuver = 0.0
            self.bodies = bodies

            # Geometry / gravity constants used by collect_model_inputs()
            self.semimajor = 6371000
            self.f = 1 / 298.257223563
            self.J2 = 1.082626E-3
            self.mu_earth = 3.9859383624e14
            self.flag_check = False

            # The PINN corrector owns the model, scalers, atmosphere table, 50 s
            # rollout and the bisection-grid bank search.  Limits: 530 kW/m², 3 g.
            # Only built when active (baseline mode skips loading the model).
            if self.use_corrector :
                self.corrector = PINNCorrector(
                    current_dir=current_dir,
                    q_limit=530.0, g_limit=3.0,
                    q_margin=20.0, g_margin=0.1,     # target 510 kW/m² / 2.9 g (safety buffer)
                    horizon_s=50.0, dt=0.5,
                    atmosphere='coesa76'
                )
                print('  PINN constraint corrector: ENABLED')
            else :
                self.corrector = None
                print('  PINN constraint corrector: DISABLED (baseline run)')
            # How often (sim seconds) the 50 s constraint look-ahead is evaluated.
            # 10 s is safe: the 50 s horizon is 5x the cadence, so every breach is
            # still seen several checks ahead; halves the corrector overhead vs 5 s.
            self.PINN_period = 10.0

        elif self.guidance_type == 'Verification' :

            self.command = False

            # Call arrival latitude and longitude
            self.latitude_f = arrival_latitude
            self.longitude_f = arrival_longitude

            # Initialize the different nodes for the trajectory (with linear variations among them)
            self.bank_nodes = np.deg2rad(np.array([ 0 , 0 , 79.6 , 56 , 59.8 , 59.8 , 54 ]))
            self.aoa_nodes = np.deg2rad(np.array([ 40 , 40 , 40 , 40 , 40 , 40 , 11.5]))
            self.guidance_times = np.array([ 0 , 264 , 290 , 554 , 686 , 924 , 1319])
            self.bank_reversal_times  = np.array( [ 719.5, 1069, 1178, 1242.5, 10000] )  # We add one last reversal time that will never be met for sizes
            self.bank_index = 0

            # Initize the deadband distances and the heading errors to be linearly interpolated
            self.deadband_distance = np.deg2rad(np.array([60 , 30 , 10 , 0.94331 , 0.91452 , 0.90739 , 0.86476 , 0.85776 , 0]))
            self.heading_error = np.deg2rad(np.array([15 , 15 , 23 , 23 , 20.36779 , 19.62079 , 15.20725 , 14.35438 , 14.35438]))

            # Initialize the sign of the bank angle
            self.sign_bank = +1

            #Initialize a flag to indicate that we are performing a bank reversal
            self.currently_reversing = False

            # Initialize a starting reversal time for the modelling of the bank reversal
            self.start_reversal_time = 0.0

            # Auxiliar variables to compute the derivative of the heading error
            self.previous_heading_error = 0.0
            self.previous_heading_time = 0.0

            # Initialize the current heading error as 0 
            self.current_heading_error = 0.0

            # Initialize max deadband error 
            self.max_deadband_error = 0.0

            # Initialize the time epoch when the previous reversal ended
            self.previous_reversal_end = 0.0

    def getAerodynamicAngles(self, current_time: float):

        """
        The function calls in waterfall manner all the functions in the
        Guidance of the HORUS and provides as output the aerodynamic angles

        Parameters
        ----------
        current_time : float
            Current flight time of HORUS
        
        Returns
        -------
        self.angle_of_attack : float
            Angle of attack resulting as output of the guidance 
        angle_of_sideslip : float
            Set to 0 by default
        self.bank_angle : float
            Bank angle resulting as output of the guidance

        """

        self.updateGuidance( current_time )

        return np.array([self.angle_of_attack, 0.0, self.bank_angle])

    def updateGuidance(self, current_time : float):

        """
        The function calls all the routines to build up the guidance law
        of the HORUS and its propagation. More in particular, in this routine
        both the trim of the vehicle and its guidance are considered.

        Parameters
        ----------
        current_time : float 
            Current flight time of HORUS
        
        Returns 
        -------
        None

        """

        if( math.isnan( current_time ) ):
            self.current_time = float("NaN")
        elif( current_time != self.current_time ):
            #Update Angle of Attack
            mach_number = self.vehicle_flight_conditions.mach_number

            ##################################################
            ########## PREDICTOR-CORRECTOR GUIDANCE ##########
            ##################################################
            
            if self.guidance_type != 'Verification' :  

                ####################################################
                ############ ANGLE OF ATTACK DEFINITION ############
                ####################################################

                angle_of_attack = self.Update_AoA()

                if self.previous_heading_time != current_time :
                    if np.absolute((angle_of_attack - self.previous_angle_of_attack) / (current_time - self.previous_heading_time)) < self.max_alpha_dot : 
                        self.angle_of_attack = angle_of_attack 
                    else : 
                        if angle_of_attack < self.previous_angle_of_attack :
                            self.angle_of_attack = self.previous_angle_of_attack - self.max_alpha_dot * ( current_time - self.previous_heading_time)
                        else : 
                            self.angle_of_attack = self.previous_angle_of_attack + self.max_alpha_dot * ( current_time - self.previous_heading_time)
                else : 
                    self.angle_of_attack = angle_of_attack

                ######################################################
                ######### LONGITUDINAL BANK ANGLE DEFINITION #########
                ######################################################

                # Extract the current energy of HORUS, compare and evaluate in which phase we are in 
                energy_list = self.get_current_energy()
                current_energy = energy_list[0]

                # Before a specified energetic level we mantain the angle to 0 degrees
                if current_energy < self.energy_phases[0] :
                    """
                    commanded_bank_angle = self.Validation_Bank( current_time )
                    computed_bank_angle = commanded_bank_angle

                    """

                    commanded_bank_angle = 0.0
                    computed_bank_angle = 0.0

                # Once the first energetic phase has been concluded, we move to the predictor corrector low-frequency phase
                elif (self.energy_phases[0] <= current_energy) and ( current_energy < self.energy_phases[1]) :
                    if self.currently_reversing == False :
                        if (current_time - self.previous_cycle_time > self.activation_period[0]) and (not self.currently_maneuvering) :
                            self.command = True
                            commanded_bank_angle = self.Predictor_Corrector_cycle( np.absolute(self.previous_bank_angle), current_time ) # [rad] absolute value
                            self.previous_commanded_angle = commanded_bank_angle
                            self.period = self.compute_period_transition( self.previous_bank_angle, commanded_bank_angle)
                            computed_bank_angle = self.perform_maneuver( current_time, commanded_bank_angle, np.absolute(self.previous_bank_angle))
                        else :
                            self.command = False
                            commanded_bank_angle =  self.previous_commanded_angle
                            computed_bank_angle = self.perform_maneuver( current_time, commanded_bank_angle, np.absolute(self.previous_bank_angle))
                    else :
                        self.command = False
                        commanded_bank_angle = self.previous_commanded_angle
                else :
                    if self.currently_reversing == False :
                        if (current_time - self.previous_cycle_time > self.activation_period[1]) and (not self.currently_maneuvering) :
                            self.command = True
                            commanded_bank_angle = self.Predictor_Corrector_cycle( np.absolute(self.previous_bank_angle), current_time ) # [rad] absolute value
                            self.previous_commanded_angle = commanded_bank_angle
                            self.period = self.compute_period_transition( self.previous_bank_angle, commanded_bank_angle)
                            computed_bank_angle = self.perform_maneuver( current_time, commanded_bank_angle, np.absolute(self.previous_bank_angle))
                        else :
                            self.command = False
                            commanded_bank_angle =  self.previous_commanded_angle
                            computed_bank_angle = self.perform_maneuver( current_time, commanded_bank_angle, np.absolute(self.previous_bank_angle))
                    else :
                        self.command = False
                        commanded_bank_angle = self.previous_commanded_angle

                ############################################################
                ############## LATERAL BANK ANGLE DEFINITION ###############
                ############################################################

                if self.evaluate_reversal( current_time, commanded_bank_angle ) : 
                    # If the flag is still false, it means that we just initiated the reversal, thus we have to change the sign of the bank angle
                    if not self.currently_reversing :
                        self.sign_bank = self.sign_bank * (-1)
                        computed_bank_angle = self.perform_bank_reversal( current_time, self.start_reversal_time, commanded_bank_angle )
                    else :
                        self.command = False 
                        computed_bank_angle = self.perform_bank_reversal( current_time, self.start_reversal_time, commanded_bank_angle )
                else : 
                    pass

                if self.guidance_type in ("PINN", "PINN_baseline") :
                    # PINN path: 50 s look-ahead + batched-grid bisection correction.
                    # Skipped entirely when the corrector is disabled -> the run is
                    # then a pure predictor-corrector baseline (no AI interface).
                    if self.use_corrector and (current_time % self.PINN_period == 0) and (current_energy > -2.7e7) and (current_energy < -1.7e7) and ( self.currently_reversing == False ) :
                        commanded_bank_angle, new_computed = self.PINN_constraint_correction( current_time, commanded_bank_angle )
                        if new_computed is not None :
                            computed_bank_angle = new_computed
                else:
                        # No violation => do nothing
                        pass

                self.bank_angle = computed_bank_angle * self.sign_bank

            else : 

                ##################################################
                ############# VERIFICATION GUIDANCE ##############
                ##################################################

                self.angle_of_attack = self.Validation_AoA( current_time ) # The angle of attack only displays positive values in the reference trajectory of HORUS 2B
                commanded_bank_angle = self.Validation_Bank( current_time ) # This is just the absolute value
                if current_time == self.bank_reversal_times[self.bank_index] : 
                    
                    self.sign_bank = self.sign_bank * (-1)
                    self.bank_angle = self.perform_bank_reversal( current_time, self.start_reversal_time, commanded_bank_angle )
                    self.bank_index += 1

                else : 
                    self.bank_angle = commanded_bank_angle * self.sign_bank

            self.previous_bank_angle = self.bank_angle
            self.previous_angle_of_attack = self.angle_of_attack
                
            current_aerodynamics_independent_variables = [self.angle_of_attack, mach_number]

            self.aerodynamic_coefficient_interface.update_coefficients(
                current_aerodynamics_independent_variables, current_time)
        
            current_moment_coefficients = self.aerodynamic_coefficient_interface.current_moment_coefficients

            # If we select the pitch trim option (set to be true as default) we include the control surfaces increments
            # in the entire propagation
            if self.pitch_trim :
                if current_moment_coefficients[0] != 0.0: 
                    if self.Check_Control_Surfaces():
                        required_delta_Cm = -current_moment_coefficients[0]
                        self.Trim_Horus(mach_number, required_delta_Cm, current_moment_coefficients[0])
                    else:
                        pass
                else:
                    self.system_models.set_control_surface_deflection( self.surface_name1 , 0.0 )
                    if self.use_elevons:
                        self.system_models.set_control_surface_deflection( self.surface_name2 , 0.0 )
            else: 
                pass

            dict_cs = self.get_dic_for_control_surfaces()

            self.aerodynamic_coefficient_interface.update_full_coefficients(
                current_aerodynamics_independent_variables, dict_cs, current_time)

            if self.command :
                self.store_json_results( current_time )

    def Update_AoA( self ):

        """
        Function returns the updated value of the Angle of Attack, based on the current
        airspeed of the HORUS-2B, adopting a similar piecewise function of that used
        for the Space Shuttle 

        """
        current_energy = self.get_current_energy()

        if current_energy[0] <= self.aoa_energy[0]: 
            angle_of_attack = self.max_AoA
        elif self.aoa_energy[0] < current_energy[0] <= self.aoa_energy[1]:
            angle_of_attack = ((self.final_AoA  - self.max_AoA)/(self.aoa_energy[1] - self.aoa_energy[0]))*(current_energy[0] - self.aoa_energy[0]) + self.max_AoA
        else:
            angle_of_attack = self.final_AoA
            
        return angle_of_attack

    def store_json_results( self, current_time ) : 

        """
        The function stores some parameters during the propagation of the Horus-2B
        in order to train the Physics Informed Neural Network

        Parameters
        ----------
        current_time : float
            The current propagation time of the HORUS-2B vehicle

        Returns
        -------
        None

        """     

        if self.guidance_type == 'Predictor-Corrector' :
            # Define the partial path to store the .json file before moving it
            if self.wind :
                pre_path = current_dir + "\\SimulationOutput\\Wind\\PredCorr\\PreAnalysis\\"
            else :
                pre_path = current_dir + "\\SimulationOutput\\NoWind\\PredCorr\\PreAnalysis\\"

            if pre_path and not os.path.exists(pre_path):
                os.makedirs(pre_path)

            output_path = os.path.join(pre_path, "Train.json")

            if os.path.exists(output_path):
                # Load existing data
                with open(output_path, 'r') as file:
                    train_parameters = json.load(file)
            else:
                # If file doesn't exist, start with an empty dictionary
                train_parameters = {}

            energy = self.get_current_energy()

            angular_distance = self.get_angular_distance()

            update_parameters = { 
                "commanded bank [rad]" : self.previous_commanded_angle,
                "latitude [rad]" : self.vehicle_flight_conditions.latitude,
                "longitude [rad]" : self.vehicle_flight_conditions.longitude,
                "altitude [m]" : self.vehicle_flight_conditions.altitude,
                "velocity [m/s]" : self.vehicle_flight_conditions.airspeed,
                "flight_path_angle [rad]" : self.aerodynamic_angle_calculator.get_angle(environment.AerodynamicsReferenceFrameAngles.flight_path_angle),
                "heading angle [rad]" : self.aerodynamic_angle_calculator.get_angle(environment.AerodynamicsReferenceFrameAngles.heading_angle),
                "current_energy [-]" : energy[0],
                "angular_distance [rad]" : angular_distance[0],
                "cycle time [s]" : self.cycle_time
            } 

            train_parameters[current_time] = update_parameters

            with open(output_path, 'w') as file:
                json.dump(train_parameters, file, indent=4)
        else :
            pass
         
    def Check_Control_Surfaces( self ):

        """
        Function evaluates whether the dynamic pressure is high enough on the vehicle to activate the 
        contributions of the control surfaces

        """
        #If the current dynamic pressure is lower than the minimum one, the vehicle is not trimmable
        if self.vehicle_flight_conditions.dynamic_pressure > self.minimum_CS_dynamic_pressure:
            Vehicle_Trim = True
        else:
            Vehicle_Trim = False

        return Vehicle_Trim

    def computeHeatFlux( self ):

        """
        The routine computes the current Heat Flux
        incoming in the HORUS vehicle throughout its 
        propagation, given its current flight conditions,
        namely the airspeed and the density encountered
        at the specified altitude

        Parameters
        ----------
        None

        Returns
        -------
        heat_flux : list[float]
            Current heat flux experienced by HORUS-2B

        """

        # Extract current air density
        density = self.vehicle_flight_conditions.density
        # Extract air speed
        airspeed = self.vehicle_flight_conditions.airspeed

        if not math.isnan(density) :
            heat_flux = Utils.compute_Heat_Flux( airspeed, density )
        else:
            heat_flux = 0.0

        return  [ heat_flux ]

    def Trim_Horus( self,
                    mach_number : float,
                    required_delta_Cm : float,
                    current_pitch_moment : float
                    ) :

        """
        The function calls all the subroutines to trim the HORUS vehicle
        and delete the Cm0 arising from the current flight conditions

        Parameters
        ----------
        mach_number : float
            Current mach number experienced by HORUS
        required_delta_Cm : float
            Pitch moment required to trim the vehicle
        current_pitch_moment : float
            Current Cm0 without accounting for control surfaces contributions

        Returns 
        -------
        None

        """  
        
        if mach_number > self.maximum_mach :
            value_to_interpolate = self.maximum_mach
        else:
            value_to_interpolate = mach_number
        # Interpolate the desired current database given the current mach number, extracting a 2D table that
        # depends on the angle of attack and deflections
        self.AoA_vs_def_Cmb = self.interpolate_3D_to_2D( self.dCm, value_to_interpolate )

        # Now, provide the current angle of attack to achieve the interpolated Cm row wrt the control surface deflection
        self.deltaCm_vs_deflection = self.interpolate_2D_to_1D( self.angle_of_attack, self.AoA, self.AoA_vs_def_Cmb )

        # Having found the limits of the Cm we have three cases : required Cm is lower than the minimum achievable by the body flap;
        # required Cm is higher than the maximum achievable by the body flap or the required Cm is within the limits.
        # For the first two cases, we have to add the elevons contribution. 
        if self.deltaCm_vs_deflection[0] >= required_delta_Cm and self.deltaCm_vs_deflection[-1] <= required_delta_Cm:
            # Apply the required deflection as it is found within the possible excursions of the control surface!
            self.deflection_bodyflap = self.interpolate_1D( required_delta_Cm , self.deltaCm_vs_deflection, self.deflections, decreasing = True )
            if self.deflection_bodyflap > self.max_deflection_bodyflap:
                self.deflection_bodyflap = self.max_deflection_bodyflap
            elif self.deflection_bodyflap < self.min_deflection_bodyflap:
                self.deflection_bodyflap = self.min_deflection_bodyflap
            else:
                pass
            self.system_models.set_control_surface_deflection( self.surface_name1 , self.deflection_bodyflap )
            if self.use_elevons:
                self.system_models.set_control_surface_deflection( self.surface_name2 , 0.0 )
        else:
            # Still apply the reverse interpolation which will stop to the limit value, but we need to include 
            # the contributions from the elevons
            self.deflection_bodyflap = self.interpolate_1D( required_delta_Cm , self.deltaCm_vs_deflection, self.deflections, decreasing = True )
            self.system_models.set_control_surface_deflection( self.surface_name1 , self.deflection_bodyflap )
            if self.use_elevons:
                self.add_elevons_contribution( current_pitch_moment, value_to_interpolate )
    
    def add_elevons_contribution(self,
                                current_pitch_moment : float,
                                mach_number : float 
                                ) :
    
        """
        If we decide to set the elevons contributions to true,
        whenever the body_flap is not enough to trim the vehicle in use,
        the elevons are added to delete the remaining part of Cm

        Parameters
        ----------
        current_pitch_moment : float
            Current pitch moment from the body without considering control surfaces
        mach_number : float
            Current mach number experienced by HORUS (or maximum Mach if it goes over 20)

        Returns
        -------
        None

        """

        if self.use_elevons : 
            delta_Cm_bf = self.get_deltaCm_control_surface()
            required_delta_Cme = - ( current_pitch_moment + delta_Cm_bf )
            # Interpolate again to get the Cm database wrt AoA and deflection of the elevons 
            self.AoA_vs_def_Cme = self.interpolate_3D_to_2D( self.dCm_e, mach_number )
            self.deltaCme_vs_deflection_e = self.interpolate_2D_to_1D( self.angle_of_attack, self.AoA_e, self.AoA_vs_def_Cme )
            # Again, we have 3 possible conditions. Nevertheless, in this case, if the elevons are still not enough to trim
            # the vehicle, no additional control surfaces can help, therefore the vehicle will not be trimmed and it is not necessary 
            # to add the remaining conditions
            self.deflection_elevons = self.interpolate_1D( required_delta_Cme , self.deltaCme_vs_deflection_e, self.deflections_e, decreasing = True )
            if self.deflection_elevons > self.max_deflection_elevons :
                self.deflection_elevons = self.max_deflection_elevons
            elif self.deflection_elevons < self.min_deflection_elevons:
                self.deflection_elevons = self.min_deflection_elevons
            self.system_models.set_control_surface_deflection( self.surface_name2 , self.deflection_elevons )
        else : 
            pass
                                 
    def get_deltaCm_control_surface( self ) -> float:

        """
        The function is used to compute a 3D interpolation in order to get the current 
        pitch moment contribution provided by the specified control surface

        Parameters
        ----------
        None

        Returns
        -------
        delta_Cm : float
            Control surface contribution to HORUS
        
        """

        # We already have the interpolated 1D Cm vs Bodyflap, we just rereverse the previous interpolation
        delta_Cm = self.interpolate_1D( self.deflection_bodyflap , self.deflections, self.deltaCm_vs_deflection, decreasing = True )

        return delta_Cm

    def get_dic_for_control_surfaces( self ) :

        """
        The function returns the dictionary that is needed to call the update full coefficients 
        in order to call the contributions to the moment and forces from the control surfaces

        Parameters
        ----------
        None

        Returns
        dic_cs : dict
            Dictionary for the two implemented control surfaces
        
        """

        dict_cs = {'body_flap' : [self.angle_of_attack, self.system_models.get_control_surface_deflection('body_flap'), self.vehicle_flight_conditions.mach_number], 
                'elevons' : [self.angle_of_attack, self.system_models.get_control_surface_deflection('elevons'), self.vehicle_flight_conditions.mach_number]}

        return dict_cs

    def get_bodyflap_deflection( self ) -> list[ float ] :

        """
        Routine to extract the current deflection of the body flap.
        Used as a custom dependent variable.
        
        Parameters
        ----------
        None

        Returns
        -------
        current_deflection : list[ float ]
            Current deflection of the body_flap [rad]
        """

        # If pitch trim is considered, we output the deflection
        if self.pitch_trim:
            return [self.system_models.get_control_surface_deflection( self.surface_name1 )]
        # Otherwise we just output 0 (the control surface object might not even have been created)
        else:
            return [0] 

    def get_elevons_deflection( self ) -> list[ float ] :

        """
        Routine to extract the current deflection of the body flap.
        Used as a custom dependent variable.
        
        Parameters
        ----------
        None

        Returns
        -------
        current_deflection : list[ float ]
            Current deflection of the body_flap [rad]
        """

        # If pitch trim is considered, we output the deflection
        if self.use_elevons:
            return [self.system_models.get_control_surface_deflection( self.surface_name2 )]
        # Otherwise we just output 0 (the control surface object might not even have been created)
        else:
            return [0] 

    def get_bodyflap_contribution( self ) -> list[ float ] :

        """
        The function extracts the current contribution provided
        by the body flap to the different coefficients to be
        considered. 

        Parameters
        ----------
        None

        Returns
        -------
        coefficients : list[float]
            The coefficient increments provided by the body flap. In order the function returns CD, CL, Cm.

        """
        
        bf_contribution_forces = self.aerodynamic_coefficient_interface.current_control_surface_force_coefficient_increment(self.surface_name1)
        bf_contribution_moment = self.aerodynamic_coefficient_interface.current_control_surface_moment_coefficient_increment(self.surface_name1)

        return [bf_contribution_forces[0], bf_contribution_forces[2], bf_contribution_moment[0]]

    def get_elevons_contribution( self ) -> list[ float ] :

        """
        The function extracts the current contribution provided
        by the body flap to the different coefficients to be
        considered. 

        Parameters
        ----------
        None

        Returns
        -------
        coefficients : list[float]
            The coefficient increments provided by the body flap. In order the function returns CD, CL, Cm.

        """
        
        elevons_contribution_forces = self.aerodynamic_coefficient_interface.current_control_surface_force_coefficient_increment(self.surface_name2)
        elevons_contribution_moment = self.aerodynamic_coefficient_interface.current_control_surface_moment_coefficient_increment(self.surface_name2)

        return [elevons_contribution_forces[0], elevons_contribution_forces[2], elevons_contribution_moment[0]]

    def get_current_energy( self ) -> list[ float ] :

        """
        The function computes the current energy of the HORUS-2B
        throughout its propagation, which can are used to define
        the phases of the re-entry descent

        Parameters
        ----------
        None

        Returns 
        -------
        current_energy : float
            Current energetic level of the vehicle 

        """

        # First, collect all the quantities required to compute the energy, the distance from Earth's C.o.M and the velocity
        altitude = self.vehicle_flight_conditions.altitude # [ m ]
        R = self.earth.shape_model.average_radius + altitude 
        airspeed = self.vehicle_flight_conditions.airspeed 

        # Then, we can compute the energy of the vehicle 
        current_energy = (1/R) - (airspeed**2/2)

        return [ current_energy ]

    def get_current_heading_error( self ) :

        """
        The function computes the current deadband error to be compared
        with the maximum allowable one in order to evaluate whether or 
        not to perform the bank reversal

        Parameters
        ----------
        None

        Returns
        -------
        current_heading_error : float
            Current deadband error of the vehicle

        """

        longitude = self.vehicle_flight_conditions.longitude # [ rad ]
        latitude = self.vehicle_flight_conditions.latitude # [ rad ]
        self.current_heading = self.aerodynamic_angle_calculator.get_angle(environment.AerodynamicsReferenceFrameAngles.heading_angle)  # [rad]
        
        # atan2 can provide results in the range [-pi, pi]
        # Formula from Mooij and Bergsma (2016)
        heading_tau = math.atan2( math.sin(self.longitude_f - longitude ) , math.tan(self.latitude_f)*math.cos(latitude) - math.cos( self.longitude_f - longitude )*math.sin(latitude)  )
        # Error appears because we overshoot the longitude ! (Latitude as well ...) So there is a big discontinuity in the heading error and the guidance goes crazy
        self.current_heading_error = ( self.current_heading - heading_tau ) 
        

        return [ heading_tau, self.current_heading_error ] #rad

    def get_max_deadband_error( self ): 

        """
        The function linearly interpolates the current angular distance from
        target with respect to the tabulated max deadband error in order to 
        evaluate when to perform a bank reversal.

        Parameters
        ----------
        None

        Returns
        -------
        max_deadband_error : float [deg]
            Maximum allowable deadband error before performing a bank reversal

        """

        # self.deadband_distance = np.deg2rad(np.array([60 , 30 , 10 , 0.94331 , 0.91452 , 0.90739 , 0.86476 , 0.85776 , 0]))
        # self.heading_error = np.deg2rad(np.array([15 , 15 , 23 , 23 , 20.36779 , 19.62079 , 15.20725 , 14.35438 , 14.35438]))

        distance_list = self.get_angular_distance() # rad
        angular_distance = np.absolute(distance_list[0]) # The angular distance is in degrees and so is the list of distances where to interpolate!

        if angular_distance > self.deadband_distance[0] : 
            self.max_deadband_error = self.heading_error[0]
        elif angular_distance < self.deadband_distance[-1] :
            self.max_deadband_error = self.heading_error[-1]
        else :
            self.max_deadband_error = np.absolute(self.interpolate_1D( angular_distance , self.deadband_distance, self.heading_error, True ))

        return [ self.max_deadband_error ] # rad

    def get_angular_distance( self ) : 

        """
        Function provides the angular distance from the TAEM point and
        saves that as a dependent variable. The variable is needed as 
        the maximum accettable deadband error before applying a bank 
        reversal is defined with respect to this distance from the target.

        Parameters
        ----------
        None

        Returns
        -------
        angular_distance : float
            Angular distance from the terminal point [rad]

        """

        longitude = self.vehicle_flight_conditions.longitude
        latitude = self.vehicle_flight_conditions.latitude

        angular_distance = float(math.acos( math.sin(latitude)*math.sin(self.latitude_f) + math.cos(latitude)*math.cos(self.latitude_f)*math.cos(longitude - self.longitude_f) )) # rad

        return [ angular_distance ] #rad

    def Validation_AoA( self, current_time ) : 

        """
        The function provides the value for the angle of attack in the case
        in which the preliminary_guidance option is set to be false. The curve
        to be followed is that provided by the nodes. The AoA changes only once 
        during the propagation and in that period of time the change is set to 
        be linear.

        Parameters
        ----------
        current_time : float
            Current time of the propagation
        
        Returns
        -------
        angle_of_attack : float
            Commanded angle of attack

        """

        if current_time < self.guidance_times[5] : 
            angle_of_attack = np.deg2rad(40)
        elif current_time > self.guidance_times[6] :
            angle_of_attack = np.deg2rad(11.5)
        else :  
            # Check again if the formula is correct (it should be)
            angle_of_attack = np.deg2rad(40) + (np.deg2rad(11.5) - np.deg2rad(40)) / ( self.guidance_times[6] - self.guidance_times[5] ) * ( current_time - self.guidance_times[5])

        return angle_of_attack # rad

    def Validation_Bank( self, current_time ) :

        """
        The function provides the value for the bank angle when the 
        preliminary_guidance option is set to be false. Similarly to the 
        angle of attack case, the curve is provided by the nodes. Again, 
        as this function is only used for validation, a simple linear
        interpolation is used between two nodes.

        Parameters
        ----------
        current_time : float
            Current time of the propagation
        
        Returns
        -------
        bank_angle : float
            Commanded absolute value of the bank angle

        """
        
        if current_time < self.guidance_times[1] : 
            bank_angle = 0 
        elif current_time > self.guidance_times[6] :
            bank_angle = self.bank_nodes[6]
        else :  
            bank_angle = np.absolute(self.interpolate_1D( current_time , self.guidance_times, self.bank_nodes ) )

        return bank_angle # rad

    def evaluate_reversal( self, current_time, commanded_bank_angle ) :

        """
        The function evaluates whether or not it is necessary to perform a bank reversal
        during the propagation of the HORUS-2B, based on the current heading error of the 
        vehicle and the maximum tolerated heading error.

        Parameters
        ----------
        current_time : float
            Current flight time 
        commanded_bank_angle : float
            Value of the angle provided by the guidance law

        Returns
        -------
        Switch : bool
            Boolean affirming that we need or we are currently performing a bank reversal

        """

        # Check if current heading is always positive or we might need to use the absolute values
        if self.currently_reversing :
            # We have already started reversing, no need of evaluate the need of a reversal, the Switch remains on
            Switch = True
        else :
            # We update the current heading error
            self.get_current_heading_error()
            # We get the maximum deadband for the angular distance the vehicle is flying at
            self.get_max_deadband_error()
            # We provide an estimate of the heading error increase in the time where the bank reversal starts
            # but we are still banking in the same direction. If this estimate that goes by excess is over the 
            # maximum deadband error then is where we start a bank reversal. The estimated time for the reversal 
            # here goes by excess and it assumes that there is linear relation between half the angle to turn an
            # the maneuver itself. 
            estimated_maneuver_time = np.rad2deg( commanded_bank_angle ) + 10
            # We now compute the derivative of how the heading error is increasing over time
            if current_time > self.previous_heading_time:
                current_derivative = np.absolute(self.current_heading_error - self.previous_heading_error) / (current_time - self.previous_heading_time)
            else:
                current_derivative = 0

            # We achieved the derivative of the heading error! We can then evaluate if the vehicle is going to exceed the deadband limits
            # thus requiring a reversal or if we can just continue with the same sign of the bank angle
            error_margin_because_of_reversal = np.absolute(current_derivative * estimated_maneuver_time / 2) # [rad]
            # Compute the actual dead-band limit we should use with the computed margin
            deadband_with_margin = (self.max_deadband_error - np.deg2rad(7) ) - error_margin_because_of_reversal 
            # If we exceed the deadband and currently we are not reversing, then we need to actuate the reversal
            if (deadband_with_margin < np.absolute(self.current_heading_error)) and (np.sign(self.sign_bank) != np.sign(self.current_heading_error)) : 
                #if (self.sign_bank < 0 and desired_heading > auxiliary) or (self.sign_bank > 0 and desired_heading < auxiliary) : NOT WORKING
                self.start_reversal_time = current_time
                Switch = True
            else : 
                Switch = False

        return Switch 

    def perform_maneuver( self, current_time, computed_bank_angle, previous_bank_angle) :

        """
        Routine models the maneuvering from one to another bank angle

        Parameters
        ----------
        current_time : float
            Current flight time of the propagation
        computed_bank_angle : float
            Provided bank angle from the Predictor Corrector algorithm
        previous_bank_angle : float
            Previous bank angle 
        
        Returns
        -------
        bank_angle

        """

        if self.currently_maneuvering == False : 
            self.currently_maneuvering = True
            self.start_maneuver_time = current_time
            self.initial_bank_maneuver = previous_bank_angle
        
        delta_t = current_time - self.start_maneuver_time

        if delta_t <= self.period : 
            bank_angle = self.initial_bank_maneuver + (computed_bank_angle - self.initial_bank_maneuver) * 0.5 * (1 - np.cos(np.pi*delta_t/self.period))   
        else : 
            bank_angle = computed_bank_angle
            self.currently_maneuvering = False

        return bank_angle     

    def perform_bank_reversal( self, current_time, reversal_start_time, commanded_bank_angle ):

        """
        The routine models and performs the bank reversal when needed.
        The bank reversal is modelled using the hyperbolic tangent function
        so to have a smoother transition and in order to respect the actual
        constraints that we have with respect to the maximum rates of change
        of the bank angle that can be sustained by the HORUS-2B vehicle
        
        Parameters
        ----------
        current_time : float 
            Current flight time of the propagation
        reversal_start_time : float
            Epoch when the HORUS-2B started reversing
        commanded_bank_angle : float
            The teorethical bank angle that is commanded from the guidance law

        Returns
        -------
        bank_angle : float
            The current bank angle of the vehicle while reversing

        """

        # If we are performing the preliminary or the PINN guidance, the reversal logic is 
        # designed as a hyperbolic tangent curve, if it is the verification algorithm, the 
        # bank angle is instantaneously flipped.
        if self.guidance_type != "Verification" : 

            if self.currently_reversing == False : 
                self.currently_reversing = True

            # We now set up the hyperbolic tangent method to model the bank angle reversal
            # First, we have define the peak of the hyperbolic tangent curve, which coincides
            # with the commanded bank angle from the guidance law
            max_tanh = commanded_bank_angle # [rad]
            # We declare the derivative of the curve, which is the constraint of the maximum bank angle rate of change
            tanh_derivative = self.max_sigma_dot  # [rad/s]
            # Compute the tanh factor
            tanh_factor = tanh_derivative / max_tanh
            # Get an approximation of time taken to perform the reversal
            reversal_time = np.rad2deg(max_tanh)
            # Compute the elapsed time from the start of the reversal
            if reversal_start_time != 0.0 :
                delta_t = current_time - reversal_start_time
            else :
                delta_t = 0.0

            # If the time from the start of the reversal is longer than the estimated time 
            if delta_t > reversal_time : 
                self.currently_reversing = False
                self.start_reversal_time = 0.0
                bank_angle =  commanded_bank_angle
                self.previous_reversal_end = current_time
            else : 
                bank_angle =  max_tanh * np.tanh( tanh_factor * ( delta_t - reversal_time/2 ) )
        else:         
            bank_angle = commanded_bank_angle

        return bank_angle

    def Predictor_Corrector_cycle( self, initial_guess, current_time ) :

        """
        The routine produces bank angle values for the HORUS-2B to reach its 
        desired location. The value produced by this algorithm only works for
        the longitudinal guidance of the vehicle, while the lateral guidance
        is specified through a bank reversal logic. The bank angle is produced
        either by means of a Predictor-Corrector cycle , or a NN interface.

        Parameters
        ----------
        initial_guess : float
            Value of the bank angle to be provided in the cycle as initial guess
        current_time : float
            Value of the current flight time 
        
        Returns
        -------
        commanded_bank_angle : float
            Value of the bank angle that results from the cycle
        
        """

        # Load spice kernels
        #spice_interface.load_standard_kernels()

        # Start time of the cycle to keep track when to recall the new cycle
        self.previous_cycle_time = current_time 


        previous_heat_check = self.heat_check

        # Initialize index, threshold and first guess
        k = 0
        threshold = 100

        # Initialize the bank angle guesses and corresponding way-to-go ranges and final energy lists
        bank_guesses = []
        range_guesses = []
        energy_guesses = []

        # Define the initial state as the current state of the vehicle
        radius = self.earth.shape_model.average_radius + self.vehicle_flight_conditions.altitude
        latitude = self.vehicle_flight_conditions.latitude
        longitude = self.vehicle_flight_conditions.longitude
        velocity = self.vehicle_flight_conditions.airspeed 
        flight_path_angle = self.aerodynamic_angle_calculator.get_angle(environment.AerodynamicsReferenceFrameAngles.flight_path_angle)
        heading_angle =  self.aerodynamic_angle_calculator.get_angle(environment.AerodynamicsReferenceFrameAngles.heading_angle)
        spherical_initial_state = np.array((radius, latitude, longitude, velocity, flight_path_angle, heading_angle))

        # First we obtain the current angular distance from the target
        current_distance_list = self.get_angular_distance()
        angular_distance = np.absolute(current_distance_list[0])

        current_energy = self.get_current_energy()

        # Obtain the current deflection of the body-flaps and the elevons
        body_flap_deflection = self.get_bodyflap_deflection()
        elevons_deflection = self.get_elevons_deflection()

        # Define the desired final altitude and the desired final velocity
        final_distance = self.earth.shape_model.average_radius + 25E3
        final_velocity = 2.5 * 300 # Assuming 300 m/s for the speed of sound at 25 km and a 2.5 desired Mach
        # Compute the final desired velocity instead of energy
        desired_energy =  (1/final_distance) - (final_velocity**2/2)

        start_time = time.time()

        # Start the while loop
        while (k < 8) and ( threshold > 0.1 ) : 
            # If we are in the first iteration, we need to perturbate the first guess as we need two first guesses and integrate them both
            bodies_2 = self.create_copy_bodies( current_time )

            if k == 0 : 
                bank_guesses.append(initial_guess)
                integrated_distance, final_energy, max_flight_path, max_g_load, max_heat_flux, last_epoch, last_mach = self.predictor_HORUS( bodies_2, current_time, spherical_initial_state, bank_guesses[0], self.angle_of_attack, body_flap_deflection[0], elevons_deflection[0], initial_guess)
                range_guesses.append( integrated_distance - angular_distance )
                energy_guesses.append( final_energy - desired_energy )

                if max_flight_path > 0:
                    flag = True
                else : 
                    flag = False

                # In the first iter guesses are manually given, so the initial guess is perturbed adding more or less lift depending on the final integrated range 
                if (flag == False) and (initial_guess != 0):
                    if ( integrated_distance > angular_distance ) :          
                        bank_guesses.append( 1.05 * bank_guesses[0] )  
                    elif (integrated_distance < angular_distance ): 
                        bank_guesses.append( 0.95 * bank_guesses[0] ) 
                elif (flag == False) and (initial_guess == 0) :  # If the previous bank is equal to 0, we are most probably going out of phase 0 of the flight, therefore we set a firt high bank
                    bank_guesses.append(np.deg2rad(70))
                else : 
                    if (initial_guess == 0) : 
                        bank_guesses.append(np.deg2rad(70))
                    else : 
                        bank_guesses.append(1.05 * bank_guesses[0])

                bodies_2 = self.create_copy_bodies( current_time )
                integrated_distance, final_energy, max_flight_path, max_g_load, max_heat_flux, last_epoch, last_mach = self.predictor_HORUS( bodies_2, current_time, spherical_initial_state, bank_guesses[1], self.angle_of_attack, body_flap_deflection[0], elevons_deflection[0], initial_guess)
                range_guesses.append(integrated_distance - angular_distance)
                energy_guesses.append( final_energy - desired_energy )

                if max_flight_path > 0:
                    flag = True
                else : 
                    flag = False
            
            else :

                integrated_distance, final_energy, max_flight_path, max_g_load, max_heat_flux, last_epoch, last_mach = self.predictor_HORUS( bodies_2, current_time, spherical_initial_state, bank_guesses[k+1], self.angle_of_attack, body_flap_deflection[0], elevons_deflection[0], initial_guess)
                range_guesses.append( integrated_distance - angular_distance )
                energy_guesses.append( final_energy - desired_energy )

                if max_flight_path > 0:
                    flag = True
                else : 
                    flag = False

            # Define the terms to update the bank angle
            range_term = ((bank_guesses[k+1] - bank_guesses[k]) / (range_guesses[k+1] - range_guesses[k])) * range_guesses[k+1]
            energy_term = ((bank_guesses[k+1] - bank_guesses[k]) / (energy_guesses[k+1] - energy_guesses[k])) * energy_guesses[k+1]

            # Update the bank angle through the use of the secant method
            if current_energy[0] < 0.95 * self.energy_phases[0] :
                if flag == False : 
                    next_bank = bank_guesses[k+1] - range_term 
                else : 
                    next_bank = bank_guesses[k+1] + np.deg2rad(np.random.uniform(2,6))
                    if next_bank > np.deg2rad(81) : 
                        next_bank = np.deg2rad(np.random.uniform(75, 80))
            elif current_energy[0] > 1.05 * self.energy_phases[1] : 
                next_bank = bank_guesses[k+1] -  energy_term
            else :
                next_bank = bank_guesses[k+1] -  range_term 
            
            # Re-compute the threshold
            threshold = np.absolute((np.rad2deg(next_bank) - np.rad2deg(bank_guesses[k+1]))/np.rad2deg(bank_guesses[k+1]))
            # Append the produced solution of the bank angle in case the threshold is still not met for the next iteration
            bank_guesses.append(next_bank)
            # Add an iteration index
            k = k + 1

        end_time = time.time()

        self.cycle_time = end_time - start_time

        commanded_bank_angle = np.absolute(bank_guesses[-1])

        sigma_QECG = self.get_sigma_QECG( radius, flight_path_angle, velocity )
        
        if max_heat_flux > 530 and self.heat_check == 0 : 
            commanded_bank_angle = 0.0
            self.previous_cycle_time = current_time - 95
            self.heat_check = 1
        else :
            self.heat_check = self.heat_check + 1

        # G_Load and QECG check are included together
        if sigma_QECG > commanded_bank_angle : 
            if current_energy[0] > self.energy_phases[1]/2 :
                if sigma_QECG > np.deg2rad(85):
                    sigma_QECG = np.deg2rad(85)
                commanded_bank_angle = sigma_QECG 
                bodies_2 = self.create_copy_bodies( current_time )
                integrated_distance, final_energy, max_flight_path, max_g_load, max_heat_flux, last_epoch, last_mach = self.predictor_HORUS( bodies_2, current_time, spherical_initial_state, commanded_bank_angle, self.angle_of_attack, body_flap_deflection[0], elevons_deflection[0], initial_guess)
                if max_g_load < 3 and last_mach < 3 : 
                    pass
                else :
                    j=1
                    while (max_g_load > 3 or last_mach > 3) and j<7 :
                        bodies_2 = self.create_copy_bodies( current_time )
                        commanded_bank_angle = sigma_QECG - np.deg2rad(np.random.uniform(1,4))*j
                        integrated_distance, final_energy, max_flight_path, max_g_load, max_heat_flux, last_epoch, last_mach = self.predictor_HORUS( bodies_2, current_time, spherical_initial_state, commanded_bank_angle, self.angle_of_attack, body_flap_deflection[0], elevons_deflection[0], initial_guess)
                        sigma_QECG = commanded_bank_angle
                        j = j+1
            else :
                pass
        
        # Add checks here as it breaks 
        if commanded_bank_angle > np.deg2rad(85) :
            commanded_bank_angle = np.deg2rad(85) - np.deg2rad(np.random.uniform( 1, 10 ))
            bodies_2 = self.create_copy_bodies( current_time )
            integrated_distance, final_energy, max_flight_path, max_g_load, max_heat_flux, last_epoch, last_mach = self.predictor_HORUS( bodies_2, current_time, spherical_initial_state, commanded_bank_angle, self.angle_of_attack, body_flap_deflection[0], elevons_deflection[0], initial_guess)
            print('Final Mach and g-load')
            print(last_mach)
            print(max_g_load)
            if max_g_load < 3 and last_mach < 3 : 
                print('Apparently both under 3')
                pass
            else :
                print('One of them is over 3')
                initial_commanded_angle = commanded_bank_angle
                z=1
                while (max_g_load > 3 or last_mach > 3) and z<7 :
                    bodies_2 = self.create_copy_bodies( current_time )
                    commanded_bank_angle = initial_commanded_angle - np.deg2rad(np.random.uniform(1,4))*z
                    integrated_distance, final_energy, max_flight_path, max_g_load, max_heat_flux, last_epoch, last_mach = self.predictor_HORUS( bodies_2, current_time, spherical_initial_state, commanded_bank_angle, self.angle_of_attack, body_flap_deflection[0], elevons_deflection[0], initial_guess)
                    initial_commanded_angle = commanded_bank_angle
                    z = z+1


        # Check if the bank angle is not nan
        if np.isnan(commanded_bank_angle) or math.isinf(commanded_bank_angle) : 
            commanded_bank_angle = self.previous_commanded_angle

        # Just if we are very close to the approach at the end, adjust the shoot
        if last_epoch - current_time < 10: 
            commanded_bank_angle = np.deg2rad(np.random.uniform(30,50))

        if self.heat_check == 2 and previous_heat_check == 1 : 
            commanded_bank_angle = np.deg2rad(np.random.uniform(75,80))

        print('P-C cycle over')

        return commanded_bank_angle

    def create_copy_bodies( self, current_time ) :

        """
        The function creates the bodies for the simulations within the while loop 
        for the predictor-corrector guidance loops.

        Parameters
        ----------
        current_time : float
            Value of the current flight time 

        Returns
        -------
        bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation

        """

        # Start with the definition of the booleans for the different functions to be called
        main = False
        verification = False
        # Define start epoch
        simulation_start_epoch = 0.0 + current_time
        # Create main bodies
        bodies_to_create, global_frame_origin, global_frame_orientation = Utils.create_bodies()
        # Create body settings
        body_settings = Utils.get_environmental_settings(bodies_to_create, global_frame_origin, global_frame_orientation, simulation_start_epoch, verification, self.wind)
        # Create bodies
        bodies = environment_setup.create_system_of_bodies(body_settings)
        # Create the copy of the HORUS vehicle 
        Utils.add_HORUS_to_body_system(bodies, main)

        return bodies

    def predictor_HORUS( self, 
                        bodies_2, 
                        current_time, 
                        spherical_initial_state, 
                        bank_angle,
                        angle_of_attack,
                        bodyflap_deflection,
                        elevons_deflection, 
                        previous_bank_angle ) :

        """
        The routine integrates the HORUS-2B from its current state to the end using the provided 
        bank angle and keeping it constant for all the propagation in order to obtain the range
        to go with the provided cited angle. It specifies the same settings that are used for the main
        propagation.

        Parameters 
        ----------
        bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
            System of bodies present in the simulation
        current_time : float 
            Value of the current flight time 
        spherical_intial_state : np.ndarray
            Current spherical state of the vehicle
        bank_angle : float
            Value of the constant bank angle to be used
        bodyflap_deflection : float
            Current value of the bodyflap deflection
        elevons_deflection : float 
            Current value of the elevons deflection
        previous_bank_angle : float
            Previous value of the bank angle adopted by HORUS-2B

        Returns
        -------
        range_to_go : float
            Computed longitudinal distance 

        """
        # Define booleans statements
        main = False
        verification = False
        guidance_type = "Predictor-Corrector"

        # Define simulation start epoch
        simulation_start_epoch = 0.0 + current_time
        # Define termination settings
        maximum_duration = constants.JULIAN_DAY - current_time  # s
        termination_altitude = 25.0E3  # m
        current_energy = self.get_current_energy()
        if current_energy[0] < self.energy_phases[1] : 
            termination_angle = np.deg2rad(1) # deg
        else : 
            termination_angle = np.deg2rad(0.75)
        # Retrieve dependent variables to save
        dependent_variables_to_save = Utils.get_dependent_variable_save_settings( bodies_2, main , guidance_type, self.wind, bank_angle, angle_of_attack, bodyflap_deflection, elevons_deflection, previous_bank_angle, self.sign_bank)
        # Specify termination settings
        termination_settings = Utils.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     termination_altitude,
                                                     termination_angle,
                                                     bodies_2,
                                                     main,
                                                     guidance_type,
                                                     self.wind)
        # Setup interpolator
        interpolator_settings = interpolators.lagrange_interpolation(
                8,boundary_interpolation = interpolators.extrapolate_at_boundary)

        # Define the propagator settings for the secondary propagation of the vehicle
        propagator_settings = Utils.get_propagator_settings(bodies_2,
                        simulation_start_epoch,
                        termination_settings,
                        dependent_variables_to_save,
                        spherical_initial_state,
                        guidance_type,
                        main, 
                        self.wind,
                        bank_angle,
                        angle_of_attack,
                        bodyflap_deflection,
                        elevons_deflection, 
                        previous_bank_angle, 
                        self.sign_bank)

        # Define the integrator settings
        propagator_settings.integrator_settings = Utils.get_integrator_settings(0.5, simulation_start_epoch, True)
        # Simulate the trajectory
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies_2, propagator_settings
        )
        # Extract the final latitude and longitude to compute the range that has been travelled with the specified bank angle
        dependent_variables = dynamics_simulator.dependent_variable_history
        # Convert the dependent variables from a dictionary to a numpy array
        dependent_variables_array = result2array(dependent_variables)
        # Collect the initial latitude and longitude
        initial_latitude = spherical_initial_state[1]
        initial_longitude = spherical_initial_state[2]
        # Extract the final latitude and longitude from the simulation
        final_latitude = dependent_variables_array[-1, 4] # [rad]
        final_longitude = dependent_variables_array[-1, 5] # [rad]
        # Extract maximum g-load and heat_flux
        max_g_load = max(dependent_variables_array[:, 12]) # [-]
        max_heat_flux = max(dependent_variables_array[:30, 13]) # [kW/m^2]
        # Last epoch and Mach number
        last_epoch = dependent_variables_array[-1, 0]
        last_mach = dependent_variables_array[-1, 1]
        # Collect the quantities needed for the predictor-corrector
        range_to_go = float(math.acos( math.sin(initial_latitude)*math.sin(final_latitude) + math.cos(initial_latitude)*math.cos(final_latitude)*math.cos(initial_longitude - final_longitude) )) # rad
        final_energy = dependent_variables_array[ -1, 10]
        # Achieve the maximum flight path angle. If it is higher than 0, repeat the simulation
        max_flight_path = max(dependent_variables_array[ : , 7 ])

        return range_to_go, final_energy, max_flight_path, max_g_load, max_heat_flux, last_epoch, last_mach

    def get_sigma_QECG( self, radius, flight_path_angle, velocity ) : 

        """ 
        The function computes the value for the bank angle that permits to respect the equilibrium 
        gliding flight condition of the capsule. 

        Parameters
        ----------
        radius : float
            Current distance of the vehicle from the c.o.m 
        flight_path_angle : float
            Current valuen of the flight path angle
        velocity : float
            Current airspeed value of HORUS-2B

        Returns
        -------
        sigma_QECG : float
            Value of the bank angle that respects the equilibrium gliding flight condition  

        """

        # Compute the lift force 
        G = 3.986e14  # Earth's gravitational constant (m^3/s^2)
        density = self.vehicle_flight_conditions.density
        current_force_coefficients = self.aerodynamic_coefficient_interface.current_force_coefficients
        bodyflap_contribution = self.get_bodyflap_contribution()
        elevons_contribution = self.get_elevons_contribution()
        CL = current_force_coefficients[2] + bodyflap_contribution[1] +  elevons_contribution[1]
        L = 0.5 * velocity**2 * density * CL * self.ref_area
        g = G/(radius**2)
        W = self.mass * g
        centripetal_acceleration = velocity**2 / radius
        Vc = 7.9054E3

        numerator = self.mass * ( g - centripetal_acceleration ) * np.cos(flight_path_angle)
        denominator = L 
        
        # Ensure that the argument of arccos is within [-1, 1] to avoid errors
        cos_sigma_QECG = numerator / denominator
        cos_sigma_QECG = np.clip(cos_sigma_QECG, -1, 1)

        # Bank angle in radians
        sigma_QECG = np.arccos(cos_sigma_QECG)

        return sigma_QECG

    def compute_period_transition( self, previous_bank, next_bank) : 

        """ 
        The function computes the period to be used when performing bank change without
        exceeding the maximum bank rate of change. 

        Parameters
        ----------
        previous_bank : float
            Previous bank angle
        next_bank : float
            Future bank angle to be achieved

        Returns
        -------
        min_period : 
            Minimum period in which the maneuver shall be performed
        
        """
        # We compute the difference that needs to be achieved
        delta_bank = np.absolute(next_bank) - np.absolute(previous_bank)
        # We obtain from differentiation of the formula used to perform the transition the period (and we add 1 second to ensure that the rate of change is not exceeded)
        min_period = 0.5 * np.absolute(delta_bank) * np.pi / self.max_sigma_dot + 1

        return min_period

    def PINN_constraint_correction( self, current_time, commanded_bank_angle ) :

        """
        PINN constraint-avoidance step.  Uses the PINNCorrector (50 s flow-map
        look-ahead) to check whether the commanded bank breaches the heat-flux or
        g-load limit within the next 50 s and, if so, reduces the bank to the
        largest value that stays safe (batched-grid bisection — robust to the
        surrogate's jitter and ~30x cheaper than the previous Newton scheme).

        The current physical state is taken from collect_model_inputs (elements
        1..12 = [r, mach, lat, lon, heading, gamma, aoa, bank, density, CD, CL, V]),
        which is exactly the corrector's state convention.

        Parameters
        ----------
        current_time : float
            Current propagation time.
        commanded_bank_angle : float
            Absolute bank magnitude requested by the predictor-corrector [rad].

        Returns
        -------
        commanded_bank_angle : float
            Possibly-reduced absolute bank magnitude [rad].
        computed_bank_angle : float or None
            New maneuver-shaped bank if a correction was applied, else None
            (None => leave the bank computed earlier in updateGuidance untouched).
        """

        cmd = float(np.absolute(commanded_bank_angle))

        # Build the corrector state (drop time @0 and g_norm @13)
        state = self.collect_model_inputs( current_time, cmd, self.angle_of_attack )[0, 1:13]

        violation = self.corrector.check( state, cmd )
        if violation == "None" :
            return commanded_bank_angle, None

        safe_bank, info = self.corrector.find_safe_bank( state, cmd )

        print(f"[PINN t={current_time:.1f}] {violation}: bank "
              f"{np.rad2deg(cmd):.1f} -> {np.rad2deg(abs(safe_bank)):.1f} deg "
              f"(peak q={info.get('peak_q', float('nan')):.0f} kW/m2, "
              f"g={info.get('peak_g', float('nan')):.2f}) [{info['status']}]")

        # Adopt the corrected bank and reshape the maneuver, mirroring the
        # legacy block's accepted-correction branch.
        commanded_bank_angle = float(abs(safe_bank))
        self.previous_commanded_angle = commanded_bank_angle
        self.period = self.compute_period_transition( self.previous_bank_angle, commanded_bank_angle )
        computed_bank_angle = self.perform_maneuver( current_time, commanded_bank_angle, np.absolute(self.previous_bank_angle) )
        self.flag_check = False
        return commanded_bank_angle, computed_bank_angle

    def collect_model_inputs( self, time_val, bank_angle, aoa ) : 

        """
        Routine computes the array with the input values for the PINN model to work

        Parameters
        ----------
        time_val : float
            Current propagation time 
        bank_angle : float 
            Current value of the bank angle 
        aoa : float
            Current value of the angle of attack

        Returns 
        -------
        input_values : list
            List with the physical input values
        
        """

        mach_number = self.vehicle_flight_conditions.mach_number
        current_aerodynamics_independent_variables = [self.angle_of_attack, mach_number]
        dict_cs = self.get_dic_for_control_surfaces()
        self.aerodynamic_coefficient_interface.update_full_coefficients(
            current_aerodynamics_independent_variables, dict_cs, time_val)

        current_force_coefficients = self.aerodynamic_coefficient_interface.current_force_coefficients
        bodyflap_contribution = self.get_bodyflap_contribution()
        elevons_contribution = self.get_elevons_contribution()

        lat_val = self.vehicle_flight_conditions.latitude
        lon_val = self.vehicle_flight_conditions.longitude
        r_val = self.semimajor * (1 - self.f*np.sin(lat_val)**2) + self.vehicle_flight_conditions.altitude   
        heading_val = self.aerodynamic_angle_calculator.get_angle(environment.AerodynamicsReferenceFrameAngles.heading_angle)   
        gamma_val = self.aerodynamic_angle_calculator.get_angle(environment.AerodynamicsReferenceFrameAngles.flight_path_angle)      
        aoa_val = aoa      
        bank_val = bank_angle   
        density_val = self.vehicle_flight_conditions.density
        CD_val = current_force_coefficients[0] + bodyflap_contribution[0] + elevons_contribution[0]
        CL_val = current_force_coefficients[2] + bodyflap_contribution[1] + elevons_contribution[1]
        airspeed_val =  self.vehicle_flight_conditions.airspeed
        mach_val = mach_number

        gr = (self.mu_earth/r_val**2)*(1 - 1.5*self.J2*((6371e3/r_val)**2)*(3*np.sin(lat_val)**2 - 1))
        gphi = (self.mu_earth/r_val**2)*(-3*((6371e3/r_val)**2)*np.sin(lat_val)*np.cos(lat_val)*self.J2)
        g_norm_val = math.sqrt(gr**2 + gphi**2)

        input_values = [
            time_val, r_val, mach_val, lat_val, lon_val,
            heading_val, gamma_val, aoa_val, bank_val,
            density_val, CD_val, CL_val, airspeed_val, g_norm_val
        ]

        input_values = np.array(input_values, dtype=np.float32).reshape(1, -1)

        return input_values

    def updateData( self ) -> list[ float ] :
        
        """
        Routine used only to update data for derivatives computation. 
        It gets called for the dependent variables update each time the cycle ends
        but it just returns three useless variables

        Parameters
        ----------
        None

        Returns
        -------
        self.useless1 : float
            First useless variable
        self.useless2 : float
            Second useless variable
        self.useless3 : float
            Third useless variables
        
        """
        
        self.previous_heading_error = self.current_heading_error
        self.previous_heading_time = self.current_time

        # Then we return the current heading information
        return [ self.useless1 , self.useless2 , self.useless3  ]