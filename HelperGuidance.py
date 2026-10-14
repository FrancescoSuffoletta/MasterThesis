"""
Author : Francesco Benigno Suffoletta

This script is used within the framework of the Predictor-Corrector algorithm
for the guidance law of HORUS-2B. More in particular, the script is almost an
exact copy of the MainGuidance.py script with a few exceptions : first, the 
bank angle is of course not called many times also here, as we would be then 
having an infinite loop, but it is kept constant to the value that is provided 
from the Main Guidance script. Moreover, as the Predictor-Corrector law that 
is here developed is obtained through an asymmetric approach with respect to 
the Main Guidance script, some of the parameters are slightly changed (maximum
rate of change for the bank angle, termination options for the angular distance)
or shifted with respect to the original (AoA profile) in order to obtain the 
a more efficient or accurate algorithm.

"""

# Basic Packages
import numpy as np 
import math
import os

# TUDAT imports
from tudatpy.kernel.interface import spice_interface 
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, environment, propagation_setup
from tudatpy.math import interpolators

# Auxiliary Functions
import Utils

current_dir = os.path.dirname(__file__)


class HORUSHelperGuidance:

    def __init__(self, 
                bodies : environment.SystemOfBodies,
                bank_angle : float ,
                angle_of_attack : float,
                bodyflap_deflection,
                elevons_deflection,
                previous_bank, 
                sign_bank,
                arrival_latitude : float = np.deg2rad(5.0),
                arrival_longitude : float = np.deg2rad(-53.07)):

        # Extract HORUS and Earth bodies
        self.HORUS = bodies.get_body("HORUS_2")
        self.earth = bodies.get_body("Earth")

        # Initialize bank angle and angle of attack as the last ones from the main propagation
        self.constant_bank = bank_angle
        self.angle_of_attack = angle_of_attack

        # Extract the STS flight conditions, angle calculator, and aerodynamic coefficient interface
        environment_setup.add_flight_conditions( bodies, 'HORUS_2', 'Earth' )
        self.vehicle_flight_conditions = bodies.get_body("HORUS_2").flight_conditions
        self.aerodynamic_angle_calculator = self.vehicle_flight_conditions.aerodynamic_angle_calculator
        self.system_models = bodies.get_body('HORUS_2').system_models
        self.aerodynamic_coefficient_interface = self.vehicle_flight_conditions.aerodynamic_coefficient_interface

        # Define limits on aerodynamic angle changes and angles themselves
        self.max_alpha_dot = np.deg2rad( 5 )  # rad/s
        self.max_sigma_dot = np.deg2rad( 10 ) # rad/s 
        self.max_sigma = np.deg2rad(87)

        # Define all the different paths that might be needed 
        self.pathCD0 = current_dir + "/body_coefficients/CD0.txt"
        self.pathCL0 = current_dir + "/body_coefficients/CL0.txt"
        self.pathCm0 = current_dir + "/body_coefficients/Cm0.txt"
        self.pathCmb = current_dir + "/ctrl_surf_coefficients/dCmb.txt"
        self.pathCme = current_dir + "/ctrl_surf_coefficients/dCmw.txt"

        # Define the maximum angle of attack
        self.max_AoA = np.deg2rad(40)
        self.final_AoA = np.deg2rad(11.5) 

        # Define the maximum Mach number 
        self.maximum_mach = 20

        # Initialize the variables to be called for the update 
        self.useless1 = 0.0
        self.useless2 = 0.0
        self.useless3 = 0.0

        # We have the current forces from the database but we might need the moment database for the interpolations.
        self.Cm0, self.AoAList, self.MachList = Utils.extract_2D_database_HORUS(self.pathCm0)
        self.CD0, self.AoAList, self.MachList = Utils.extract_2D_database_HORUS(self.pathCD0)
        self.CL0, self.AoAList, self.MachList = Utils.extract_2D_database_HORUS(self.pathCL0)

        self.interpolate_2D_to_1D = Utils.TwoD_2_OneD_interpolation
        self.current_time = float("NaN")

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
        self.initial_constant_deflection1 = bodyflap_deflection # rad
        self.system_models.set_control_surface_deflection( self.surface_name1 , self.initial_constant_deflection1 )

        # Get the elevons database
        self.dCm_e, self.AoA_e, self.deflections_e = Utils.extract_3D_database_HORUS( self.pathCme )
        # Initialize also the elevons
        self.surface_name2 = 'elevons'
        self.initial_constant_deflection2 = elevons_deflection # rad
        self.system_models.set_control_surface_deflection( self.surface_name2 , self.initial_constant_deflection2 )
        # Provide maximum and minimum deflections
        self.min_deflection_elevons = np.deg2rad(-40)
        self.max_deflection_elevons = np.deg2rad(+40)

        self.latitude_f = arrival_latitude # rad
        self.longitude_f = arrival_longitude # rad

        # Initialize the flag for the first time it enters the cycle
        self.first_flag = True

        # Define the starting bank angle
        self.initial_bank = previous_bank

        # Initialize the period for the first maneuver to be applied 

        # Initize the deadband distances and the heading errors to be linearly interpolated
        self.deadband_distance = np.deg2rad(np.array([60 , 30 , 10 , 0.94331 , 0.91452 , 0.90739 , 0.86476 , 0.85776 , 0]))  # rad
        self.heading_error = np.deg2rad(np.array([15 , 15 , 23 , 23 , 20.36779 , 19.62079 , 15.20725 , 14.35438 , 14.35438]))  # rad

        # Initialize the sign of the bank angle as of the one that was used in the previous simulation
        self.sign_bank = sign_bank
        
        # Initialize a flag to indicate that we are performing a bank reversal or a maneuver
        self.currently_reversing = False
        self.currently_maneuvering = False

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

        # Define energy phases
        self.energy_phases = np.array([ -27000000.0, -2000000.0 ])
        self.aoa_energy = np.array([-10000000.00, -300000.00])

        # Initialize the variable taking count of the activation time of the previous guidance cycle
        self.previous_cycle_time = 0.0 

        # Define the periods over which the guidance  cycle shall be intialized depending on the phase we are in
        self.activation_period = np.array([ 20 , 2 ])

        # Initialize the variable for the previous bank angle and angle of attack
        self.previous_angle_of_attack = 0.0

        # Define bodies in the simulations
        self.bodies = bodies

        # Define flag whether we use of not the elevons
        self.use_elevons = True

    def getAngles(self, current_time: float):

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

        self.HelperGuidance( current_time )
        
        return np.array([self.angle_of_attack, 0.0, self.bank_angle])

    def HelperGuidance( self, current_time ) :

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
        # Update the aerodynamic angles
        if( math.isnan( current_time ) ):
            self.current_time = float("NaN")

        elif( current_time != self.current_time ):
            mach_number = self.vehicle_flight_conditions.mach_number

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

            # Similarly to what happens in the main guidance, we adjust the maximum bank rate depending on the energetic phase
            energy_list = self.get_current_energy()
            current_energy = energy_list[0]

            """
            if current_energy < self.energy_phases[0] :
                self.max_sigma_dot = np.deg2rad(10)
            elif (current_energy >= self.energy_phases[0]) and ( current_energy < self.energy_phases[1] ) :
                self.max_sigma_dot = np.deg2rad( 15 )
            else : 
                self.max_sigma_dot = np.deg2rad( 20 )
            """

            # Add maneuver if we just started simulation, from the previous bank to the commanded constant one
            if (self.currently_reversing == False ) :
                if (self.first_flag == True) or (self.currently_maneuvering == True) : 
                    self.period = self.compute_period_transition( self.initial_bank, self.constant_bank)
                    commanded_bank_angle = self.perform_maneuver( current_time, self.constant_bank, np.absolute(self.initial_bank))
                else : 
                    commanded_bank_angle = self.constant_bank
            else: 
                commanded_bank_angle = self.constant_bank

            # Turn the flag to false, only the first maneuver shall be introduced
            self.first_flag = False

            # Evaluate the need of reversals
            if self.evaluate_reversal( current_time, commanded_bank_angle ) : 
                # If the flag is still false, it means that we just initiated the reversal, thus we have to change the sign of the bank angle
                if not self.currently_reversing :
                    self.sign_bank = self.sign_bank * (-1)
                    commanded_bank_angle = self.perform_bank_reversal( current_time, self.start_reversal_time, commanded_bank_angle )
                else :
                    commanded_bank_angle = self.perform_bank_reversal( current_time, self.start_reversal_time, commanded_bank_angle )
            else : 
                pass

            self.bank_angle = commanded_bank_angle * self.sign_bank

            # Once the angles are updated, trim the vehicle
            current_aerodynamics_independent_variables = [self.angle_of_attack, mach_number]

            self.aerodynamic_coefficient_interface.update_coefficients(
                current_aerodynamics_independent_variables, current_time)
            
            # Current moment coefficients
            current_moment_coefficients = self.aerodynamic_coefficient_interface.current_moment_coefficients

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

            dict_cs = self.get_dic_for_control_surfaces()

            self.aerodynamic_coefficient_interface.update_full_coefficients(
                current_aerodynamics_independent_variables, dict_cs, current_time)

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
            self.system_models.set_control_surface_deflection( self.surface_name2 , 0.0 )

        else:
            # Still apply the reverse interpolation which will stop to the limit value, but we need to include 
            # the contributions from the elevons
            self.deflection_bodyflap = self.interpolate_1D( required_delta_Cm , self.deltaCm_vs_deflection, self.deflections, decreasing = True )
            self.system_models.set_control_surface_deflection( self.surface_name1 , self.deflection_bodyflap )
            if self.use_elevons :
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

    def Update_AoA( self ) :

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
        

        """
        #Method from ChatGPT
        heading_tau = math.atan2( math.sin( self.longitude_f - longitude ) * math.cos( self.latitude_f) , math.cos( latitude ) * math.sin( self.latitude_f ) - math.sin( latitude )*math.cos( self.latitude_f ) * math.cos( self.longitude_f - longitude ) ) # rad
        self.current_heading_error = self.current_heading - heading_tau
        """

        """
        
        heading_tau = np.arctan2( self.latitude_f - latitude , self.longitude_f - longitude )
        # Compute the current heading error in the range [-180,180]
        heading_error_range_180_180 = ( self.current_heading - heading_tau + np.pi ) % (2*np.pi) - np.pi
        # Store the current heading error as an attribute
        self.current_heading_error = heading_error_range_180_180

        """

        return [ heading_tau, self.current_heading_error ] #rad

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

        distance_list = self.get_angular_distance() # rad
        angular_distance = np.absolute(distance_list[0]) # The angular distance is in degrees and so is the list of distances where to interpolate!

        if angular_distance > self.deadband_distance[0] : 
            self.max_deadband_error = self.heading_error[0]
        elif angular_distance < self.deadband_distance[-1] :
            self.max_deadband_error = self.heading_error[-1]
        else :
            self.max_deadband_error = np.absolute(self.interpolate_1D( angular_distance , self.deadband_distance, self.heading_error, True ))

        return [ self.max_deadband_error ] # rad

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
            error_margin_because_of_reversal = current_derivative * estimated_maneuver_time / 2 # [rad]
            # Compute the actual dead-band limit we should use with the computed margin

            deadband_with_margin = self.max_deadband_error - error_margin_because_of_reversal 

            # If we exceed the deadband and currently we are not reversing, then we need to actuate the reversal
            if (deadband_with_margin < np.absolute(self.current_heading_error)) and (np.sign(self.sign_bank) != np.sign(self.current_heading_error)) : 
                #if (self.sign_bank < 0 and desired_heading > auxiliary) or (self.sign_bank > 0 and desired_heading < auxiliary) : NOT WORKING
                self.start_reversal_time = current_time
                Switch = True
            else : 
                Switch = False

        return Switch

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

        return bank_angle

    def updateData2( self ) -> list[ float ] :
    
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

        self.previous_angle_of_attack = self.angle_of_attack

        # Then we return the current heading information
        return [ self.useless1 , self.useless2 , self.useless3  ]
