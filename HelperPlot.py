"""
Author : Francesco Benigno Suffoletta

This Python scripts collects some commonly implemented plots to be 
used for the re-entry propagation of a vehicle. The class takes as 
input the produced dependent variables that are collected from
TUDAT backend and it uses them for the production of plots for 
results visualization.

"""
import os
import numpy as np
import math
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, environment, propagation_setup

import Utils

class HORUS2B_Plotting:

    def __init__(self,
                bodies,
                state_history_array,
                dependent_variables_array): 
        """
        Inizialization of variables for the various plotting

        """
        # Recall the two bodies that have been created for the simulations
        self.HORUS = bodies.get_body("HORUS")
        self.earth = bodies.get_body("Earth")

        #Store the dependent variables and the state history
        self.dep_vars = dependent_variables_array
        self.state_history = state_history_array


        # Set the vehicle parameters which might be useful for the plots
        self.HORUS_mass = 26029.0
        self.HORUS_surface = 110.0
        self.HORUS_Rn = 0.8

        # Start by initializing all the different indices for the variables stored in the dependent variables
        self.epochs_index = 0
        self.mach_number_index  = 1
        self.altitude_index = 2
        self.tot_acceleration_index = 3
        self.latitude_index = 4
        self.longitude_index = 5
        self.heading_angle_index = 6
        self.flight_path_angle_index = 7
        self.angle_of_attack_index = 8
        self.bank_angle_index = 9
        self.sideslip_angle_index = 10
        self.g_load_index = 11
        self.density_index = 12
        self.temperature_index = 13
        self.dynamic_pressure_index = 14
        self.aerodynamic_acceleration_index = 15
        self.drag_coefficient_index = 16
        self.side_coefficient_index = 17
        self.lift_coefficient_index = 18
        self.pitch_moment_coefficient_index = 19
        self.airspeed_index = 22
        self.gravity_index = 23
        self.heat_flux_index = 24
        self.bodyflap_deflection_index = 25
        self.bodyflap_contribution_CD_index = 26
        self.bodyflap_contribution_CL_index = 27
        self.bodyflap_contribution_Cm_index = 28
        self.elevons_deflection_index = 29
        self.elevons_contribution_CD_index = 30
        self.elevons_contribution_CL_index = 31
        self.elevons_contribution_Cm_index = 32
        self.angular_distance_index = 33
        self.max_deadband_error_index = 34
        self.desired_heading_index = 35
        self.current_heading_error_index = 36

        # These are the indices for the state history in Cartesian coordinates
        self.x_index = 1
        self.y_index = 2
        self.z_index = 3
        self.Vx_index = 4
        self.Vy_index = 5
        self.Vz_index = 6

        # Define the constraints for the simulations (max g-load, max heat flux etc)
        self.heat_flux_constraint = 530000 # W/m^2
        self.g_load_constraint = 3 

        # Define parameters for the computation of constraints
        self.Vc = 7.9054E3
        self.n = 0.5
        self.m = 3.15
        self.c_star = 1.1097E8
        self.rho_0 = 1.225

    def plot_reentry_corridor( self ):

        """
        The function computes the re-entry corridor of the HORUS-2B
        and the current simulation, overlapping the propagated trajectory
        to ensure that no constraints have been violated

        Parameters 
        ----------
        dependent_variables_array : np.ndarray
            Array containing all the saved dependent variables

        Returns
        -------
        None

        """
        #Collect the different curves to define the constraints in the h-V graph
        V_q = self.get_heat_flux_constraint()

        V_g = self.get_gload_constraint()

        V_eq = self.get_equilibrium_glide_constraint()

        altitude = self.dep_vars[ : , self.altitude_index ] / 1e3

        velocity = self.dep_vars[ : , self.airspeed_index ]

        """
        plt.figure(figsize=(10, 6))
        plt.plot(dependent_variables_array[:, 0], np.rad2deg(dependent_variables_array[:, 9]), 'r', linewidth=2.5, label='Simulated HORUS bank angle', alpha=0.8)
        plt.plot(dependent_variables_array[:, 0], np.rad2deg(dependent_variables_array[:, 8]), 'b--', linewidth=2.5, label='Simulated HORUS angle of attack', alpha=0.8)
        plt.title('Bank Angle and AoA profiles using hyperbolic tangent curve', fontsize=16, fontweight='bold')
        plt.xlabel('Time [s]', fontsize=14)
        plt.ylabel('Control Angles [deg]', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.show()
        """

        plt.figure(figsize=(10, 6))

        # Plot the re-entry corridor boundaries
        plt.plot(V_eq, altitude, label='Quasi-equilibrium Glide Condition', linestyle='--', color='blue', linewidth =2.5, alpha = 0.8)
        plt.plot(V_g, altitude, label='Maximum g-load', linestyle='--', color='red', linewidth =2.5, alpha = 0.8)
        plt.plot(V_q, altitude, label='Maximum Heat Flux', linestyle='--', color='green', linewidth =2.5, alpha = 0.8)

        # Plot the actual trajectory
        plt.plot(velocity, altitude, label='Re-entry Trajectory', color='black', linewidth =2.5, alpha = 0.8)

        # Adding labels and title
        plt.xlabel('Velocity (m/s)', fontsize = 14)
        plt.ylabel('Altitude (km)', fontsize = 14)
        plt.xlim(0, 8000)
        plt.ylim(0, 150)
        plt.legend(loc='upper left', fontsize=14)
        plt.tight_layout()
        # Display the plot
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.show()

    def plot_ground_track( self ) :

        """
        The plot displays the ground-track of the HORUS-2B vehicle

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        longitude = np.rad2deg(self.dep_vars[:, self.longitude_index])
        latitude = np.rad2deg(self.dep_vars[:, self.latitude_index])

        fig, ax = plt.subplots(figsize=(10, 8))

        # Main map
        m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90,
                    llcrnrlon=-180, urcrnrlon=180, resolution='c')
        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(np.arange(-90., 91., 30.), labels=[True, False, False, False])
        m.drawmeridians(np.arange(-180., 181., 60.), labels=[False, False, False, True])

        # Plot trajectory
        x, y = m(longitude, latitude)
        m.plot(x, y, 'b', linewidth=2, label='HORUS Ground Track')

        # Landing area circle
        landing_lon, landing_lat = -53.07, 5.0
        x_center, y_center = m(landing_lon, landing_lat)
        landing_area_radius = m(0.75, 0)[0] - m(0, 0)[0]
        landing_area = Circle((x_center, y_center), radius=landing_area_radius,
                            color='red', fill=False, linewidth=2, linestyle='--', label='Terminal Landing Area')
        ax.add_patch(landing_area)

        # Define zoomed area in degrees
        zoom_lon_min, zoom_lon_max = landing_lon - 7.0, landing_lon + 7.0
        zoom_lat_min, zoom_lat_max = landing_lat - 7.0, landing_lat + 7.0

        # Correct rectangle in main plot
        x_rect_ll, y_rect_ll = m(zoom_lon_min, zoom_lat_min)
        x_rect_ur, y_rect_ur = m(zoom_lon_max, zoom_lat_max)
        rect_width = x_rect_ur - x_rect_ll
        rect_height = y_rect_ur - y_rect_ll
        rect = Rectangle((x_rect_ll, y_rect_ll), rect_width, rect_height,
                        linewidth=1.5, edgecolor='gray', linestyle='--', fill=False)
        ax.add_patch(rect)

        # Inset zoom
        axins = inset_axes(ax, width="25%", height="25%", loc='lower right', borderpad=2)
        m2 = Basemap(projection='mill',
                    llcrnrlon=zoom_lon_min, urcrnrlon=zoom_lon_max,
                    llcrnrlat=zoom_lat_min, urcrnrlat=zoom_lat_max,
                    resolution='i', ax=axins)
        m2.drawcoastlines()
        m2.drawcountries()

        # Plot trajectory in inset
        x2, y2 = m2(longitude, latitude)
        m2.plot(x2, y2, 'b', linewidth=2)

        # Landing area circle in inset
        x_center_zoom, y_center_zoom = m2(landing_lon, landing_lat)
        landing_area_zoom_radius = m2(0.75, 0)[0] - m2(0, 0)[0]
        landing_area_zoom = Circle((x_center_zoom, y_center_zoom), radius=landing_area_zoom_radius,
                                edgecolor='red', fill=False, linestyle='--', linewidth=2)
        axins.add_patch(landing_area_zoom)

        # Connect inset clearly to the main map manually
        ax.plot([x_rect_ll, x_rect_ur], [y_rect_ll, y_rect_ll], 'k--', linewidth=0.7)
        ax.plot([x_rect_ll, x_rect_ur], [y_rect_ur, y_rect_ur], 'k--', linewidth=0.7)
        ax.plot([x_rect_ll, x_rect_ll], [y_rect_ll, y_rect_ur], 'k--', linewidth=0.7)
        ax.plot([x_rect_ur, x_rect_ur], [y_rect_ll, y_rect_ur], 'k--', linewidth=0.7)

        # Hide inset ticks
        axins.set_xticks([])
        axins.set_yticks([])

        ax.legend(loc='lower left', fontsize=14)
        plt.tight_layout()

        plt.show()



    def plot_hV_trajectory( self ) : 

        """
        The plot displays the trajectory in the h-V graph for the
        HORUS-2B vehicle, annotating the highlights of the trajectory, 
        such as the maximum heat flux or the maximum g-load

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        
        g_load = self.dep_vars[ : , self.g_load_index ]
        heat_flux = self.dep_vars[ : , self.heat_flux_index ]

        max_heat_flux_idx = np.argmax(heat_flux)
        max_g_load_idx = np.argmax(g_load)

        velocity = self.dep_vars[ : , self.airspeed_index ]
        altitude = self.dep_vars[ : , self.altitude_index ]
        
        # Plot the trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(velocity, altitude, label='Trajectory')

        # Highlight the points of maximum heat flux and g-load
        plt.scatter(velocity[max_heat_flux_idx], altitude[max_heat_flux_idx], color='r', label='Max Heat Flux', zorder=5)
        plt.scatter(velocity[max_g_load_idx], altitude[max_g_load_idx], color='b', label='Max G-load', zorder=5)

        # Annotate the points
        plt.annotate('Max Heat Flux', (velocity[max_heat_flux_idx], altitude[max_heat_flux_idx]), textcoords="offset points", xytext=(-10,-10), ha='center', color='red')
        plt.annotate('Max G-load', (velocity[max_g_load_idx], altitude[max_g_load_idx]), textcoords="offset points", xytext=(-10,-10), ha='center', color='blue')


        # Add labels and legend
        plt.xlabel('Velocity (km/s)', fontsize = 14)
        plt.ylabel('Height (km)', fontsize = 14)
        plt.title('Re-entry Trajectory: Height vs. Velocity', fontsize = 16, fontweight = 'bold')
        plt.legend()
        plt.grid(True)

        # Show plot
        plt.show()

        
    def plot_footprint(latitudes, longitudes, target_latitude=None, target_longitude=None, title='Vehicle Footprint'):

        """
        Plots the footprint of the vehicle on a map of the Earth and highlights the target point if provided.

        Parameters
        ----------
        latitudes : np.ndarray
            Array of latitudes
        longitudes : np.ndarray
            Array of longitudes
        target_latitude : float
            Latitude of the target point. Defaults to None.
        target_longitude : float
            Longitude of the target point. Defaults to None.
        title : str
            Title of the plot

        Returns
        -------
        None

        """
        # Create a plot with a specific projection
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Add features to the map
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        # Plot the footprint
        ax.scatter(longitudes, latitudes, color='red', s=10, zorder=5, label='Footprint')

        # Plot the target point if provided
        if target_latitude is not None and target_longitude is not None:
            ax.scatter(target_longitude, target_latitude, color='blue', s=100, zorder=10, edgecolor='black', label='Target')

        # Add labels and title
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()

        # Add gridlines
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # Show the plot
        plt.show()

    def plot_3d_reentry_trajectory(self , target_latitude=None, target_longitude=None, target_radius= 0.75, title='3D Re-entry Trajectory'):

        """
        Plots the 3D re-entry trajectory in altitude, latitude, and longitude, with the Earth projected at altitude zero
        and a zoomed-in view of the target point.

        Parameters
        ----------
        target_latitude : float
            Latitude of the target point. Defaults to None
        target_longitude : float
            Longitude of the target point. Defaults to None
        target_radius : float
            Radius around the target point where the simulation terminates. Defaults to 1 degree
        title : str
            Title of the plot

        Returns
        -------
        None

        """
        fig = plt.figure(figsize=(14, 8))

        altitudes = self.dep_vars[ : , self.altitude_index]

        latitudes = self.dep_vars[ : , self.latitude_index]

        longitudes = self.dep_vars[ : , self.longitude_index]

        # 3D Trajectory Plot
        ax = fig.add_subplot(121, projection='3d')
        ax.plot(longitudes, latitudes, altitudes, label='Re-entry Trajectory', color='red')
        ax.scatter(longitudes[-1], latitudes[-1], altitudes[-1], color='blue', s=50, label='Final Point')

        # Plot target point if provided
        if target_latitude is not None and target_longitude is not None:
            ax.scatter(target_longitude, target_latitude, 0, color='green', s=100, edgecolor='black', label='Target')
            # Plot the target circle at zero altitude
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_lon = target_longitude + target_radius * np.cos(theta)
            circle_lat = target_latitude + target_radius * np.sin(theta)
            ax.plot(circle_lon, circle_lat, 0, color='green', linestyle='--')

        # Labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude (km)')
        ax.set_title(title)
        ax.legend()

        # Earth Projection at Altitude Zero
        ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
        ax2.add_feature(cfeature.LAND)
        ax2.add_feature(cfeature.OCEAN)
        ax2.add_feature(cfeature.COASTLINE)
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        ax2.add_feature(cfeature.LAKES, alpha=0.5)
        ax2.add_feature(cfeature.RIVERS)

        # Plot the footprint
        ax2.scatter(longitudes, latitudes, color='red', s=10, zorder=5, label='Footprint')

        # Plot the target point if provided
        if target_latitude is not None and target_longitude is not None:
            ax2.scatter(target_longitude, target_latitude, color='blue', s=100, edgecolor='black', label='Target')
            # Plot the target circle
            ax2.plot(circle_lon, circle_lat, color='green', linestyle='--')

        # Labels and title for the map
        ax2.set_title('Earth Projection with Footprint')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.legend()

        # Add gridlines
        ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # Show the plot
        plt.show()


    def get_heat_flux_constraint( self ) -> np.ndarray :

        """
        The function computes the velocity for the heat flux constraint for the HORUS-2B
        given the maximum heat flux set for the propagations

        Parameters
        ----------
        None
        
        Returns
        -------
        V_q : np.ndarray
            Array containing the velocity boundary at each altitude given by the heat flux constraint
        
        """

        # First we compute the vector of g-accelerations 
        g = self.dep_vars[ : , self.gravity_index ]

        rho = self.dep_vars[ : , self.density_index ]

        V_q = np.zeros((len(rho)))

        #Iterate over densities to get the heat flux velocity constraint
        for i in range(len(rho)):
            V_q[i] = self.Vc * ( self.HORUS_Rn**self.n * self.heat_flux_constraint * math.sqrt(self.rho_0/rho[i])/self.c_star)**(1/self.m)

        return V_q

    def get_gload_constraint( self ) -> np.ndarray :

        """
        The function computes the velocity for the heat flux constraint for the HORUS-2B
        given the maximum heat flux set for the propagations

        Parameters
        ----------
        None
        
        Returns
        -------
        V_q : np.ndarray
            Array containing the velocity boundary at each altitude given by the heat flux constraint
        
        """

        # First we compute the vector of g-accelerations 
        g = self.dep_vars[ : , self.gravity_index ]

        rho = self.dep_vars[ : , self.density_index ]

        CD = self.dep_vars[ : , self.drag_coefficient_index ]

        CL = self.dep_vars[ : , self.lift_coefficient_index ]

        V_g = np.zeros((len(rho)))

        #Iterate over densities to get the g-load velocity constraint
        for i in range(len(rho)):
            V_g[i] = math.sqrt((2 * self.g_load_constraint * self.HORUS_mass * g[i])/( self.HORUS_surface * rho[i] * math.sqrt(CD[i]**2 + CL[i]**2)))

        return V_g

    def get_equilibrium_glide_constraint( self ) -> np.ndarray :

        """
        The function computes the velocity for the heat flux constraint for the HORUS-2B
        given the maximum heat flux set for the propagations

        Parameters
        ----------
        None
        
        Returns
        -------
        V_eq : np.ndarray
            Array containing the velocity boundary at each altitude given by the equilibrium glide constraint
        
        """

        # First we compute the vector of g-accelerations 
        g = self.dep_vars[ : , self.gravity_index ]

        rho = self.dep_vars[ : , self.density_index ]

        CL = self.dep_vars[ : , self.lift_coefficient_index ]

        V_eq = np.zeros((len(rho)))

        #Iterate over densities to get the equilibrium glide constraint velocity
        for i in range(len(rho)):
            V_eq[i] = math.sqrt( self.Vc**2 / ( self.Vc**2 * ( CL[i] * self.HORUS_surface / (g[i]*self.HORUS_mass) ) * 0.5 * rho[i] + 1 ))

        return V_eq  

     