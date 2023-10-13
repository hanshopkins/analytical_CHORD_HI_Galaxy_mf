import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import healpy as hp
from scipy.interpolate import LinearNDInterpolator
from cpp_amf_wrapper import amf

chord_theta = np.deg2rad(40.8)
chord_phi = 0
D = 6 #meters
L = 7 #meters
speedoflight = 3E8
wavelength = speedoflight/1000E6
source_theta = np.deg2rad(41)
source_phi_0 = 0.0087

time_samples = 2000
delta_tau = 24.0*3600/time_samples

phi_lower, phi_upper, theta_lower, theta_upper, m = np.deg2rad(-0.3), np.deg2rad(1), np.deg2rad(41.4), np.deg2rad(38), 22 #m is the number of dishes per row and column, so 8 or 22

####################setting healpix coords
#first calculating the needed resolution
sky_fraction_to_plot = ((phi_upper-phi_lower)*(theta_lower-theta_upper))/(2*np.pi**2) #sky fraction to plot
wanted_npix_in_plot = 200*300
wanted_total_pix = wanted_npix_in_plot/sky_fraction_to_plot
nside = hp.pixelfunc.get_min_valid_nside(wanted_total_pix)
npix = hp.nside2npix(nside)
#polygon of relevant pixels
poly_vertices = np.empty([4,3])
poly_vertices[0] = hp.pixelfunc.ang2vec(theta_upper, phi_lower)
poly_vertices[1] = hp.pixelfunc.ang2vec(theta_upper, phi_upper)
poly_vertices[2] = hp.pixelfunc.ang2vec(theta_lower, phi_upper)
poly_vertices[3] = hp.pixelfunc.ang2vec(theta_lower, phi_lower)

pix = hp.query_polygon(nside, poly_vertices, fact=8, inclusive=True)
test_vecs = np.asarray(hp.pixelfunc.pix2vec(nside, pix)).T

#declaring/calculating some stuff for the below function
npix_width, npix_height = 300, 300
phipal = (phi_upper-phi_lower)/npix_width #stands for phi pixel angular length
phi_values = np.linspace(phi_lower+phipal/2.0, phi_upper-phipal/2.0, npix_width)
thetapal = (phi_upper-phi_lower)/npix_height #stands for theta pixel angular length
theta_values = np.linspace(theta_lower+thetapal/2.0, theta_upper-thetapal/2.0, npix_height)
pix_angles = (np.stack(hp.pixelfunc.pix2ang(nside, pix), axis=1)+np.pi)%(2*np.pi) - np.pi
def square_pixel_plot (values_array, title):
    #linearly interpolate between healpix
    interpolator = LinearNDInterpolator(pix_angles, values_array)
    plotting_values = interpolator(*np.meshgrid(theta_values,phi_values)).T
    fig = plt.figure()
    plt.imshow(plotting_values, origin = "upper", interpolation = "none", 
        extent = (np.rad2deg(phi_lower), np.rad2deg(phi_upper), np.rad2deg(theta_upper), np.rad2deg(theta_lower)))
    plt.title(title)
    plt.xlabel(r"$\Phi$")
    plt.ylabel(r"$\theta$")
    
if __name__ == "__main__":
    mfo = amf(chord_theta, wavelength, source_theta, source_phi_0, m, test_vecs, delta_tau, time_samples)
    square_pixel_plot (mfo, "Matched filter")
    plt.show()
