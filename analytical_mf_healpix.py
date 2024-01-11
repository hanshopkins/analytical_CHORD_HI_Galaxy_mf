import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import healpy as hp
from scipy.special import jv
from scipy.interpolate import LinearNDInterpolator

chord_theta = np.deg2rad(40.8)
chord_phi = 0
D = 6 #meters
L = 7 #meters
speedoflight = 3E8
wavelength = speedoflight/1000E6

source_theta = np.deg2rad(41)
omega = 2*np.pi/86400 #earth angular velocity in seconds
def source_phi (t): return (omega*t + np.pi + 0.0087)%(2*np.pi) - np.pi;

def rotation_matrix (t) : return np.array([[np.cos(omega*t), -np.sin(omega*t), np.zeros(t.shape[0])],[np.sin(omega*t), np.cos(omega*t), np.zeros(t.shape[0])],[np.zeros(t.shape[0]), np.zeros(t.shape[0]), np.ones(t.shape[0])]]).transpose(2,0,1)

phi_lower, phi_upper, theta_lower, theta_upper, m = np.deg2rad(-0.3), np.deg2rad(1), np.deg2rad(41.4), np.deg2rad(38), 22 #m is the number of dishes per row and column, so 8 or 22

chord_pointing = hp.pixelfunc.ang2vec(chord_theta, chord_phi)
dir1_proj_vec = hp.pixelfunc.ang2vec(chord_theta + np.pi/2, chord_phi) #north/south chord direction
dir2_proj_vec = np.cross(dir1_proj_vec, chord_pointing) #east/west chord direction

def B_sq (x):
    return (2*jv(1,np.pi*D*x/wavelength)/(np.pi*D*x/wavelength))**2

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
u = np.asarray(hp.pixelfunc.pix2vec(nside, pix)).T

time_samples = 1
times = np.linspace(0, 3600.0*24, time_samples+1)[:-1]
source_phi_array = source_phi(times)

def calc_denominator_single_t ():
    u_eq_u_p = np.linalg.norm(u - chord_pointing, axis = 1) <= 1E-12
    Bsq_values = np.where(u_eq_u_p, 1, B_sq(np.arccos(np.tensordot(u, chord_pointing, axes = 1))))
    return Bsq_values

def calc_denominator ():
    return np.sum(Bsq_values, axis = 0)

def calc_numerator_single_t (t):    
    source_pointing = hp.pixelfunc.ang2vec(source_theta, source_phi(t))
    u_rot = (rotation_matrix(t)@(u.T)).T
    
    cdir1 = np.pi*L/wavelength*np.dot(-u_rot+source_pointing,dir1_proj_vec)
    cdir2 = np.pi*L/wavelength*np.dot(-u_rot+source_pointing,dir2_proj_vec)
    
    u_s_eq_u_p = np.linalg.norm(source_pointing - chord_pointing) == 0
    if u_s_eq_u_p:
        Bsq_source_value = 1
    else:
        Bsq_source_value = B_sq(np.arccos(np.dot(source_pointing, chord_pointing)))
    
    
    u_eq_u_p = np.linalg.norm(u_rot - chord_pointing, axis = 1) <= 1E-12
    Bsq_values = np.where(u_eq_u_p, 1, B_sq(np.arccos(np.tensordot(u_rot, chord_pointing, axes = 1))))
    return Bsq_source_value * np.sin(cdir1*m)**2/np.sin(cdir1)**2 * np.sin(cdir2*m)**2/np.sin(cdir2)**2 * Bsq_values
    
def calc_numerator ():
    
    cdir1 = np.pi*L/wavelength*np.dot(source_pointing[:,np.newaxis,:]-u_rot,dir1_proj_vec)
    cdir2 = np.pi*L/wavelength*np.dot(source_pointing[:,np.newaxis,:]-u_rot,dir2_proj_vec)
    
    u_s_eq_u_p = np.linalg.norm(source_pointing - chord_pointing) == 0
    if u_s_eq_u_p:
        Bsq_source_value = 1
    else:
        Bsq_source_value = B_sq(np.arccos(np.dot(source_pointing, chord_pointing)))
    
    return np.sum(Bsq_source_value[:,np.newaxis]*np.sin(cdir1*m)**2/np.sin(cdir1)**2 * np.sin(cdir2*m)**2/np.sin(cdir2)**2 * Bsq_values, axis = 0)

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
    source_pointing = hp.pixelfunc.ang2vec(source_theta*np.ones(times.shape[0]), source_phi_array)
    print("Rotating test vectors for all times...")
    rot_matricies = rotation_matrix(times)
    u_rot = np.transpose(rot_matricies@(u.T), [0,2,1])

    print("Calculating B squared values...")
    u_eq_u_p = np.linalg.norm(u_rot - chord_pointing, axis = 2) <= 1E-12
    Bsq_values = np.where(u_eq_u_p, 1, B_sq(np.arccos(np.dot(chord_pointing, u_rot.transpose([0,2,1])))))

    #hmap = np.full(npix, hp.UNSEEN)
    
    print("Calculating the denominator...")
    denom = calc_denominator()
    #hmap[pix] = denom

    ##we need to calculate some values for the gnomview plotting
    #rot = [np.rad2deg((phi_lower+phi_upper)/2), 90-np.rad2deg((theta_lower+theta_upper)/2)]
    #plot_reso = np.rad2deg(phi_upper-phi_lower)*60.0/300 #in arcmins
    
    #hp.gnomview(hmap, rot=rot, xsize = 300, ysize=50, reso=plot_reso, title="Denominator")
    #hp.graticule()
    #hp.visufunc.projscatter(chord_theta, chord_phi, lonlat=False, color="red", marker="x", label="Chord pointing")
    #plt.legend()

    square_pixel_plot (denom, "Denominator")
    plt.plot(np.rad2deg(chord_phi), np.rad2deg(chord_theta), 'rx', ms=15, label="Chord pointing")
    plt.legend()
    
    #hmap = np.full(npix, hp.UNSEEN)
    if len(sys.argv) > 1 and sys.argv[1] == "--noSumt":
        num = calc_numerator_single_t(0)
    else:
        print("Calculating numerator...")
        num = calc_numerator()
    #hmap[pix] = num/denom
        
    #hp.gnomview(hmap, rot=rot, xsize = 300, ysize=50, reso=plot_reso, title="Matched filter")
    #hp.graticule()
    #hp.visufunc.projscatter(chord_theta, chord_phi, lonlat=False, color="red", marker="x", label="Chord pointing")
    square_pixel_plot (num/denom, "") #square_pixel_plot (num/denom, "Matched filter")
    plt.plot(np.rad2deg(chord_phi), np.rad2deg(chord_theta), 'rx', ms=15, label="Chord pointing")
    #if (source_phi(0) >= phi_lower and source_phi(0) <= phi_upper):  plt.plot(np.rad2deg(source_phi(0)), np.rad2deg(source_theta), 'bx', ms=15, label="Source location")
    #plt.legend()
     
    if len(sys.argv) > 1 and sys.argv[1] == "--show":
        plt.show()
    else:
        plt.savefig("matched_filter.png")
