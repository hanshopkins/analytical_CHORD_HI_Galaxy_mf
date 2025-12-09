import numpy as np
import ctypes
import os
from numpy.ctypeslib import ndpointer
from util import vec2ang

amfLib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "amf.so"))
cpp_amf = amfLib.analytic_matched_filter
cpp_amf.argtypes = [ndpointer(ctypes.c_double), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_ushort, ctypes.c_ushort, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_uint, ctypes.c_double, ctypes.c_uint, ctypes.c_uint, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

cpp_amf_su = amfLib.analytic_matched_filter_single_u
cpp_amf_su.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_ushort, ctypes.c_ushort, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.c_double, ctypes.c_uint, ctypes.c_uint]
cpp_amf_su.restype = ctypes.c_double

cpp_sb = amfLib.synthesized_beam
cpp_sb.argtypes = [ndpointer(ctypes.c_double), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_ushort, ctypes.c_ushort, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_uint, ctypes.c_double, ctypes.c_uint, ctypes.c_uint, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

def amf (chord_theta, wavelength, source_theta, source_phi_0, m1, m2, u, delta_tau, time_samples, chord_phi=0):
        mfo = np.empty(u.shape[0])
        
        num_u = u.shape[0]
        u = np.ascontiguousarray(u.flatten()) ##sometimes this can be out of order
        
        #dealing with dithers
        if isinstance(chord_theta,np.float64) or isinstance(chord_theta,float):
            dither_thetas = np.array([chord_theta])
        elif isinstance(chord_theta, np.ndarray):
            dither_thetas = chord_theta
        else:
        	raise TypeError("chord_theta is unexpected type")
        	return None
        
        cpp_amf(dither_thetas, chord_phi, wavelength, source_theta, source_phi_0, m1, m2, u, num_u, delta_tau, time_samples, dither_thetas.shape[0], mfo)
        
        return mfo

def amf_su (chord_theta, wavelength, source_theta, source_phi_0, m1, m2, u, delta_tau, time_samples):
    #dealing with dithers
    if isinstance(chord_theta,np.float64) or isinstance(chord_theta,float):
        dither_thetas = np.array([chord_theta])
    elif isinstance(chord_theta, np.ndarray):
        dither_thetas = chord_theta
    
    return cpp_amf_su(dither_thetas, wavelength, source_theta, source_phi_0, m1, m2, u, delta_tau, time_samples, dither_thetas.shape[0])

def correlation_coefficient (u1, u2, chord_theta, wavelength, m1, m2, delta_tau, time_samples):
    #dealing with dithers
    if isinstance(chord_theta,np.float64) or isinstance(chord_theta,float):
        dither_thetas = np.array([chord_theta])
    elif isinstance(chord_theta, np.ndarray):
        dither_thetas = chord_theta
    
    #we have to convert u1 into phi and theta
    u1theta, u1phi = vec2ang(u1)
    return cpp_amf_su(dither_thetas, wavelength, u1theta, u1phi, m1, m2, u2, delta_tau, time_samples, dither_thetas.shape[0])
    
def synthesized_beam (chord_theta, wavelength, source_theta, source_phi_0, m1, m2, u, delta_tau, time_samples, chord_phi=0):
    #chord_theta : CHORD's angle away from the North Pole in radians. If this is a numpy array, this adds dithering. It can just be a float if you don't want dithering.
    #wavelength : the wavelength
    #source_theta, source_phi_0 : the source's true location when time starts (in radians, and theta is away from the North Pole, phi is away from CHORD)
    #m1, m2 : Number of dishes. m1 is north-south and m2 is east-west.
    #u : A vector of unit vectors for every pixel that you want. You can get this with healpy functions.
    # delta_tau : time difference between each time step in seconds.
    # time_samples : number of time samples
    
    sb = np.empty(u.shape[0])
    num_u = u.shape[0]
    u = np.ascontiguousarray(u.flatten()) ##sometimes this can be out of order
    
    #dealing with dithers
    if isinstance(chord_theta,np.float64) or isinstance(chord_theta,float):
        dither_thetas = np.array([chord_theta])
    elif isinstance(chord_theta, np.ndarray):
        dither_thetas = chord_theta
    
    cpp_sb(dither_thetas, chord_phi, wavelength, source_theta, source_phi_0, m1, m2, u, num_u, delta_tau, time_samples, dither_thetas.shape[0],sb)
    return sb

#def amf_autosampling (u, source_phi, source_theta, dec_separation, wavelength, m1, m2):
#    beam_angle = 6/wavelength
#    total angle = 5*beam_angle
