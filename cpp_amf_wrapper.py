import numpy as np
import ctypes
import os
from numpy.ctypeslib import ndpointer

amfLib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "amf.so"))
cpp_amf = amfLib.analytic_matched_filter
cpp_amf.argtypes = [ndpointer(ctypes.c_double), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_ushort, ctypes.c_ushort, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_uint,
                        ctypes.c_double, ctypes.c_uint, ctypes.c_uint, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

cpp_amf_su = amfLib.analytic_matched_filter_single_u
cpp_amf_su.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_ushort, ctypes.c_ushort, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.c_double, ctypes.c_uint, ctypes.c_uint]
cpp_amf_su.restype = ctypes.c_double

cpp_sb = amfLib.analytic_matched_filter
cpp_sb.argtypes = [ndpointer(ctypes.c_double), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_ushort, ctypes.c_ushort, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_uint,
                        ctypes.c_double, ctypes.c_uint, ctypes.c_uint, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

def amf (chord_theta, wavelength, source_theta, source_phi_0, m1, m2, u, delta_tau, time_samples):
        mfo = np.empty(u.shape[0])
        
        num_u = u.shape[0]
        u = np.ascontiguousarray(u.flatten()) ##sometimes this can be out of order
        
        #dealing with dithers
        if isinstance(chord_theta,np.float64) or isinstance(chord_theta,float):
            dither_thetas = np.array([chord_theta])
        elif isinstance(chord_theta, np.ndarray):
            dither_thetas = chord_theta
        
        cpp_amf(dither_thetas, wavelength, source_theta, source_phi_0, m1, m2, u, num_u, delta_tau, time_samples, dither_thetas.shape[0], mfo)
        
        return mfo

def amf_su (chord_theta, wavelength, source_theta, source_phi_0, m1, m2, u, delta_tau, time_samples):
    #dealing with dithers
    if isinstance(chord_theta,np.float64) or isinstance(chord_theta,float):
        dither_thetas = np.array([chord_theta])
    elif isinstance(chord_theta, np.ndarray):
        dither_thetas = chord_theta
    
    return cpp_amf_su(dither_thetas, wavelength, source_theta, source_phi_0, m1, m2, u, delta_tau, time_samples, dither_thetas.shape[0])
    
def synthesized_beam (chord_theta, wavelength, source_theta, source_phi_0, m1, m2, u, delta_tau, time_samples):
    sb = np.empty(u.shape[0])
    num_u = u.shape[0]
    u = np.ascontiguousarray(u.flatten()) ##sometimes this can be out of order
    
    #dealing with dithers
    if isinstance(chord_theta,np.float64) or isinstance(chord_theta,float):
        dither_thetas = np.array([chord_theta])
    elif isinstance(chord_theta, np.ndarray):
        dither_thetas = chord_theta
    
    cpp_sb(dither_thetas, wavelength, source_theta, source_phi_0, m1, m2, u, num_u, delta_tau, time_samples, dither_thetas.shape[0],sb)
    return sb
