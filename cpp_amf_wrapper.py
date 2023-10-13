import numpy as np
import ctypes
import os
from numpy.ctypeslib import ndpointer

amfLib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "amf.so"))
cpp_amf = amfLib.analytic_matched_filter
cpp_amf.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_ushort, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_uint,
                        ctypes.c_double, ctypes.c_uint, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

def amf (chord_theta, wavelength, source_theta, source_phi_0, m, u, delta_tau, time_samples):
        mfo = np.empty(u.shape[0])
        
        num_u = u.shape[0]
        u = np.ascontiguousarray(u.flatten()) ##sometimes this can be out of order
        cpp_amf(chord_theta, wavelength, source_theta, source_phi_0, m, u, num_u, delta_tau, time_samples, mfo)
        
        return mfo
