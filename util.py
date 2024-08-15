import numpy as np

def rotate_by_phi (v,phi):
    r = np.array([[np.cos(phi), -np.sin(phi), 0],[np.sin(phi), np.cos(phi), 0],[0,0,1]])
    if v.ndim == 1:
        return r@v
    elif v.ndim == 2:
        return (r@v.T).T
    else:
        print("rotate_by_phi formatted v wrong (wrong ndim)")

def rotate_arbitrary_axis (v, phi, a):
    #a is the axis
    #source https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    ax = np.asarray([
    [0, -a[2], a[1]],
    [a[2],0,-a[0]],
    [-a[1],a[0],0]])
    r = np.cos(phi)*np.identity(3) + np.sin(phi)*ax + (1-np.cos(phi))*np.outer(a,a)
    if v.ndim == 1:
        return r@v
    elif v.ndim == 2:
        return (r@v.T).T
    else:
        print("rotate_by_phi formatted v wrong (wrong ndim)")

def peakfind (f, guess_low, guess_high, tol=1e-5):
    h = guess_high
    l = guess_low
    while True:
        midpoint = (h+l)/2.0
        width = h-l
        if width < tol:
            return midpoint
        deriv = (f(midpoint+width/10.0)-f(midpoint-width/10.0))/(2*width/10)
        if deriv == 0:
            return midpoint
        elif deriv > 0:
            l = midpoint
        else:
            h = midpoint
