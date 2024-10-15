import numpy as np

def ang2vec (theta,phi):
    return np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta), np.cos(theta)]).T

def vec2phi (v):
    if v[0] > 0:
        return np.arctan(v[1]/v[0])
    elif v[0] < 0 and v[1] >= 0:
        return np.arctan(v[1]/v[0]) + np.pi
    elif v[0] < 0 and v[1] < 0:
        return np.arctan(v[1]/v[0]) - np.pi
    elif v[0] == 0 and v[1] > 0:
        return np.pi/2
    elif v[0] == 0 and v[1] < 0:
        return np.pi/2
    else:
        return 0

def vec2ang(v):
    if v[2] > 0:
        theta = np.arctan(np.sqrt(v[0]**2 + v[1]**2)/v[2])
    elif v[2] < 0:
        theta = np.pi + np.arctan(np.sqrt(v[0]**2 + v[1]**2)/v[2])
    else:
        theta = np.pi/2
    
    if v[0] > 0:
        phi = np.arctan(v[1]/v[0])
    elif v[0] < 0 and v[1] >= 0:
        phi = np.arctan(v[1]/v[0]) + np.pi
    elif v[0] < 0 and v[1] < 0:
        phi = np.arctan(v[1]/v[0]) - np.pi
    elif v[0] == 0 and v[1] > 0:
        phi = np.pi/2
    elif v[0] == 0 and v[1] < 0:
        phi = np.pi/2
    else:
        phi = 0
    
    return theta, phi

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
