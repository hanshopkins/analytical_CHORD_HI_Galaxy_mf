import numpy as np

def ang2vec (theta,phi):
    return np.stack(np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta), np.cos(theta)]), axis=-1)

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
        return -np.pi/2
    else:
        return 0

def vec2ang(v):
    if v.ndim == 1 and v.shape[0] == 3:
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
            phi = -np.pi/2
        else:
            phi = 0

        return theta, phi

    elif v.ndim == 2 and v.shape[1] == 3:
        theta = np.full(v.shape[0], np.pi/2)
        phi = np.zeros(v.shape[0])

        theta = np.where(v[:,2] > 0, np.arctan(np.sqrt(v[:,0]**2 + v[:,1]**2)/v[:,2]), theta)
        theta = np.where(v[:,2] < 0, np.pi + np.arctan(np.sqrt(v[:,0]**2 + v[:,1]**2)/v[:,2]), theta)

        phi = np.where((v[:,0] > 0), np.arctan(v[:,1]/v[:,0]), phi)
        phi = np.where(np.logical_and(v[:,0] < 0, v[:,1] >= 0), np.arctan(v[:,1]/v[:,0]) + np.pi, phi)
        phi = np.where(np.logical_and(v[:,0] < 0, v[:,1] < 0), np.arctan(v[:,1]/v[:,0]) - np.pi, phi)
        phi = np.where(np.logical_and(v[:,0] == 0, v[:,1] > 0), np.pi/2, phi)
        phi = np.where(np.logical_and(v[:,0] == 0, v[:,1] < 0), -np.pi/2, phi)

        return theta, phi

    else:
        raise ValueError("Invalid unit vector input")

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

def fitquadratic1d (f0,f1,f2):
	a = 0.5 * (f2 - 2*f1 + f0)
	if (a == 0.0): raise ValueError("fitquadratic1d failed")
	b = f1 - a - f0
	out = -0.5 * b / a
	if (out < 0) or (out > 2): raise ValueError("fitquadratic1d failed")
	return out

def find_peak_3x3 (m, i, j):
	#source for this
	#https://github.com/dstndstn/astrometry.net/blob/main/util/dcen3x3.c#L82

	#this method won't work if we're right on the edge. In practice this shouldn't happen, or we just make the map bigger.
	if (i == 0) or (i + 1 == m.shape[0]) or (j == 0) or (j+1 == m.shape[1]):
		raise ValueError("The center of your peakpixels can't be on the edge.")
	
	mx0 = fitquadratic1d(m[i-1][j-1], m[i-1][j+0], m[i-1][j+1])
	mx1 = fitquadratic1d(m[i+0][j-1], m[i+0][j+0], m[i+0][j+1])
	mx2 = fitquadratic1d(m[i+1][j-1], m[i+1][j+0], m[i+1][j+1])
	my0 = fitquadratic1d(m[i-1][j-1], m[i+0][j-1], m[i+1][j-1])
	my1 = fitquadratic1d(m[i-1][j+0], m[i+0][j+0], m[i+1][j+0])
	my2 = fitquadratic1d(m[i-1][j+1], m[i+0][j+1], m[i+1][j+1])
	
	bx = (mx0 + mx1 + mx2) / 3.
	mx = (mx2 - mx0) / 2.
    
	by = (my0 + my1 + my2) / 3.
	my = (my2 - my0) / 2.
    
	xcen = (mx * (by - my - 1.) + bx) / (1. + mx * my);
	ycen = (xcen - 1.) * my + by;
	#apparently xcen and ycen are between 0 and 2. We want to convert this back to pixel units first.
	
	return xcen, ycen
    
def fitted_peak_3x3 (m,i,j):
	#this method won't work if we're right on the edge. In practice this shouldn't happen, or we just make the map bigger.
	if (i == 0) or (i + 1 == m.shape[0]) or (j == 0) or (j+1 == m.shape[1]):
		raise ValueError("The center of your peakpixels can't be on the edge.")
	
	X = np.empty([9,6])
	x = np.tile(np.array([j-0.5, j+0.5, j+1.5]),3)
	y = np.tile(np.array([i-0.5, i+0.5, i+1.5])[np.newaxis].T,3).flatten()
	X[:,0] = x**2
	X[:,1] = x * y
	X[:,2] = y**2
	X[:,3] = x
	X[:,4] = y
	X[:,5] = 1
	
	Z = m[i-1:i+2,j-1:j+2].flatten()
	
	A = (np.linalg.inv(X.T @ X) @ X.T @ Z.flatten()[np.newaxis].T)[:,0]
	
	peaky = (A[4]/A[1] - A[3]/(2*A[0]))/(A[1]/(2*A[0])-2*A[2]/A[1])
	peakx = (-A[1]*peaky - A[3])/(2*A[0])
	
	xcen = peakx - j + 1
	ycen = peaky - i + 1
	if xcen < -0.5 or xcen > 2.5 or ycen < -0.5 or ycen > 2.5:
		print("Array that we're fitting the peak on:", m[i-1:i+2, j-1:j+2])
		raise ValueError("fitted peak not in the right area. (xcen, ycen) found is ("+str(xcen)+", "+str(ycen)+")")

	return A[0]*peakx**2 + A[1]*peakx*peaky + A[2]*peaky**2 + A[3]*peakx + A[4]*peaky + A[5], xcen, ycen
	
def fitted_peak_rectangle (m):
	X = np.empty([m.size,6])
	x, y = np.meshgrid(np.linspace(0, m.shape[1]-1, m.shape[1]), np.linspace(0, m.shape[0]-1, m.shape[0]))
	x = x.flatten()
	y = y.flatten()
	X[:,0] = x**2
	X[:,1] = x * y
	X[:,2] = y**2
	X[:,3] = x
	X[:,4] = y
	X[:,5] = 1
	
	Z = m.flatten()
	
	A = (np.linalg.inv(X.T @ X) @ X.T @ Z.flatten()[np.newaxis].T)[:,0]
	peaky = (A[4]/A[1] - A[3]/(2*A[0]))/(A[1]/(2*A[0])-2*A[2]/A[1])
	peakx = (-A[1]*peaky - A[3])/(2*A[0])
	return A[0]*peakx**2 + A[1]*peakx*peaky + A[2]*peaky**2 + A[3]*peakx + A[4]*peaky + A[5], peakx, peaky
