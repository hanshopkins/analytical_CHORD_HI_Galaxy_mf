import numpy as np
import matplotlib.pyplot as plt
#import healpy as hp
from cpp_amf_wrapper import amf
from cpp_amf_wrapper import amf_su
from util import peakfind, rotate_arbitrary_axis

chord_zenith = np.deg2rad(90-49.320750)

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

def sidelobe_brightness_NS (chord_theta_possibly_dithered, source_theta, source_phi_0, wavelength, delta_tau, time_samples, L1, m1, m2): #deprecated
    if isinstance(chord_theta_possibly_dithered,np.ndarray): #dithering case
        chord_theta = chord_theta_possibly_dithered[0]
    else: #no dithering case
        chord_theta = chord_theta_possibly_dithered
    source_u = ang2vec(source_theta, source_phi_0)
    ##we want to analytically compute where the nearest side lobes are going to be
    ##a few intermediate steps
    ce1 = wavelength/L1 + source_u[0]*np.cos(chord_theta) - source_u[2]*np.sin(chord_theta)
    ce2 = ce1**2 + (source_u[1]**2-1)*np.cos(chord_theta)**2
    alias_z = -ce1*np.sin(chord_theta) + np.sqrt(ce1**2*np.sin(chord_theta)**2 -ce2)
    #
    
    test_u = np.array([np.sqrt(1-source_u[1]**2-alias_z**2), source_u[1], alias_z])
    #test_u = ang2vec(source_theta+wavelength/7.0,source_phi_0)
    #print("sidelobe spot: ", np.rad2deg(hp.pixelfunc.vec2ang(test_u)[0][0]))
    
    #print("Source theta:", source_theta, "Chord theta: ", chord_theta, "Test theta:", hp.pixelfunc.vec2ang(test_u)[0][0])
    
    mfo_test = amf_su(chord_theta_possibly_dithered, wavelength, source_theta, source_phi_0, m1, m2, test_u, delta_tau, time_samples)
    #mfo_source = amf_su(chord_theta, wavelength, source_theta, source_phi_0, m, source_u, delta_tau, time_samples)
    #print("mfo_test:",mfo_test,"mfo_source",mfo_source)
    return mfo_test

def approxAliasNorthu (chord_theta, u_s, wavelength, phi=None):
    #there's two situations to call this function for. If we only have one time sample, or if the source is well-sampled. Let's say that a phi (being CHORD phi) is provided when there is only one sample
    if phi == None: 
        #in this case, if the source is well sampled, it should pass over CHORD evently and we expect the aliases to occur at the place where CHORD's phi lines up perfectly
        #we need to get the phi from u_s:
        phi = vec2phi(u_s)

    #creating the three basis vectors
    e_3 = ang2vec(chord_theta, phi)
    e_1 = ang2vec(chord_theta+np.pi/2, phi)
    e_2 = np.cross(e_1, e_3)
    
    L1 = 8.5*np.cos(np.abs(chord_theta-chord_zenith))
    t1 = np.dot(e_1, u_s - wavelength/L1 * e_1)
    t2 = np.dot(e_2, u_s)
    t3 = np.sqrt(1-t1**2-t2**2)
    return e_1*t1 + e_2*t2 + e_3*t3
    
def approxAliasSouthu (chord_theta, u_s, wavelength, phi=None):
    #there's two situations to call this function for. If we only have one time sample, or if the source is well-sampled. Let's say that a phi (being CHORD phi) is provided when there is only one sample
    if phi == None: 
        #in this case, if the source is well sampled, it should pass over CHORD evently and we expect the aliases to occur at the place where CHORD's phi lines up perfectly
        #we need to get the phi from u_s:
        phi = vec2phi(u_s)

    #creating the three basis vectors
    e_3 = ang2vec(chord_theta, phi)
    e_1 = ang2vec(chord_theta+np.pi/2, phi)
    e_2 = np.cross(e_1, e_3)
    
    L1 = 8.5*np.cos(np.abs(chord_theta-chord_zenith))
    t1 = np.dot(e_1, u_s + wavelength/L1 * e_1)
    t2 = np.dot(e_2, u_s)
    t3 = np.sqrt(1-t1**2-t2**2)
    return e_1*t1 + e_2*t2 + e_3*t3
    
def approxAliasEastu (chord_theta, u_s, wavelength, phi=None):
    #unfortunately, the correct chord phi is the phi of the alias, which we don't know.
    if phi==None:
        #so for the phi=none case, we'll procede with 1 iteration and that should be okay
        #we need to get the phi from u_s:
        phi = vec2phi(u_s)
        newguess = approxAliasEastu (chord_theta, u_s, wavelength, phi=phi)
        phi = vec2phi(newguess)

    #creating the three basis vectors
    e_3 = ang2vec(chord_theta, phi)
    e_1 = ang2vec(chord_theta+np.pi/2, phi)
    e_2 = np.cross(e_1, e_3)
    
    t1 = np.dot(e_1, u_s)
    t2 = np.dot(e_2, u_s - wavelength/6.3 * e_2)
    t3 = np.sqrt(1-t1**2-t2**2)
    return e_1*t1 + e_2*t2 + e_3*t3

def AliasNorth (chord_theta_possibly_dithered, source_theta, source_phi_0, wavelength, delta_tau, time_samples, m1, m2, tol=1e-5, return_extra=False):
    if isinstance(chord_theta_possibly_dithered,np.ndarray):
        chord_theta = chord_theta_possibly_dithered[0] #for dithering let's assume it's where the first chord theta would predict
    else:
        chord_theta = chord_theta_possibly_dithered
    source_u = ang2vec(source_theta, source_phi_0)
    aau = approxAliasNorthu(chord_theta, source_u, wavelength)
    
    #step 1 is we sample a bunch of points next to the approx alias
    alw = wavelength/(24*8.5)#approximate lobe width
    ats = np.arcsin(np.sqrt(np.sum(np.cross(aau, source_u)**2))) #approximate theta separation
    ttc = np.linspace(ats-2*alw, ats+2*alw, 16) #thetas to check
    sample_values = amf(chord_theta_possibly_dithered, wavelength, source_theta, source_phi_0, m1, m2, ang2vec(source_theta-ttc, source_phi_0), delta_tau,time_samples)
    guess_theta = ttc[np.argmax(sample_values)]
    
    def amf_along_line (x):
        u = ang2vec(source_theta-x,source_phi_0)
        return amf_su(chord_theta_possibly_dithered, wavelength, source_theta, source_phi_0, m1, m2, u, delta_tau, time_samples)
    peak_theta = peakfind(amf_along_line, guess_theta-alw/4, guess_theta+alw/4, tol)
    if return_extra:
        return source_theta - peak_theta, amf_along_line(peak_theta), source_theta-ats, alw
    else:
        return amf_along_line(peak_theta)

def AliasSouth (chord_theta_possibly_dithered, source_theta, source_phi_0, wavelength, delta_tau, time_samples, m1, m2, tol=1e-5, return_extra=False):
    if isinstance(chord_theta_possibly_dithered,np.ndarray):
        chord_theta = chord_theta_possibly_dithered[0] #for dithering let's assume it's where the first chord theta would predict
    else:
        chord_theta = chord_theta_possibly_dithered
    source_u = ang2vec(source_theta, source_phi_0)
    aau = approxAliasSouthu(chord_theta, source_u, wavelength)
    
    #step 1 is we sample a bunch of points next to the approx alias
    alw = wavelength/(24*8.5)#approximate lobe width
    ats = np.arcsin(np.sqrt(np.sum(np.cross(aau, source_u)**2))) #approximate theta separation
    ttc = np.linspace(ats-2*alw, ats+2*alw, 16) #thetas to check
    sample_values = amf(chord_theta_possibly_dithered, wavelength, source_theta, source_phi_0, m1, m2, ang2vec(source_theta+ttc, source_phi_0), delta_tau,time_samples)
    guess_theta = ttc[np.argmax(sample_values)]
    
    def amf_along_line (x):
        u = ang2vec(source_theta+x,source_phi_0)
        return amf_su(chord_theta_possibly_dithered, wavelength, source_theta, source_phi_0, m1, m2, u, delta_tau, time_samples)
    peak_theta = peakfind(amf_along_line, guess_theta-alw/4, guess_theta+alw/4, tol)
    if return_extra:
        return source_theta - peak_theta, amf_along_line(peak_theta), source_theta-ats, alw
    else:
        return amf_along_line(peak_theta)
        
def AliasNorthQuick (chord_theta_possibly_dithered, source_theta, source_phi_0, wavelength, delta_tau, time_samples, m1, m2):
    #returns the same thing as aliasnorth but just using the initial estimate
    if isinstance(chord_theta_possibly_dithered,np.ndarray):
        chord_theta = chord_theta_possibly_dithered[0] #for dithering let's assume it's where the first chord theta would predict
    else:
        chord_theta = chord_theta_possibly_dithered
    source_u = ang2vec(source_theta, source_phi_0)
    aau = approxAliasNorthu(chord_theta, source_u, wavelength)
    return amf_su(chord_theta_possibly_dithered, wavelength, source_theta, source_phi_0, m1, m2, aau, delta_tau, time_samples)

def AliasSouthQuick (chord_theta_possibly_dithered, source_theta, source_phi_0, wavelength, delta_tau, time_samples, m1, m2):
    #returns the same thing as aliasnorth but just using the initial estimate
    if isinstance(chord_theta_possibly_dithered,np.ndarray):
        chord_theta = chord_theta_possibly_dithered[0] #for dithering let's assume it's where the first chord theta would predict
    else:
        chord_theta = chord_theta_possibly_dithered
    source_u = ang2vec(source_theta, source_phi_0)
    aau = approxAliasSouthu(chord_theta, source_u, wavelength)
    return amf_su(chord_theta_possibly_dithered, wavelength, source_theta, source_phi_0, m1, m2, aau, delta_tau, time_samples)
        
def SearchOverAngleGrid(center, n, halfwidth, tol, func):
    centertheta, centerphi = vec2ang(center)
    axis1 = ang2vec(centertheta-np.pi/2, centerphi)
    axis2 = np.cross(center,axis1)
    assert(n%2==1) #we want the grid to be odd by odd
    max_ = -1E300
    argmax = None
    for i in range(n):
        rot1 = rotate_arbitrary_axis(center, halfwidth * (i-n//2)/(n//2), axis1)
        for j in range(n):
            rot2 = rotate_arbitrary_axis(rot1, halfwidth * (j-n//2)/(n//2), axis2)
            val = func(rot2)
            if val > max_:
                max_ = val
                argmax = rot2
    if (halfwidth * 2)/n < tol:
        return argmax
    else:
        return SearchOverAngleGrid(argmax, n, halfwidth/n, tol, func)

def AliasEast (chord_theta, source_theta, source_phi_0, wavelength, delta_tau, time_samples, m1, m2, tol=1e-5, return_extra=False):
    source_u = ang2vec(source_theta, source_phi_0)
    aau = approxAliasEastu(chord_theta, source_u, wavelength)
    
    #in this direction we no longer have any symmetry to search along only one line. We have to find a peak via an iterative grid method.
    def func (u):
        return amf_su(chord_theta, wavelength, source_theta, source_phi_0, m1, m2, u, delta_tau, time_samples)
    bestu = SearchOverAngleGrid(aau, 5, 0.2 * np.pi/180, 1e-5, func)
    
    if return_extra: return bestu, amf_su(chord_theta, wavelength, source_theta, source_phi_0, m1, m2, bestu, delta_tau, time_samples)
    else: return amf_su(chord_theta, wavelength, source_theta, source_phi_0, m1, m2, bestu, delta_tau, time_samples)

if __name__ == "__main__":
    if False: #individual
        print(AliasNorth(np.deg2rad(90), np.deg2rad(90), 0, 0.21, 24.0*3600/600, 600, 22, 24, return_extra=True))
    if False:
        ts = np.linspace(1,500, 300, dtype=int)
        ratios = np.empty(300)
        for i in range(ts.shape[0]):
            #good example chord 40.8 deg, source np.deg2rad(41), 0.0087
            ratios[i] = AliasNorth (np.deg2rad(48), np.deg2rad(41), 0, 0.21, 24.0*3600/ts[i], ts[i], 22, 24)
            print("\x1b[2K",str((i+1)/ts.shape[0] * 100)+"% complete", end='\r')
        print("\n")
        plt.plot(ts, ratios)
        plt.title("Alias dimming")
        plt.xlabel("Num time samples")
        plt.ylabel("Dimming ratio")
        plt.show()
    if False:
        thetas = np.linspace(np.pi/2-0.05,np.pi/2+0.05, 300)
        ratios = np.empty(300)
        for i in range(thetas.shape[0]):
            ratios[i] = AliasNorth (thetas[i], thetas[i], 0, 3.0E8/1000E6, 24.0*3600/600, 600, 22, 24)
            print("\x1b[2K",str((i+1)/thetas.shape[0] * 100)+"% complete", end='\r')
        print("\n")
        plt.plot(thetas, ratios)
        plt.title("Alias dimming")
        plt.xlabel(r"$\theta$")
        plt.ylabel("Dimming ratio")
        plt.show()
    if False:
        lambdas = np.linspace(0.21,0.23, 300)
        ratios = np.empty(300)
        for i in range(lambdas.shape[0]):
            ratios[i] = sidelobe_brightness_NS (np.deg2rad(40.8), np.deg2rad(41), 0.0087, lambdas[i], 24.0*3600/2000, 2000, 8.5, 22, 22)
            print("\x1b[2K",str((i+1)/ratios.shape[0] * 100)+"% complete", end='\r')
        print("\n")
        plt.plot(lambdas, ratios)
        plt.title("Alias dimming")
        plt.xlabel("wavelength (cm)")
        plt.ylabel("Dimming ratio")
        plt.show()
    if False: #shows for a source in the bad position
        thetas = np.linspace(0.5,np.pi/2, 300)
        ratios = np.empty(300)
        wavelength = 0.21
        for i in range(thetas.shape[0]):
            ratios[i] = sidelobe_brightness_NS (thetas[i], thetas[i]-wavelength/7.0/2, 0, wavelength, 24.0*3600/2000, 2000, 8.5, 22, 22)
            #print("\x1b[2K",str((i+1)/thetas.shape[0] * 100)+"% complete", end='\r')
        #print("\n")
        plt.plot(thetas, ratios)
        plt.title("Alias dimming")
        plt.xlabel(r"$\theta$")
        plt.ylabel("Dimming ratio")
        plt.show()
    if True: #shows for a source in the bad position with dithers
        chord_theta_base = 90-49.322
        thetas = np.linspace(np.deg2rad(chord_theta_base-29),np.deg2rad(chord_theta_base+29), 300)
        #thetas = np.linspace(-0.1,0.1, 600)
        thetas_dithered = np.empty([300,3])
        thetas_dithered[:,0] = thetas
        thetas_dithered[:,1] = thetas-np.deg2rad(1)
        thetas_dithered[:,2] = thetas-np.deg2rad(1)
        ratios_dither = np.empty(300)
        ratios_nodither = np.empty(300)
        wavelength = 0.21
        nsamples = 1000
        omega = 360.0/(24.0*3600)
        #version in worstcase
        #source_offset = wavelength/8.5/2
        source_offset = np.deg2rad(0.8)
        for i in range(thetas.shape[0]):
            ratios_dither[i] = AliasNorthQuick (thetas_dithered[i], thetas[i]+source_offset, 0, wavelength, 360.0/omega/nsamples, nsamples, 24, 22)
            ratios_nodither[i] = AliasNorthQuick (thetas[i], thetas[i]+source_offset, 0, wavelength, 360.0/omega/nsamples, nsamples, 24, 22)
        plt.plot(90-np.rad2deg(thetas), ratios_dither, label="Dithered")
        plt.plot(90-np.rad2deg(thetas), ratios_nodither, label="No dither")
        plt.legend()
        plt.title("Dec Dependence of Alias Disambiguation")
        plt.xlabel(r"Dec")
        plt.ylabel("North Alias Correlation")
        plt.show()
    if False: #find exactly where peak thetas are
        m1, m2, wavelength, tsamples = 22, 24, 0.21, 300
        thetas = np.linspace(np.deg2rad(15),np.deg2rad(25),1000)#np.linspace(np.deg2rad(42.1),np.deg2rad(42.4),1000)
        chord_theta = np.deg2rad(20)
        source_theta, source_phi_0 = chord_theta, -np.deg2rad(6)
        us = ang2vec(thetas, np.ones(1000)*source_phi_0)
            #one with dithers np.asarray([chord_theta, chord_theta-np.deg2rad(2)])
        ratios = amf(chord_theta, wavelength, source_theta, source_phi_0, m1, m2, us, 2*np.deg2rad(6)/(2*np.pi/(24.0*3600))/tsamples,tsamples)
        plt.plot(np.rad2deg(thetas), ratios)
        plt.title("Sidelobe Peaks")
        plt.xlabel(r"$\theta$")
        plt.ylabel("ratio")
        estim_alias_theta, estim_alias_value, original_estimate, alw = AliasNorth(chord_theta, source_theta, source_phi_0, wavelength, 24.0*3600/tsamples, tsamples, m1, m2, return_extra=True)
        print(estim_alias_theta)
        plt.axvline(np.rad2deg(estim_alias_theta), c="red")
        plt.axvline(np.rad2deg(original_estimate), c="green")
        plt.axvline(np.rad2deg(original_estimate-alw), c="yellow", linestyle="--")
        plt.axvline(np.rad2deg(original_estimate+alw), c="yellow", linestyle="--")
        plt.show()
    if False: #testing differences between the two implementations
        wavelength = 0.21
        theta = 1.56
        u = ang2vec(theta+wavelength/7.0, 0)
        print(amf_su(theta, wavelength, theta, 0, 22, u, 24.0*3600/600, 600))
        print(amf(theta, wavelength, theta, 0, 22, u.reshape([1,3]), 24.0*3600/600,600))
    if False: #compare two points
        wavelength = 0.21
        chord_theta = 1.56
        u1 = ang2vec(chord_theta+wavelength/7.0, 0)
        u2 = ang2vec(chord_theta-wavelength/7.0, 0)
        print("u1: "+str(amf_su(chord_theta, wavelength, chord_theta, 0, 22, u1, 24.0*3600/50, 50)))
        print("u2: "+str(amf_su(chord_theta, wavelength, chord_theta, 0, 22, u2, 24.0*3600/50, 50)))
    if False: #test findAliasNorth
        wavelength=0.21
        timeSamples = 1000
        print(wavelength/8.5, findAliasNorth (np.deg2rad(45), np.deg2rad(45), 0, wavelength, 3600*24.0/timeSamples, timeSamples, 22, 24))
    if False: #looking at dimming as a function of dithering angle
        n = 600
        dangles = np.linspace(-np.deg2rad(4), np.deg2rad(4),n) #dithering angles
        cc = np.empty(n) #correlation coefficients
        chord_theta, source_theta, wavelength = np.deg2rad(90-49.322), np.deg2rad(90-49.322), 0.21
        source_phi_0 = -np.deg2rad(5)
        omega = 2*np.pi/(3600*24)
        time_samples = 500
        delta_tau = 2*(-source_phi_0)/omega/time_samples
        for i in range(n):
            dither_thetas = np.asarray([chord_theta, chord_theta+dangles[i]])
            cc[i] = AliasNorth (dither_thetas, source_theta, source_phi_0, wavelength, delta_tau, time_samples, 24, 22, tol=1e-5, return_extra=False)
        plt.plot(np.rad2deg(dangles), cc)
        plt.title("Dithering Effectiveness")
        plt.xlabel("Dithering angle (deg)")
        plt.ylabel("North sidelobe correlation")
        plt.savefig("dithering_effectiveness")
