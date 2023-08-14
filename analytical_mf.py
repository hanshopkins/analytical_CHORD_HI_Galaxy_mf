import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import sys
from scipy.special import jv

chord_theta = np.deg2rad(40.8)
chord_phi = 0
D = 6 #meters
L = 7 #meters
speedoflight = 3E8
wavelength = speedoflight/1000E6

source_theta = np.deg2rad(41)
def source_phi (t): return (2*np.pi/86400*t + np.pi - 0.0087)%(2*np.pi) - np.pi;

npix_width, npix_height, phi_lower, phi_upper, theta_lower, theta_upper, m = 1024,512, np.deg2rad(-10), np.deg2rad(10), np.deg2rad(46), np.deg2rad(37), 22 #m is the number of dishes per row and column, so 8 or 22

def ang_2_3vec(phi, theta):
    return np.asarray([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T

chord_pointing = ang_2_3vec(chord_phi, chord_theta)

def B_sq (x):
    return (2*jv(1,np.pi*D*x/wavelength)/(np.pi*D*x/wavelength))**2


#def calc_denominator ():
#    phipal = (phi_upper-phi_lower)/npix_width #stands for phi pixel angular length
#    phi_values = np.linspace(phi_lower+phipal/2.0, phi_upper-phipal/2.0, npix_width)
#    
#    z_lower = np.cos(theta_lower)
#    z_upper = np.cos(theta_upper)
#    zpl = (z_upper-z_lower)/npix_height #stands for z pixel length
#    z_values = np.linspace(z_lower+zpl/2.0, z_upper - zpl/2.0, npix_height)
#    theta_values = np.arccos(z_values)
#    
#    phi_grid, theta_grid = np.meshgrid(phi_values, theta_values)
#    
#    pi = ang_2_3vec(phi_grid.flatten(), theta_grid.flatten())
#    
#    dir1_proj_vec = ang_2_3vec(chord_phi, chord_theta + np.pi/2) #north/south chord direction
#    dir2_proj_vec = ang_2_3vec(chord_phi+np.pi/2, chord_theta) #east/west chord direction
#    
#    cdir1 = 2*np.pi*L/wavelength*np.dot(pi,dir1_proj_vec) #4 comes from 2 copies of pi
#    cdir2 = 2*np.pi*L/wavelength*np.dot(pi,dir2_proj_vec)
#    
#    pi_eq_pi_p = np.linalg.norm(pi - chord_pointing, axis = 1) == 0
#    Bsq_values = np.where(pi_eq_pi_p, 1, B_sq(np.arccos(np.tensordot(pi, chord_pointing, axes = 1))))
#    
#    return np.abs(np.sin(cdir1*m)/np.sin(cdir1) * np.sin(cdir2*m)/np.sin(cdir2) * Bsq_values).reshape(npix_height, npix_width) #no squared because the denominator gets square rooted

def calc_denominator ():
    phipal = (phi_upper-phi_lower)/npix_width #stands for phi pixel angular length
    phi_values = np.linspace(phi_lower+phipal/2.0, phi_upper-phipal/2.0, npix_width)

    z_lower = np.cos(theta_lower)
    z_upper = np.cos(theta_upper)
    zpl = (z_upper-z_lower)/npix_height #stands for z pixel length
    z_values = np.linspace(z_lower+zpl/2.0, z_upper - zpl/2.0, npix_height)
    theta_values = np.arccos(z_values)
    
    phi_grid, theta_grid = np.meshgrid(phi_values, theta_values)
    
    pi = ang_2_3vec(phi_grid.flatten(), theta_grid.flatten())

    pi_eq_pi_p = np.linalg.norm(pi - chord_pointing, axis = 1) <= 1E-12
    Bsq_values = np.where(pi_eq_pi_p, 1, B_sq(np.arccos(np.tensordot(pi, chord_pointing, axes = 1))))
    
    return (m**2 * Bsq_values).reshape(npix_height, npix_width)

def calc_numerator (t):
    phipal = (phi_upper-phi_lower)/npix_width #stands for phi pixel angular length
    phi_values = np.linspace(phi_lower+phipal/2.0, phi_upper-phipal/2.0, npix_width)
    
    z_lower = np.cos(theta_lower)
    z_upper = np.cos(theta_upper)
    zpl = (z_upper-z_lower)/npix_height #stands for z pixel length
    z_values = np.linspace(z_lower+zpl/2.0, z_upper - zpl/2.0, npix_height)
    theta_values = np.arccos(z_values)
    
    phi_grid, theta_grid = np.meshgrid(phi_values, theta_values)
    
    pi = ang_2_3vec(phi_grid.flatten(), theta_grid.flatten())
    
    dir1_proj_vec = ang_2_3vec(chord_phi, chord_theta + np.pi/2) #north/south chord direction
    dir2_proj_vec = np.cross(dir1_proj_vec, chord_pointing) #east/west chord direction
    
    source_pointing = ang_2_3vec(source_phi(t), source_theta)
    cdir1 = np.pi*L/wavelength*np.dot(-pi+source_pointing,dir1_proj_vec)
    cdir2 = np.pi*L/wavelength*np.dot(-pi+source_pointing,dir2_proj_vec)
    
    pi_eq_pi_p = np.linalg.norm(pi - chord_pointing, axis = 1) == 0
    Bsq_values = np.where(pi_eq_pi_p, 1, B_sq(np.arccos(np.tensordot(pi, chord_pointing, axes = 1))))
    
    pi_s_eq_pi_p = np.linalg.norm(source_pointing - chord_pointing) == 0
    if pi_s_eq_pi_p:
        Bsq_source_value = 1
    else:
        Bsq_source_value = B_sq(np.arccos(np.dot(source_pointing, chord_pointing)))
    
    return m**4 * Bsq_source_value * (np.sin(cdir1*m)**2/np.sin(cdir1)**2 * np.sin(cdir2*m)**2/np.sin(cdir2)**2 * Bsq_values).reshape(npix_height, npix_width)

if __name__ == "__main__":
    denom = calc_denominator()

    fig1 = plt.figure()
    plt.imshow(denom, origin = "upper", extent = (phi_lower, phi_upper, theta_upper, theta_lower), interpolation = "none")
    plt.title("Denominator")
    plt.xlabel(r"$\Phi$")
    plt.ylabel(r"$\theta$")
    plt.plot(chord_phi, chord_theta, 'rx', ms=15, label="Chord pointing")
    #plt.colorbar()
    plt.legend()

    if len(sys.argv) > 1 and sys.argv[1] == "--movie":
        #times = np.linspace(0.0,24*3600, 200) [:-1]
        times = np.linspace(0.0, 360, 200) [:-1]

        print("Calculating numerator and matched filters for all times:")
        #vmin_num, vmax_num, vmin_mf, vmax_mf = None, None, None, None
        num = np.empty([times.shape[0],npix_height, npix_width])
        for i in range(times.shape[0]):
            t = times[i]
            num[i] = calc_numerator(t)
            print("\x1b[2K",str((i+1)/times.shape[0] * 100)+"% complete", end='\r')
        
        vmin_mf = np.min(num/denom)
        vmax_mf = np.max(num/denom)
        
        print("\nPlotting...")
        for i in range(times.shape[0]):
            t = times[i]
            
            #fig2 = plt.figure()
            #plt.imshow(num, origin = "upper", extent = (phi_lower, phi_upper, theta_upper, theta_lower), interpolation = "none")
            #plt.title("Numerator")
            #plt.xlabel(r"$\Phi$")
            #plt.ylabel(r"$\theta$")
            #plt.plot(chord_phi, chord_theta, 'rx', ms=15, label="Chord pointing")
            #if (source_phi(t) >= phi_lower and source_phi(t) <= phi_upper): plt.plot(source_phi(t), source_theta, 'bx', ms=15, label="Source location")
            #plt.colorbar()
            #plt.legend()
            #plt.savefig("num"+str(i)+".png")
            #plt.close()

            fig3 = plt.figure()
            plt.imshow(num[i]/denom, origin = "upper", extent = (phi_lower, phi_upper, theta_upper, theta_lower), vmin = vmin_mf, vmax = vmax_mf, interpolation = "none")
            plt.title("Matched filter")
            plt.xlabel(r"$\Phi$")
            plt.ylabel(r"$\theta$")
            plt.plot(chord_phi, chord_theta, 'rx', ms=15, label="Chord pointing")
            if (source_phi(t) >= phi_lower and source_phi(t) <= phi_upper):  plt.plot(source_phi(t), source_theta, 'bx', ms=15, label="Source location")
            #plt.colorbar()
            plt.legend()
            plt.savefig("mf"+str(i)+".png")
            plt.close()
            
            print("\x1b[2K",str((i+1)/times.shape[0] * 100)+"% complete", end='\r')

        print("\nSaving gifs...")
        #with iio.imopen('numerator.gif', "w") as gif:
        #    for i in range(times.shape[0]):
        #        filename = "num"+str(i)+".png"
        #        image = iio.imread(filename)
        #        gif.write(image, loop=0)
        #        os.remove(filename)
        with iio.imopen('matched filter.gif', "w") as gif:
            for i in range(times.shape[0]):
                filename = "mf"+str(i)+".png"
                image = iio.imread(filename)
                gif.write(image, loop=0)
                os.remove(filename)
                print("\x1b[2K",str((i+1)/times.shape[0] * 100)+"% complete", end='\r')
        print("\n")
    else:
        num = calc_numerator(0)

        fig2 = plt.figure()
        plt.imshow(num, origin = "upper", extent = (phi_lower, phi_upper, theta_upper, theta_lower), interpolation = "none")
        plt.title("Numerator")
        plt.xlabel(r"$\Phi$")
        plt.ylabel(r"$\theta$")
        plt.plot(chord_phi, chord_theta, 'rx', ms=15, label="Chord pointing")
        if (source_phi(0) >= phi_lower and source_phi(0) <= phi_upper): plt.plot(source_phi(0), source_theta, 'bx', ms=15, label="Source location")
        #plt.colorbar()
        plt.legend()
        
        fig3 = plt.figure()
        plt.imshow(num/denom, origin = "upper", extent = (phi_lower, phi_upper, theta_upper, theta_lower), interpolation = "none")
        plt.title("Matched filter")
        plt.xlabel(r"$\Phi$")
        plt.ylabel(r"$\theta$")
        plt.plot(chord_phi, chord_theta, 'rx', ms=15, label="Chord pointing")
        if (source_phi(0) >= phi_lower and source_phi(0) <= phi_upper):  plt.plot(source_phi(0), source_theta, 'bx', ms=15, label="Source location")
        #plt.colorbar()
        plt.legend()
        
        if len(sys.argv) > 1 and sys.argv[1] == "--pbenv": #plotting the primary meab envelope to show the dropoff aliasing should have
            z_lower = np.cos(theta_lower)
            z_upper = np.cos(theta_upper)
            zpl = (z_upper-z_lower)/npix_height #stands for z pixel length
            z_values = np.linspace(z_lower+zpl/2.0, z_upper - zpl/2.0, npix_height)
            theta_values = np.arccos(z_values)
            pixel_row_idx = np.searchsorted(theta_values, source_theta)
            
            fig4 = plt.figure()
            plt.plot(np.linspace(phi_lower, phi_upper, num.shape[1]), (num/denom)[pixel_row_idx])
            plt.title("Slice showing aliasing")
            plt.yscale("log")
            plt.xlabel(r"$\phi$")
        
        plt.show()
