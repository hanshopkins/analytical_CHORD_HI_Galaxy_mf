import numpy as np
import matplotlib.pyplot as plt
from cpp_amf_wrapper import amf
from numpy.linalg import inv
from math import floor, ceil

def ang_2_3vec(phi, theta):
    return np.asarray([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T

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

def tan_plane_plot (base_theta, base_phi, chord_theta, nx, ny, extent1, extent2, wavelength, source_theta, source_phi_0, m1, m2, delta_tau, time_samples, title, gridlines=False):
    testvecs = np.empty([nx,ny,3])
    basevec = ang_2_3vec(base_phi,base_theta)
    v1 = ang_2_3vec(base_phi,base_theta - np.pi/2)
    v2 = np.cross(basevec, v1)
    
    xls = np.linspace(-1,1,nx) * np.tan(extent1)
    yls = np.linspace(-1,1,ny) * np.tan(extent2)
    testvecs = (basevec[np.newaxis, np.newaxis, :]
      + v1[np.newaxis, np.newaxis, :] * yls[:, np.newaxis, np.newaxis]
      + v2[np.newaxis, np.newaxis, :] * xls[np.newaxis, :, np.newaxis])
    
    #testvecs = np.transpose(np.tile(np.tensordot(yls,v1, axes=0), (nx,1,1)), (1,0,2)) + np.tile(np.tensordot(xls,v2, axes=0), (ny,1,1)) + chvec

    #normalizing
    norms = np.linalg.norm(testvecs, axis=2)
    np.divide(testvecs[:,:,0],norms, testvecs[:,:,0])
    np.divide(testvecs[:,:,1],norms, testvecs[:,:,1])
    np.divide(testvecs[:,:,2],norms, testvecs[:,:,2])
    
    values = amf (chord_theta, wavelength, source_theta, source_phi_0, m1, m2, testvecs.reshape([nx*ny,3]), delta_tau, time_samples).reshape([ny,nx])
    
    fig = plt.figure()
    plt.imshow(values, origin = "upper", interpolation = "none",
        extent = (-1,1,-1,1))
        #extent = (np.rad2deg(base_phi-extent1), np.rad2deg(base_phi-extent1), np.rad2deg(base_theta-extent2), np.rad2deg(base_theta+extent2)))
    plt.title(title)
    plt.colorbar()
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    
    if gridlines:
        #handling the grid lines
        #hacky way to get
        corner_sph_coords = np.asarray([vec2ang(testvecs[0][0]), vec2ang(testvecs[0][-1]), vec2ang(testvecs[-1][0]), vec2ang(testvecs[-1][-1])])
        high_theta = np.max(corner_sph_coords[:,0])
        low_theta = np.min(corner_sph_coords[:,0])
        high_phi = np.max(corner_sph_coords[:,1])
        low_phi = np.min(corner_sph_coords[:,1])
        print(np.rad2deg(corner_sph_coords))
        #step 1 is to figure out if we are plotting a pole
        pole = (np.sin(high_phi)*np.sin(low_phi)<0) and (np.cos(high_phi)*np.cos(np.sin(low_phi))<0)
        if pole: #this super breaks it
            e1grid = 1
            phitp = np.linspace(-e1grid, e1grid,1000) #phistoplot
            e2grid = (high_theta+low_theta)/2
            thetatp = np.linspace(base_theta-e2grid, base_theta+e2grid,1000) #thetastoplot
        else:
            e1grid = (high_phi-low_phi)/2
            phitp = np.linspace(base_phi-e1grid, base_phi+e1grid,100) #phistoplot
            e2grid = (high_theta-low_theta)/2
            thetatp = np.linspace(base_theta-e2grid, base_theta+e2grid,100) #thetastoplot

        #next figuring out the line spacing. We don't want to plot too many gridlines.
        deg_thresholds = np.array([5,10,25,50,150],dtype=int)
        deg_separations = np.array([1,2,5,10,30,60],dtype=int)
        phi_separation = deg_separations[np.searchsorted(deg_thresholds,np.rad2deg(e1grid))]
        theta_separation = deg_separations[np.searchsorted(deg_thresholds,np.rad2deg(e2grid))]
        phi_ticks = np.arange(np.ceil(np.rad2deg(low_phi)/phi_separation)*phi_separation,(np.floor(np.rad2deg(high_phi)/phi_separation)+1)*phi_separation,phi_separation)
        theta_ticks = np.arange(np.ceil(np.rad2deg(low_theta)/theta_separation)*theta_separation,(np.floor(np.rad2deg(high_theta)/theta_separation)+1)*theta_separation,theta_separation)
        
        #the formula for the positions of the gridlines is
        # sec(alpha)(ch_of_basis_mat @ (cos phi sin theta, sin phi sin theta, cos(theta)), where alpha is the 
        #angular distance cos^-1(sin phi_base sin phi + cos phi_base cos phi cos(theta_base - theta))
        chob = np.linalg.inv(np.hstack((v1[np.newaxis].T, v2[np.newaxis].T, basevec[np.newaxis].T)))#change of basis matrix
        for phi_deg in phi_ticks: # range(ceil(np.rad2deg(base_phi-e1grid)), floor(np.rad2deg(base_phi+e1grid)) + 1):
            phi = np.deg2rad(phi_deg)
            secalpha = 1/(np.sin(base_phi)*np.sin(phi) + np.cos(base_phi)*np.cos(phi)*np.cos(base_theta-thetatp))
            tempvec = np.vstack((np.cos(phi)*np.sin(thetatp), np.sin(phi)*np.sin(thetatp), np.cos(thetatp)))
            line_in_tangent_plane_coords_unscaled = secalpha * (chob @ tempvec)[:2]
            #the imshow goes from -1 to 1, so we need to scale these
            line_in_tangent_plane_coords = np.empty([2,thetatp.shape[0]])
            line_in_tangent_plane_coords[0] = line_in_tangent_plane_coords_unscaled[0]/np.tan(extent2) #the top of this array should be the ys and the bottom are the xs
            line_in_tangent_plane_coords[1] = line_in_tangent_plane_coords_unscaled[1]/np.tan(extent1)
            plt.plot(line_in_tangent_plane_coords[1], line_in_tangent_plane_coords[0],color="grey",linestyle=(0, (3, 10)))
        for theta_deg in theta_ticks: #range(ceil(np.rad2deg(base_theta-e2grid)), floor(np.rad2deg(base_theta+e2grid)) + 1):
            theta = np.deg2rad(theta_deg)
            secalpha = 1/(np.sin(base_phi)*np.sin(phitp) + np.cos(base_phi)*np.cos(phitp)*np.cos(base_theta-theta))
            tempvec = np.vstack((np.cos(phitp)*np.sin(theta), np.sin(phitp)*np.sin(theta), np.cos(theta)*np.ones(phitp.shape[0])))
            line_in_tangent_plane_coords_unscaled = secalpha * (chob @ tempvec)[:2]
            #the imshow goes from -1 to 1, so we need to scale these
            line_in_tangent_plane_coords = np.empty([2,phitp.shape[0]])
            line_in_tangent_plane_coords[0] = line_in_tangent_plane_coords_unscaled[0]/np.tan(extent2) #the top of this array should be the ys and the bottom are the xs
            line_in_tangent_plane_coords[1] = line_in_tangent_plane_coords_unscaled[1]/np.tan(extent1)
            plt.plot(line_in_tangent_plane_coords[1], line_in_tangent_plane_coords[0],color="grey",linestyle=(0, (3, 10)))
    plt.show()
                

if __name__ == "__main__":
    tan_plane_plot (np.deg2rad(45), 0, np.deg2rad(45), 300,300, np.deg2rad(3), np.deg2rad(3), 0.21, np.deg2rad(45), 0, 24, 22, 24.0*3600/1000, 1, "test tan plane plot", gridlines=True)
    plt.show()
