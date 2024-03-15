import numpy as np
import matplotlib.pyplot as plt
from cpp_amf_wrapper import amf
from numpy.linalg import inv
from math import floor, ceil

def get_integer_ticks (low, high, separation):
    return np.arange(np.ceil(low/separation)*separation, (np.floor(high/separation)+1)*separation, separation)

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
    
def peak_at_pos_for_testing(u, source_theta, source_phi):
    u_source = ang_2_3vec(source_phi,source_theta)
    alpha = np.arcsin(np.sqrt(np.sum(np.cross(u_source,u)**2,axis=-1)))
    return np.exp(-(alpha/0.003490658503988659)**2)

def tan_plane_plot (base_theta, base_phi, chord_theta, nx, ny, extent1, extent2, wavelength, source_theta, source_phi_0, m1, m2, delta_tau, time_samples, title, gridlines=False, plot_chord=False, plot_source=False):
    testvecs = np.empty([nx,ny,3])
    basevec = ang_2_3vec(base_phi,base_theta)
    v1 = ang_2_3vec(base_phi,base_theta - np.pi/2)
    v2 = np.cross(v1, basevec)
    
    xls = np.linspace(-1,1,nx) * np.tan(extent1)
    yls = np.linspace(-1,1,ny) * np.tan(extent2)
    testvecs = (basevec[np.newaxis, np.newaxis, :]
      + v1[np.newaxis, np.newaxis, :] * yls[:, np.newaxis, np.newaxis]
      + v2[np.newaxis, np.newaxis, :] * xls[np.newaxis, :, np.newaxis])

    #normalizing
    norms = np.linalg.norm(testvecs, axis=2)
    np.divide(testvecs[:,:,0],norms, testvecs[:,:,0])
    np.divide(testvecs[:,:,1],norms, testvecs[:,:,1])
    np.divide(testvecs[:,:,2],norms, testvecs[:,:,2])
    
    values = amf (chord_theta, wavelength, source_theta, source_phi_0, m1, m2, testvecs.reshape([nx*ny,3]), delta_tau, time_samples).reshape([ny,nx])
    
    fig = plt.figure()
    plt.imshow(values, origin="lower", interpolation = "none", extent = (-1,1,-1,1))
    plt.title(title)
    #plt.colorbar()
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    
    if gridlines or plot_chord or plot_source:
        chob = np.linalg.inv(np.hstack((v1[np.newaxis].T, v2[np.newaxis].T, basevec[np.newaxis].T)))#change of basis matrix. The negative is to fix the orientations being opposite.
        def ang_2_tpp_coords (theta, phi): #let's define a function that can convert between spherical coordinates and tangent plane projection coodinates
            #the formula is
            # sec(alpha)(ch_of_basis_mat @ (cos phi sin theta, sin phi sin theta, cos(theta)), where alpha is the angular distance
            tempvec = np.asarray([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)*np.ones(np.atleast_1d(phi).shape[0])])
            secalpha = 1/(np.dot(tempvec.T,basevec))
            tpp_coords_unscaled = secalpha*(chob @ tempvec)[:2]
            return tpp_coords_unscaled[1]/np.tan(extent1), tpp_coords_unscaled[0]/np.tan(extent2) #x,y
        
        if plot_chord:
            x,y = ang_2_tpp_coords (chord_theta, 0)
            plt.plot(x, y, 'rx', ms=15, label="CHORD location")
        if plot_source:
            x,y = ang_2_tpp_coords (source_theta, source_phi_0)
            plt.plot(x, y, 'bx', ms=15, label="Source location")
        if gridlines:
            #handling the grid lines
            #hacky way to get
            corner_sph_coords = np.asarray([vec2ang(testvecs[0][0]), vec2ang(testvecs[0][-1]), vec2ang(testvecs[-1][0]), vec2ang(testvecs[-1][-1])])
            high_theta = np.max(corner_sph_coords[:,0])
            low_theta = np.min(corner_sph_coords[:,0])
            high_phi = np.max(corner_sph_coords[:,1])
            low_phi = np.min(corner_sph_coords[:,1])
            #step 1 is to figure out if we are plotting a pole
            pole = base_theta < 2 * extent2
            #(np.sin(high_phi)*np.sin(low_phi)<0) and (np.cos(high_phi)*np.cos(low_phi)<0)
            if pole: #this super breaks it
                e1grid = np.pi
                phitp = np.linspace(-e1grid, e1grid,1000) #phistoplot
                e2grid = high_theta
                thetatp = np.linspace(0, e2grid,10) #thetastoplot
            else:
                e1grid = (high_phi-low_phi)#/2
                phitp = np.linspace(base_phi-e1grid, base_phi+e1grid,100) #phistoplot
                e2grid = (high_theta-low_theta)/2
                thetatp = np.linspace(base_theta-e2grid, base_theta+e2grid,100) #thetastoplot

            #next figuring out the line spacing. We don't want to plot too many gridlines.
            deg_thresholds = np.array([5,10,25,50,150],dtype=int)
            deg_separations = np.array([1,2,5,10,30,60],dtype=int)
            phi_separation = deg_separations[np.searchsorted(deg_thresholds,np.rad2deg(e1grid))]
            theta_separation = deg_separations[np.searchsorted(deg_thresholds,np.rad2deg(e2grid))]
            phi_ticks = get_integer_ticks(np.rad2deg(phitp[0]), np.rad2deg(phitp[-1]), phi_separation)
            theta_ticks = get_integer_ticks(np.rad2deg(thetatp[0]), np.rad2deg(thetatp[-1]), theta_separation)
            
            def plot_const_RA_gridline (phi, color="grey",label=None):
                tempvec = np.vstack((np.cos(phi)*np.sin(thetatp), np.sin(phi)*np.sin(thetatp), np.cos(thetatp)*np.ones(phitp.shape[0])))
                secalpha = 1/(np.dot(tempvec.T,basevec))
                line_in_tangent_plane_coords_unscaled = secalpha*(chob @ tempvec)[:2]
                #assert(np.all(np.abs(np.dot((line_in_tangent_plane_coords_unscaled - np.array([[0],[0],[1]])).T, np.array([0,0,1]))) < 1e-10))
                #line_in_tangent_plane_coords_unscaled[:2]
                #the imshow goes from -1 to 1, so we need to scale these
                line_in_tangent_plane_coords = np.empty([2,thetatp.shape[0]])
                line_in_tangent_plane_coords[0] = line_in_tangent_plane_coords_unscaled[0]/np.tan(extent2) #the top of this array should be the ys and the bottom are the xs
                line_in_tangent_plane_coords[1] = line_in_tangent_plane_coords_unscaled[1]/np.tan(extent1)
                plt.plot(line_in_tangent_plane_coords[1], line_in_tangent_plane_coords[0], color=color,linestyle=(0, (3, 10)),label=label)
                #we want to find if it crosses the boundary, and if so, write a tick marker
                cross = np.searchsorted(line_in_tangent_plane_coords[0][::-1], -1)
                if cross < line_in_tangent_plane_coords.shape[1] and line_in_tangent_plane_coords[1][::-1][cross] > -1 and line_in_tangent_plane_coords[1][::-1][cross] < 1:
                    plt.text(line_in_tangent_plane_coords[1][::-1][cross]-0.02,-1.07, "{degvalue:n}".format(degvalue=np.rad2deg(phi)))
            
            for phi_deg in phi_ticks:
                phi = np.deg2rad(phi_deg)
                plot_const_RA_gridline (phi)
            for theta_deg in theta_ticks: #plot lines of constant dec
                theta = np.deg2rad(theta_deg)
                tempvec = np.vstack((np.cos(phitp)*np.sin(theta), np.sin(phitp)*np.sin(theta), np.cos(theta)*np.ones(phitp.shape[0])))
                secalpha = 1/(np.dot(tempvec.T,basevec))
                line_in_tangent_plane_coords_unscaled = secalpha * (chob @ tempvec)[:2]
                #the imshow goes from -1 to 1, so we need to scale these
                line_in_tangent_plane_coords = np.empty([2,phitp.shape[0]])
                line_in_tangent_plane_coords[0] = line_in_tangent_plane_coords_unscaled[0]/np.tan(extent2) #the top of this array should be the ys and the bottom are the xs
                line_in_tangent_plane_coords[1] = line_in_tangent_plane_coords_unscaled[1]/np.tan(extent1)
                plt.plot(line_in_tangent_plane_coords[1], line_in_tangent_plane_coords[0], color="grey",linestyle=(0, (3, 10)))
                
                #we want to find if it crosses the boundary, and if so, write a tick marker
                cross = np.searchsorted(line_in_tangent_plane_coords[1], -1)
                if cross < line_in_tangent_plane_coords.shape[1] and line_in_tangent_plane_coords[0][cross] < 1 and line_in_tangent_plane_coords[0][cross] > -1:
                    plt.text(-1.14, line_in_tangent_plane_coords[0][cross], "{degvalue:n}".format(degvalue=theta_deg))
            #plt.text(-0.8,-1.1,"Gridline separations: "+str(phi_separation)+" deg (RA), "+str(theta_separation)+" deg (dec)")
        
    plt.show()
                

if __name__ == "__main__":
    tan_plane_plot (np.deg2rad(1), 0, np.deg2rad(0), 300,300, np.deg2rad(3), np.deg2rad(3), 0.21, np.deg2rad(0), 0, 24, 22, 24.0*3600/1000, 1, "test tan plane plot", gridlines=True)
    plt.show()
