import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cpp_amf_wrapper import amf
from cpp_amf_wrapper import synthesized_beam
from numpy.linalg import inv
from math import floor, ceil
from sidelobe_brightness import approxAliasNorthu, approxAliasSouthu, AliasEast, approxAliasEastu
from util import ang2vec, vec2ang

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')

def get_integer_ticks (low, high, separation):
    return np.arange(np.ceil(low/separation)*separation, (np.floor(high/separation)+1)*separation, separation)
    
def peak_at_pos_for_testing(u, source_theta, source_phi):
    u_source = ang2vec(source_theta, source_phi)
    alpha = np.arcsin(np.sqrt(np.sum(np.cross(u_source,u)**2,axis=-1)))
    return np.exp(-(alpha/0.003490658503988659)**2)

def get_tan_plane_pixelvecs (nx,ny, base_theta, base_phi, extent1, extent2):
    testvecs = np.empty([nx,ny,3])
    basevec = ang2vec(base_theta, base_phi)
    v1 = ang2vec(base_theta - np.pi/2, base_phi)
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
    return testvecs, basevec, v1, v2

def tan_plane_plot (base_theta, base_phi, chord_theta, nx, ny, extent1, extent2, wavelength, source_theta, source_phi_0, m1, m2, delta_tau, time_samples, title, gridlines=False, plot_chord=False, plot_source=False, axis_labels=True, mode="matched filter", chord_phi=0,vmax=None, log=False, colorbar=False, highlight_aliases = False, cmap="Greys"):
    #design idea for plot chord is it's "point" or True for a single point and "line" for a gridline at CHORD
    testvecs, basevec, v1, v2 = get_tan_plane_pixelvecs(nx,ny,base_theta,base_phi,extent1,extent2)
    
    if mode == "matched filter":
        values = amf (chord_theta, wavelength, source_theta, source_phi_0, m1, m2, testvecs.reshape([nx*ny,3]), delta_tau, time_samples, chord_phi=chord_phi).reshape([ny,nx])
    elif mode == "synthesized beam":
        values = synthesized_beam (chord_theta, wavelength, source_theta, source_phi_0, m1, m2, testvecs.reshape([nx*ny,3]), delta_tau, time_samples, chord_phi=chord_phi).reshape([ny,nx])
    else:
        print("Invalid mode. Options are 'matched filter' or 'synthesized beam'")
        assert(1==0)
    
    if log:
        values = np.log10(values)
    
    fig = plt.figure()
    if log:
        plt.imshow(values, origin="lower", interpolation = "none", extent = (-1,1,-1,1), cmap=cmap, vmax=vmax), #vmin=vmax-18)
        gridlinecolor="mediumorchid"
    else:
        plt.imshow(values, origin="lower", interpolation = "none", extent = (-1,1,-1,1), cmap=cmap, vmax=vmax)
        gridlinecolor="grey"
    plt.title(title)
    ax = plt.gca()
    if not gridlines: plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect(ny/nx)
    ax.xaxis.set_inverted(True) #RA axis increases to the left
    if colorbar:
        plt.colorbar()
    
    if gridlines or plot_chord or plot_source:
        chob = np.linalg.inv(np.hstack((v1[np.newaxis].T, v2[np.newaxis].T, basevec[np.newaxis].T)))#change of basis matrix. The negative is to fix the orientations being opposite.
        def ang_2_tpp_coords (theta, phi): #let's define a function that can convert between spherical coordinates and tangent plane projection coodinates
            #the formula is
            # sec(alpha)(ch_of_basis_mat @ (cos phi sin theta, sin phi sin theta, cos(theta)), where alpha is the angular distance
            if isinstance(phi,np.ndarray): #we need some logic here depending on whether the inputs are arrays or not
                tempvec = np.asarray([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)*np.ones(phi.shape[0])])
            else:
                tempvec = np.asarray([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
            secalpha = 1/(np.dot(tempvec.T,basevec))
            tpp_coords_unscaled = secalpha*(chob[:2] @ tempvec)
            return tpp_coords_unscaled[1]/np.tan(extent1), tpp_coords_unscaled[0]/np.tan(extent2) #x,y
        
        if plot_chord and plot_chord!="line":
            if plot_chord is np.ndarray:
                print("error: it looks like you're trying to plot dithers as points. Plot them as lines instead.")
            else:
                x,y = ang_2_tpp_coords (chord_theta, chord_phi)
                plt.plot(x, y, 'rx', ms=15, label="CHORD location")
        if plot_source:
            x,y = ang_2_tpp_coords (source_theta, source_phi_0)
            plt.plot(x, y, 'bs', mfc='none', ms=15, label="Source location")
        if highlight_aliases:
            if time_samples == 1:
                #currently not implemented for chord_phi neq 0
                x,y = ang_2_tpp_coords (*vec2ang(approxAliasNorthu(np.atleast_1d(chord_theta)[0], ang2vec(source_theta,source_phi_0), wavelength, phi=chord_phi)))
                plt.plot(x, y, 'gs', mfc='none', ms=10, label="North alias location")
                x,y = ang_2_tpp_coords (*vec2ang(approxAliasSouthu(np.atleast_1d(chord_theta)[0], ang2vec(source_theta,source_phi_0), wavelength, phi=chord_phi)))
                plt.plot(x, y, 'gs', mfc='none', ms=10, label="South alias location")
                eastaliasAngularCoords = vec2ang(approxAliasEastu(np.atleast_1d(chord_theta)[0], ang2vec(source_theta,source_phi_0), wavelength, phi=chord_phi))
                x,y = ang_2_tpp_coords (*eastaliasAngularCoords)
                plt.plot(x, y, 'gs', mfc='none', ms=10, label="East alias location")
                x,y = ang_2_tpp_coords (eastaliasAngularCoords[0], source_phi_0 - (eastaliasAngularCoords[1] - source_phi_0))
                plt.plot(x, y, 'gs', mfc='none', ms=10, label="West alias location")
            else:
                x,y = ang_2_tpp_coords (*vec2ang(approxAliasNorthu(np.atleast_1d(chord_theta)[0], ang2vec(source_theta,source_phi_0), wavelength)))
                plt.plot(x, y, 'gs', mfc='none', ms=10, label="North alias location")
                x,y = ang_2_tpp_coords (*vec2ang(approxAliasSouthu(np.atleast_1d(chord_theta)[0], ang2vec(source_theta,source_phi_0), wavelength)))
                plt.plot(x, y, 'gs', mfc='none', ms=10, label="South alias location")
                eastaliasAngularCoords = vec2ang(AliasEast(chord_theta, source_theta, source_phi_0, wavelength, delta_tau, time_samples, m1,m2, return_extra=True)[0])
                x,y = ang_2_tpp_coords (*eastaliasAngularCoords)
                plt.plot(x, y, 'gs', mfc='none', ms=10, label="East alias location")
                x,y = ang_2_tpp_coords (eastaliasAngularCoords[0], source_phi_0 - (eastaliasAngularCoords[1] - source_phi_0))
                plt.plot(x, y, 'gs', mfc='none', ms=10, label="West alias location")
                
            
        if gridlines or plot_chord=="line":
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

            if gridlines:
                #next figuring out the line spacing. We don't want to plot too many gridlines.
                deg_thresholds = np.array([5,10,25,50,150],dtype=int)
                deg_separations = np.array([1,2,5,10,30,60],dtype=int)
                phi_separation = deg_separations[np.searchsorted(deg_thresholds,np.rad2deg(e1grid))]
                theta_separation = deg_separations[np.searchsorted(deg_thresholds,np.rad2deg(e2grid))]
                phi_ticks = get_integer_ticks(np.rad2deg(phitp[0]), np.rad2deg(phitp[-1]), phi_separation)
                theta_ticks = get_integer_ticks(np.rad2deg(thetatp[0]), np.rad2deg(thetatp[-1]), theta_separation)
                
                #holding values for pyplot ticks which we'll add later
                x_axis_tick_positions = []
                x_axis_tick_labels = []
                y_axis_tick_positions = []
                y_axis_tick_labels = []
                
                #convenient function
                def plot_const_RA_gridline (phi, color="grey",label=None):
                    x,y = ang_2_tpp_coords(thetatp,phi)
                    plt.plot(x, y, color=color, #linestyle=(0, (3, 10)),
                        alpha=0.3, label=label)
                    if axis_labels:
                        #we want to find if it crosses the boundary, and if so, write a tick marker
                        cross = np.searchsorted(y[::-1], -1)
                        if cross < x.shape[0] and x[::-1][cross] > -1 and x[::-1][cross] < 1:
                            x_axis_tick_positions.append(x[::-1][cross])
                            x_axis_tick_labels.append("${degvalue:n}$".format(degvalue=np.rad2deg(phi)))
                            #plt.text(x[::-1][cross]-0.02,-1.07, "{degvalue:n}".format(degvalue=np.rad2deg(phi)))
                
                for phi_deg in phi_ticks:
                    phi = np.deg2rad(phi_deg)
                    plot_const_RA_gridline (phi, color=gridlinecolor)
                for theta_deg in theta_ticks: #plot lines of constant dec
                    theta = np.deg2rad(theta_deg)
                    x,y = ang_2_tpp_coords(theta,phitp)
                    plt.plot(x, y, color=gridlinecolor, alpha=0.3)#linestyle=(0, (3, 10)))
                    if axis_labels:
                        #we want to find if it crosses the boundary, and if so, write a tick marker
                        cross = np.searchsorted(x, -1)
                        if cross < y.shape[0] and y[cross] < 1 and y[cross] > -1:
                            y_axis_tick_positions.append(y[cross])
                            y_axis_tick_labels.append("${degvalue:n}$".format(degvalue=90-theta_deg)) #converting to dec
                            #plt.text(-1.14, y[cross], "{degvalue:n}".format(degvalue=theta_deg))
                #plt.text(-0.8,-1.1,"Gridline separations: "+str(phi_separation)+" deg (RA), "+str(theta_separation)+" deg (dec)"))
                #plt.xticks(x_axis_tick_positions, x_axis_tick_labels)
                ax.set_xticks(x_axis_tick_positions, x_axis_tick_labels)
                if len(x_axis_tick_positions) > 0: plt.xlabel("RA (Deg)")
                ax.set_yticks(y_axis_tick_positions, y_axis_tick_labels)
                if len(y_axis_tick_positions) > 0: plt.ylabel("Dec (Deg)")
                plt.draw()
            if plot_chord=="line":
                if isinstance(chord_theta, np.ndarray):
                    for i in range(chord_theta.shape[0]):
                        x,y = ang_2_tpp_coords(chord_theta[i],phitp)
                        plt.plot(x, y, color="red",linestyle=(0, (3, 10)), label="CHORD")
                else:
                    x,y = ang_2_tpp_coords(chord_theta,phitp)
                    plt.plot(x, y, color="red",linestyle=(0, (3, 10)), label="CHORD")
                
    
if __name__ == "__main__":
    nsamples = 200
    omega = 360.0/(24.0*3600)
    #tan_plane_plot (np.deg2rad(40), -np.deg2rad(10), np.deg2rad(40), 400,400, np.deg2rad(3), np.deg2rad(3), 0.21, np.deg2rad(40)-np.deg2rad(0.8), -np.deg2rad(10), 24, 22, 20/omega/nsamples, nsamples, "", plot_chord="line", plot_source=True, gridlines=True, highlight_aliases=True)
    #tan_plane_plot (np.deg2rad(40), -np.deg2rad(5), np.array([np.deg2rad(40),np.deg2rad(38)]), 400,400, np.deg2rad(3), np.deg2rad(3), 0.21, np.deg2rad(40)-np.deg2rad(0.8), -np.deg2rad(5), 24, 22, 10/omega/nsamples, nsamples, "", plot_chord="line", plot_source=True, gridlines=True, plot_nalias=True, plot_ealias=True)
    tan_plane_plot (np.deg2rad(45), np.deg2rad(-15), np.deg2rad(45), 300, 300, np.deg2rad(5), np.deg2rad(5), 0.21, np.deg2rad(45), np.deg2rad(-15), 22, 24, 24*3600.0/nsamples, nsamples, "", mode="matched filter", gridlines=True, plot_chord=False)
    plt.show()
    
