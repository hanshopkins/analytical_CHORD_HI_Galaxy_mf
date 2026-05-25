import numpy as np
import matplotlib.pyplot as plt
from util import ang2vec, vec2ang

def find_minmax_phi_from_pixelvecs (v, basephi):
	pixelthetas, pixelphis = vec2ang(v)
	relative_pixelphis = pixelphis - basephi
	clamped_relative_pixelphis = relative_pixelphis % 2*np.pi
	correct_domain_relative_pixelphis = np.where(clamped_relative_pixelphis >= np.pi, clamped_relative_pixelphis-2*np.pi, clamped_relative_pixelphis)
	maxpixelphi = np.max(correct_domain_relative_pixelphis) + basephi
	minpixelphi = np.min(correct_domain_relative_pixelphis) + basephi
	return minpixelphi, maxpixelphi

def find_minmax_theta_from_pixelvecs (v):
	pixelthetas, pixelphis = vec2ang(v)
	return np.min(pixelthetas), np.max(pixelthetas)

def angular_difference (a1, a2):
	return np.min((a1-a2)%(2*np.pi), (a2-a1)%(2*np.pi))

def get_integer_ticks (low, high, separation):
    return np.arange(np.ceil(low/separation)*separation, (np.floor(high/separation)+1)*separation, separation)

class tangent_plane_information:
	def __init__(self, nx, ny, base_theta, base_phi, extent1, extent2):
		#1 is NS, 2 is vaguely EW
		self.nx = nx
		self.ny = ny
		self.base_theta = base_theta
		self.base_phi = base_phi
		self.extent1 = extent1
		self.extent2 = extent2
		
		#computing basis vectors
		self.basevec = ang2vec(base_theta, base_phi)
		self.v1 = ang2vec(base_theta - np.pi/2, base_phi)
		self.v2 = np.cross(self.v1, self.basevec)
		#we need these when plotting lines. They're relatively expensive to compute, so it'd be nice to just save them when we do.
		self.maxpixelphi = None
		self.minpixelphi = None
		self.maxpixeltheta = None
		self.minpixeltheta = None

	def get_pixelvecs(self):
		#we want to find the great circle going through the north pole, and the great circle that is perpendicular to that at the base point. Those are our two axes.
		testvecs = np.empty([self.nx,self.ny,3])

		xls = np.linspace(-1,1,self.nx) * np.tan(self.extent1)
		yls = np.linspace(-1,1,self.ny) * np.tan(self.extent2)
		testvecs = (self.basevec[np.newaxis, np.newaxis, :]
		  + self.v1[np.newaxis, np.newaxis, :] * yls[:, np.newaxis, np.newaxis]
		  + self.v2[np.newaxis, np.newaxis, :] * xls[np.newaxis, :, np.newaxis])

		#normalizing
		norms = np.linalg.norm(testvecs, axis=2)
		np.divide(testvecs[:,:,0],norms, testvecs[:,:,0])
		np.divide(testvecs[:,:,1],norms, testvecs[:,:,1])
		np.divide(testvecs[:,:,2],norms, testvecs[:,:,2])

		#computing min/maxpixelphi for later if we need to
		if self.maxpixelphi == None:
			self.minpixelphi, self.maxpixelphi = find_minmax_phi_from_pixelvecs (testvecs.reshape([self.nx*self.ny,3]), self.base_phi)
			self.minpixeltheta, self.maxpixeltheta = find_minmax_theta_from_pixelvecs (testvecs.reshape([self.nx*self.ny,3]))

		return testvecs
	
	def ang_2_tpp_coords (self, theta, phi): #let's define a function that can convert between spherical coordinates and tangent plane projection coodinates
		#the formula is
		# sec(alpha)(ch_of_basis_mat @ (cos phi sin theta, sin phi sin theta, cos(theta)), where alpha is the angular distance
		chob = np.linalg.inv(np.hstack((self.v1[np.newaxis].T, self.v2[np.newaxis].T, self.basevec[np.newaxis].T))) #change of basis matrix.
		if isinstance(phi,np.ndarray): #we need some logic here depending on whether the inputs are arrays or not
			tempvec = np.asarray([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)*np.ones(phi.shape[0])])
		else:
			tempvec = np.asarray([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
		secalpha = 1/(np.dot(tempvec.T,self.basevec))
		tpp_coords_unscaled = secalpha*(chob[:2] @ tempvec)
		return tpp_coords_unscaled[1]/np.tan(self.extent1), tpp_coords_unscaled[0]/np.tan(self.extent2) #x,y
	
	def vec_2_tpp_coords (self, v):
		chob = np.linalg.inv(np.hstack((self.v1[np.newaxis].T, self.v2[np.newaxis].T, self.basevec[np.newaxis].T))) #change of basis matrix.
		secalpha = 1/(np.dot(v.T,self.basevec))
		tpp_coords_unscaled = secalpha*(chob[:2] @ v)
		return tpp_coords_unscaled[1]/np.tan(self.extent1), tpp_coords_unscaled[0]/np.tan(self.extent2) #x,y

	def const_dec_line_tpp_coords (self, theta):
		#if we haven't already computed min/maxpixelphi, we need to do that now
		if self.maxpixelphi == None:
			vecs = tpi.get_pixelvecs() #this will compute the min/max
			#self.minpixelphi, self.maxpixelphi = find_minmax_phi_from_pixelvecs (vecs, self.base_phi)

		out_tpp_coords = np.empty([300,2])
		pixelphidiff = self.maxpixelphi-self.minpixelphi
		if pixelphidiff > np.pi and angular_difference(self.minpixelphi, self.maxpixelphi) < np.deg2rad(10): #it's basically the whole circle so just plot the whole circle
			phis = np.linspace(0,2*np.pi, 300)
		else:
			phis = np.linspace(self.minpixelphi-pixelphidiff*0.1, self.maxpixelphi+pixelphidiff*0.1, 300)

		return self.ang_2_tpp_coords(phis,theta)

	def const_RA_line_tpp_coords (self, phi):
		#if we haven't already computed min/maxpixelphi, we need to do that now
		if self.maxpixelphi == None:
			vecs = tpi.get_pixelvecs() #this will compute the min/max
			#self.minpixelphi, self.maxpixelphi = find_minmax_phi_from_pixelvecs (vecs, self.base_phi)

		out_tpp_coords = np.empty([300,2])
		pixelthetadiff = self.maxpixeltheta-self.minpixeltheta
		if self.minpixeltheta < np.deg2rad(5): #we're really close to the north pole, so let's just set the min to 0
			thetas = np.linspace(0, maxpixeltheta, 300)
		else:
			thetas = np.linspace(self.minpixeltheta-pixelthetadiff*0.1, self.maxpixeltheta+pixelthetadiff*0.1, 300)

		return self.ang_2_tpp_coords(phi,thetas)

def tangent_plane_plot (values, tpi, cmap="Greys", vmax=None, logcolor=False, title=None, gridlines=True, plot_chord=False, chord_theta=None, chord_phi=None,
		 plot_source=False, source_phi = None, source_theta = None, colorbar=False, axis_labels=False, ax=None):
	if ax is None:
		fig = plt.figure()
		ax = fig.get_axes()[0]
	ax.imshow(values, origin="lower", interpolation = "none", extent = (-1,1,-1,1), cmap=cmap, vmax=vmax)
	if logcolor:
		gridlinecolor="mediumorchid"
	else:
		gridlinecolor="grey"

	if title:
		plt.suptitle(title)

	ax.set_xlim([-1, 1])
	ax.set_ylim([-1, 1])
	ax.set_aspect(tpi.ny/tpi.nx)
	ax.xaxis.set_inverted(True) #RA axis increases to the left

	if colorbar:
		plt.colorbar()

	if not gridlines:
		ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
	else:
		deg_thresholds = np.array([5,10,25,50,150],dtype=int)
		deg_separations = np.array([1,2,5,10,30,60],dtype=int)
		if tpi.maxpixelphi == None:
			vecs = tpi.get_pixelvecs() #this will compute the min/max

		phi_separation = deg_separations[np.searchsorted(deg_thresholds,np.rad2deg(tpi.maxpixelphi-tpi.minpixelphi))]
		theta_separation = deg_separations[np.searchsorted(deg_thresholds,np.rad2deg(tpi.maxpixeltheta-tpi.minpixeltheta))]
		phi_ticks = get_integer_ticks(np.rad2deg(tpi.minpixelphi), np.rad2deg(tpi.maxpixelphi), phi_separation)
		theta_ticks = get_integer_ticks(np.rad2deg(tpi.maxpixeltheta), np.rad2deg(tpi.minpixeltheta), theta_separation)

		#holding values for pyplot ticks which we'll add later
		x_axis_tick_positions = []
		x_axis_tick_labels = []
		y_axis_tick_positions = []
		y_axis_tick_labels = []

		for phi_deg in phi_ticks:
			phi = np.deg2rad(phi_deg)
			x,y = tpi.const_RA_line_tpp_coords (phi)
			ax.plot(x, y, color="grey", alpha=0.3)
			if axis_labels:
				#we want to find if it crosses the boundary, and if so, write a tick marker
				cross = np.searchsorted(y[::-1], -1)
				if cross < x.shape[0] and x[::-1][cross] > -1 and x[::-1][cross] < 1:
					x_axis_tick_positions.append(x[::-1][cross])
					x_axis_tick_labels.append("${degvalue:n}$".format(degvalue=np.rad2deg(phi)))
		for theta_deg in theta_ticks: #plot lines of constant dec
			theta = np.deg2rad(theta_deg)
			x,y = tpi.const_dec_line_tpp_coords (theta)
			ax.plot(x, y, color=gridlinecolor, alpha=0.3)
			if axis_labels:
				#we want to find if it crosses the boundary, and if so, write a tick marker
				cross = np.searchsorted(x, -1)
				if cross < y.shape[0] and y[cross] < 1 and y[cross] > -1:
					y_axis_tick_positions.append(y[cross])
					y_axis_tick_labels.append("${degvalue:n}$".format(degvalue=90-theta_deg)) #converting to dec
		ax.set_xticks(x_axis_tick_positions, x_axis_tick_labels)
		if len(x_axis_tick_positions) > 0: plt.xlabel("RA (Deg)")
		ax.set_yticks(y_axis_tick_positions, y_axis_tick_labels)
		if len(y_axis_tick_positions) > 0: plt.ylabel("Dec (Deg)")
		plt.draw()

	if plot_chord == True:
		if not (isinstance(chord_theta,float) and isinstance(chord_phi,float)):
			raise ValueError("Expecting chord location inputs if plot_chord==True")
		x,y = tpi.ang_2_tpp_coords (chord_theta, chord_phi)
		ax.plot(x, y, 'rx', ms=15, label="CHORD location")
	elif plot_chord == "line":
		if isinstance(chord_theta,float):
			x,y = tpi.const_dec_line_tpp_coords (chord_theta)
			ax.plot(x, y, color="red",linestyle=(0, (3, 10)), label="CHORD")
		elif isinstance(chord_theta,np.ndarray):
			for i in range(chord_theta.shape[0]):
				x,y = tpi.const_dec_line_tpp_coords (chord_theta[i])
				ax.plot(x, y, color="red",linestyle=(0, (3, 10)), label="CHORD")

	if plot_source == True:
		x,y = tpi.ang_2_tpp_coords (source_theta, source_phi)
		ax.plot(x, y, 'bs', mfc='none', ms=15, label="Source location")
