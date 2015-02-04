"""
Streamlines
====================

Functions for creating streamlines.

Each function has a detailed docstring.
"""

import numpy as np

def stream2(u,v,startx,starty,x_u,y_u,x_v=None,y_v=None,t_int=2592000,delta_t=3600,interpolation='bilinear'):

	if x_v == None:
		x_v = x_u
	if y_v == None:
		y_v = y_u


	t = 0 #set the initial time to be zero



	# Interpolate velocities to initial location
	u_loc = bilinear_interp([startx,starty],u,x_u,y_u)

	# Runge-Kutta fourth order method to estimate next position.
	u_loc = 







def bilinear_interp(location,vel,x,y):#_x0,vel_x1,x0,x1,y0,y1):
	"""Do bilinear interpolation of the velocity field to get nice accurate streamlines. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles.

	location is an array of the point to interpolate from, (x_int,y_int).
	The velocity field is passed as two tuples; one at the lower value of x, vel_x0 = (vel(x0,y0), vel(x0,y1)), and the other at the upper value of x, vel_x1 = (vel(x1,y0), vel(x1,y1).
	"""

	assert len(location) == 2, "location must be a tuple :("
	assert len(vel_x0) == 2, "vel_x0 must be a tuple :("
	assert len(vel_x1) == 2, "vel_x1 must be a tuple :("

	# Compute indeces at location
	x_index = np.searchsorted(x,location[0])
	y_index = np.searchsorted(y,location[1])

	vel = (vel_x0[0]*(x1 - location[0])*(y1 - location[1]) + 
		   vel_x1[0]*(location[0] - x0)*(y1 - location[1]) +
		   vel_x0[1]*(x1 - location[0])*(location[1] - y0) + 
		   vel_x1[1]*(location[0] - x0)*(location[1] - y0))/
		   ((y1 - y0)*(x1 - x0)) 

	return vel



class InputError(Exception):
	pass