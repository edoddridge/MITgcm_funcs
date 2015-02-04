"""
Streamlines
====================

Functions for creating streamlines.

Each function has a detailed docstring.
"""

import numpy as np

def stream2(u,v,startx,starty,x_u,y_u,x_v,y_v,t_int=2592000,delta_t=3600,interpolation='bilinear',grid='C'):

	if grid =='A':
		if u.shape != v.shape:
			raise InputError('U and V must be the same shape on an A-grid')
		else:
			pass
	if grid =='B':
		if u.shape != v.shape:
			raise InputError('U and V must be the same shape on a B-grid')
		else:
			pass
	elif grid =='C':
		pass
		#if u.shape[0] != v.shape[0] + 1:
	else:
		raise InputError('Only Arakawa A, B and C grids are supported.')


	t = 0 #set the initial time to be zero

	# Runge-Kutta method to estimate next position.







def bilinear_interp(location,vel_x0,vel_x1,x0,x1,y0,y1):
	"""Do bilinear interpolation of the velocity field to get nice accurate streamlines. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles.

	location is a tuple of the point to interpolate to, (x_int,y_int).
	The velocity field is passed as two tuples; one at the lower value of x, vel_x0 = (vel(x0,y0), vel(x0,y1)), and the other at the upper value of x, vel_x1 = (vel(x1,y0), vel(x1,y1).
	"""

	assert len(location) == 2, "location must be a tuple :("
	assert len(vel_x0) == 2, "vel_x0 must be a tuple :("
	assert len(vel_x1) == 2, "vel_x1 must be a tuple :("

	vel = (vel_x0[0]*(x1 - location[0])*(y1 - location[1]) + 
		   vel_x1[0]*(location[0] - x0)*(y1 - location[1]) +
		   vel_x0[1]*(x1 - location[0])*(location[1] - y0) + 
		   vel_x1[1]*(location[0] - x0)*(location[1] - y0))/
		   ((y1 - y0)*(x1 - x0)) 

	return vel



class InputError(Exception):
	pass