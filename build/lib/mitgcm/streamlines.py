"""
Streamlines
====================

Functions for creating streamlines.

Each function has a detailed docstring.
"""

import numpy as np

def stream2(u,v,startx,starty,x_u,y_u,x_v=None,y_v=None,t_int=2592000,delta_t=3600,interpolation='bilinear'):
    """A two-dimensional streamline solver. The velocity fields must be two dimensional and not vary in time.
    """
    if x_v == None:
        x_v = x_u
    if y_v == None:
        y_v = y_u

    x_stream = np.ones((1))*startx
    y_stream = np.ones((1))*starty
    t_stream = np.zeros((1))

    t = 0 #set the initial time to be zero

    # Runge-Kutta fourth order method to estimate next position.
    while t < t_int:
      # Interpolate velocities to initial location
      u_loc = bilinear_interp([startx,starty],u,x_u,y_u)
      v_loc = bilinear_interp([startx,starty],v,x_v,y_v)
      dx1 = delta_t*u_loc
      dy1 = delta_t*v_loc

      u_loc1 = bilinear_interp([startx + 0.5*dx1,starty + 0.5*dy1],u,x_u,y_u)
      v_loc1 = bilinear_interp([startx + 0.5*dx1,starty + 0.5*dy1],v,x_v,y_v)
      dx2 = delta_t*u_loc1
      dy2 = delta_t*v_loc1

      u_loc2 = bilinear_interp([startx + 0.5*dx2,starty + 0.5*dy2],u,x_u,y_u)
      v_loc2 = bilinear_interp([startx + 0.5*dx2,starty + 0.5*dy2],v,x_v,y_v)
      dx3 = delta_t*u_loc2
      dy3 = delta_t*v_loc2

      u_loc3 = bilinear_interp([startx + dx3,starty + dy3],u,x_u,y_u)
      v_loc3 = bilinear_interp([startx + dx3,starty + dy3],v,x_v,y_v)
      dx4 = delta_t*u_loc3
      dy4 = delta_t*v_loc3

      startx = startx + (dx1 + 2*dx2 + 2*dx3 + dx4)/6
      starty = starty + (dy1 + 2*dy2 + 2*dy3 + dy4)/6

      t += delta_t

      x_stream = np.append(x_stream,startx)
      y_stream = np.append(y_stream,starty)
      t_stream = np.append(t_stream,delta_t)

    return x_stream,y_stream,t_stream



def bilinear_interp(location,vel,x,y):
	"""Do bilinear interpolation of the velocity field to get nice accurate streamlines. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles.

	location is an array of the point to interpolate to, (x_int,y_int).
	"""

	# Compute indeces at location
	x_index = np.searchsorted(x,location[0])
	if x_index == 0:
	  raise ValueError('Given x location is outside the model grid - too small')
	elif x_index == len(x):
	  raise ValueError('Given x location is outside the model grid - too big')
	  
	y_index = np.searchsorted(y,location[1])
	if y_index == 0:
	  raise ValueError('Given y location is outside the model grid - too small')
	elif y_index == len(y):
	  raise ValueError('Given y location is outside the model grid - too big')
	
	#print 'x index = ' + str(x_index)
	#print 'y index = ' + str(y_index)
    
	vel = ((vel[y_index-1,x_index-1]*(x[x_index] - location[0])*(y[y_index] - location[1]) + 
		   vel[y_index-1,x_index]*(location[0] - x[x_index-1])*(y[y_index] - location[1]) +
		   vel[y_index,x_index]*(x[x_index] - location[0])*(location[1] - y[y_index-1]) + 
		   vel[y_index,x_index]*(location[0] - x[x_index-1])*(location[1] - y[y_index-1]))/
		   ((y[y_index] - y[y_index-1])*(x[x_index] - x[x_index-1]))) 

	return vel



class InputError(Exception):
	pass