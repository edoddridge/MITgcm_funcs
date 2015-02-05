"""
Streamlines
====================

Functions for creating streamlines.

Each function has a detailed docstring.
"""

import numpy as np

def stream2(u,v,startx,starty,x_u,y_u,x_v='None',y_v='None',t_int=2592000,delta_t=3600,interpolation='bilinear'):
    """A two-dimensional streamline solver. The velocity fields must be two dimensional and not vary in time.
    """
    if x_v == 'None':
        x_v = x_u
    if y_v == 'None':
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


def stream3(u,v,w,
            startx,starty,startz,
            x_u,y_u,z_u,
            x_v='None',y_v='None',z_v='None',
            x_w='None',y_w='None',z_w='None',
            t_int=2592000,delta_t=3600):
    """A three-dimensional streamline solver. The velocity fields must be three dimensional and not vary in time.
    """
    if x_v == 'None':
        x_v = x_u
    if y_v == 'None':
        y_v = y_u
    if z_v == 'None':
        z_v = z_u

    if x_w == 'None':
        x_w = x_u
    if y_w == 'None':
        y_w = y_u
    if z_w == 'None':
        z_w = z_u

    x_stream = np.ones((1))*startx
    y_stream = np.ones((1))*starty
    z_stream = np.ones((1))*startz
    t_stream = np.zeros((1))

    t = 0 #set the initial time to be zero

    # Runge-Kutta fourth order method to estimate next position.
    while t < t_int:
      # Interpolate velocities to initial location
      u_loc = trilinear_interp([startx,starty,startz],u,x_u,y_u,z_u)
      v_loc = trilinear_interp([startx,starty,startz],v,x_v,y_v,z_v)
      w_loc = trilinear_interp([startx,starty,startz],w,x_w,y_w,z_w)
      dx1 = delta_t*u_loc
      dy1 = delta_t*v_loc
      dz1 = delta_t*w_loc

      u_loc1 = trilinear_interp([startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1],u,x_u,y_u,z_u)
      v_loc1 = trilinear_interp([startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1],v,x_v,y_v,z_v)
      w_loc1 = trilinear_interp([startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1],w,x_w,y_w,z_w)
      dx2 = delta_t*u_loc1
      dy2 = delta_t*v_loc1
      dz2 = delta_t*w_loc1

      u_loc2 = trilinear_interp([startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2],u,x_u,y_u,z_u)
      v_loc2 = trilinear_interp([startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2],v,x_v,y_v,z_v)
      w_loc2 = trilinear_interp([startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2],w,x_w,y_w,z_w)
      dx3 = delta_t*u_loc2
      dy3 = delta_t*v_loc2
      dz3 = delta_t*w_loc2

      u_loc3 = trilinear_interp([startx + dx3,starty + dy3,startz + dz3],u,x_u,y_u,z_u)
      v_loc3 = trilinear_interp([startx + dx3,starty + dy3,startz + dz3],v,x_v,y_v,z_v)
      w_loc3 = trilinear_interp([startx + dx3,starty + dy3,startz + dz3],w,x_w,y_w,z_w)
      dx4 = delta_t*u_loc3
      dy4 = delta_t*v_loc3
      dz4 = delta_t*w_loc3

      #recycle the "start_" variables to keep the code clean
      startx = startx + (dx1 + 2*dx2 + 2*dx3 + dx4)/6
      starty = starty + (dy1 + 2*dy2 + 2*dy3 + dy4)/6
      startz = startz + (dz1 + 2*dz2 + 2*dz3 + dz4)/6

      t += delta_t

      x_stream = np.append(x_stream,startx)
      y_stream = np.append(y_stream,starty)
      z_stream = np.append(z_stream,startz)
      t_stream = np.append(t_stream,delta_t)

    return x_stream,y_stream,z_stream,t_stream



def bilinear_interp(x0,y0,vel,x,y):
  """Do bilinear interpolation of the velocity field to get nice accurate streamlines. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles.

  x0,y0 are the points to interpolate to.
  """

  # Compute indeces at location
  x_index = np.searchsorted(x,x0)
  if x_index == 0:
    raise ValueError('Given x location is outside the model grid - too small')
  elif x_index == len(x):
    raise ValueError('Given x location is outside the model grid - too big')
    
  y_index = np.searchsorted(y,y0)
  if y_index == 0:
    raise ValueError('Given y location is outside the model grid - too small')
  elif y_index == len(y):
    raise ValueError('Given y location is outside the model grid - too big')
  
  #print 'x index = ' + str(x_index)
  #print 'y index = ' + str(y_index)
    
  vel_interp = ((vel[y_index-1,x_index-1]*(x[x_index] - x0)*(y[y_index] - y0) + 
       vel[y_index-1,x_index]*(x0 - x[x_index-1])*(y[y_index] - y0) +
       vel[y_index,x_index]*(x[x_index] - x0)*(y0 - y[y_index-1]) + 
       vel[y_index,x_index]*(x0 - x[x_index-1])*(y0 - y[y_index-1]))/
       ((y[y_index] - y[y_index-1])*(x[x_index] - x[x_index-1]))) 

  return vel_interp

def trilinear_interp(x0,y0,z0,vel,x,y,z):
  """Do trilinear interpolation of the velocity field in three spatial dimensions to get nice accurate streamlines. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles.

  x0,y0, and z0 represent the point to interpolate to.
  """

  # Compute indices at location given
  x_index = np.searchsorted(x,x0)
  if x_index == 0:
    raise ValueError('Given x location is outside the model grid - too small')
  elif x_index == len(x):
    raise ValueError('Given x location is outside the model grid - too big')
    
  y_index = np.searchsorted(y,y0)
  if y_index == 0:
    raise ValueError('Given y location is outside the model grid - too small')
  elif y_index == len(y):
    raise ValueError('Given y location is outside the model grid - too big')
  
  z_index = np.searchsorted(z,z0)
  if y_index == 0:
    raise ValueError('Given y location is outside the model grid - too small')
  elif y_index == len(y):
    raise ValueError('Given y location is outside the model grid - too big')

  #print 'x index = ' + str(x_index)
  #print 'y index = ' + str(y_index)
    
  vel_interp = ((vel[z_index-1,y_index-1,x_index-1]*
                    (x[x_index] - x0)*(y[y_index] - y0)*(z[z_index] - z0) + 
                vel[z_index,y_index-1,x_index-1]*
                    (x[x_index] - x0)*(y[y_index] - y0)*(z0 - z[z_index-1]) + 
                vel[z_index,y_index,x_index-1]*
                    (x[x_index] - x0)*(y0 - y[y_index-1])*(z0 - z[z_index-1]) + 
                vel[z_index-1,y_index,x_index-1]*
                    (x[x_index] - x0)*(y0 - y[y_index-1])*(z[z_index] - z0) + 
                vel[z_index-1,y_index-1,x_index]*
                    (x0 - x[x_index-1])*(y[y_index] - y0)*(z[z_index] - z0) + 
                vel[z_index,y_index-1,x_index]*
                    (x0 - x[x_index-1])*(y[y_index] - y0)*(z0 - z[z_index-1]) + 
                vel[z_index,y_index,x_index]*
                    (x0 - x[x_index-1])*(y0 - y[y_index-1])*(z0 - z[z_index-1]) + 
                vel[z_index-1,y_index,x_index]*
                    (x0 - x[x_index-1])*(y0 - y[y_index-1])*(z[z_index] - z0))/
             ((z[z_index] - z[z_index-1])*(y[y_index] - y[y_index-1])*(x[x_index] - x[x_index-1]))) 

  return vel_interp

def quadralinear_interp(location,mitgcm_model_instance,field,x,y,z,t):
  #NOT FINISHED YET
  """Do quadralinear interpolation of the velocity field in three spatial dimensions and one temporal dimension to get nice accurate streaklines. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles.

  location is an array of the point to interpolate to in four dimensional space-time (x_int,y_int,z_int,t_int).

  The velocity field needs to be passed as either a 4D variable, which is big and expensive, or as a handle to the place where it can be obtained from disk
  """

  # Compute indices at location given
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
  
  z_index = np.searchsorted(z,location[2])
  if z_index == 0:
    raise ValueError('Given y location is outside the model grid - too small')
  elif z_index == len(z):
    raise ValueError('Given y location is outside the model grid - too big')

  t_index = np.searchsorted(t,location[3])
  if t_index == 0:
    raise ValueError('Given t location is outside the model - too small')
  elif t_index == len(t):
    raise ValueError('Given t location is outside the model - too big')

    
  vel_interp = ((
        vel[t_index-1,z_index-1,y_index-1,x_index-1]*
            (x[x_index] - location[0])*(y[y_index] - location[1])*(z[z_index] - location[2]) + 
        vel[z_index,y_index-1,x_index-1]*
            (x[x_index] - location[0])*(y[y_index] - location[1])*(location[2] - z[z_index-1]) + 
        vel[z_index,y_index,x_index-1]*
            (x[x_index] - location[0])*(location[1] - y[y_index-1])*(location[2] - z[z_index-1]) + 
        vel[z_index-1,y_index,x_index-1]*
            (x[x_index] - location[0])*(location[1] - y[y_index-1])*(z[z_index] - location[2]) + 
        vel[z_index-1,y_index-1,x_index]*
            (location[0] - x[x_index-1])*(y[y_index] - location[1])*(z[z_index] - location[2]) + 
        vel[z_index,y_index-1,x_index]*
            (location[0] - x[x_index-1])*(y[y_index] - location[1])*(location[2] - z[z_index-1]) + 
        vel[z_index,y_index,x_index]*
            (location[0] - x[x_index-1])*(location[1] - y[y_index-1])*(location[2] - z[z_index-1]) + 
        vel[z_index-1,y_index,x_index]*
            (location[0] - x[x_index-1])*(location[1] - y[y_index-1])*(z[z_index] - location[2]))/
       ((z[z_index] - z[z_index-1])*(y[y_index] - y[y_index-1])*(x[x_index] - x[x_index-1]))) 

  return vel_interp

class InputError(Exception):
	pass