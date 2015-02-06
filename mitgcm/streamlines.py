"""
Streamlines
====================

Functions for creating streamlines.

Each function has a detailed docstring.
"""

import numpy as np
import netCDF4
from numba import jit

def stream2(u,v,startx,starty,x_u,y_u,x_v='None',y_v='None',t_int=2592000,delta_t=3600,interpolation='bilinear'):
    """A two-dimensional streamline solver. The velocity fields must be two dimensional and not vary in time.
    """
    if x_v == 'None':
        x_v = x_u
    if y_v == 'None':
        y_v = y_u

    len_x_u = len(x_u)
    len_y_u = len(y_u)
    
    len_x_v = len(x_v)
    len_y_v = len(y_v)
    
    x_stream = np.ones((int(t_int/delta_t)+2))*startx
    y_stream = np.ones((int(t_int/delta_t)+2))*starty
    t_stream = np.zeros((int(t_int/delta_t)+2))

    t = 0 #set the initial time to be zero
    i=0

    # Runge-Kutta fourth order method to estimate next position.
    while t < t_int:
      # Interpolate velocities to initial location
      u_loc = bilinear_interp(startx,starty,u,x_u,y_u,len_x_u,len_y_u)
      v_loc = bilinear_interp(startx,starty,v,x_v,y_v,len_x_v,len_y_v)
      dx1 = delta_t*u_loc
      dy1 = delta_t*v_loc

      u_loc1 = bilinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,u,x_u,y_u,len_x_u,len_y_u)
      v_loc1 = bilinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,v,x_v,y_v,len_x_v,len_y_v)
      dx2 = delta_t*u_loc1
      dy2 = delta_t*v_loc1

      u_loc2 = bilinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,u,x_u,y_u,len_x_u,len_y_u)
      v_loc2 = bilinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,v,x_v,y_v,len_x_v,len_y_v)
      dx3 = delta_t*u_loc2
      dy3 = delta_t*v_loc2

      u_loc3 = bilinear_interp(startx + dx3,starty + dy3,u,x_u,y_u,len_x_u,len_y_u)
      v_loc3 = bilinear_interp(startx + dx3,starty + dy3,v,x_v,y_v,len_x_v,len_y_v)
      dx4 = delta_t*u_loc3
      dy4 = delta_t*v_loc3

      startx = startx + (dx1 + 2*dx2 + 2*dx3 + dx4)/6
      starty = starty + (dy1 + 2*dy2 + 2*dy3 + dy4)/6
      t += delta_t

      t += delta_t
      i += 1

      x_stream[i] = startx
      y_stream[i] = starty
      t_stream[i] = t
      
      
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
        
    len_x_u = len(x_u)
    len_y_u = len(y_u)
    len_z_u = len(z_u)
    
    len_x_v = len(x_v)
    len_y_v = len(y_v)
    len_z_v = len(z_v)
    
    len_x_w = len(x_w)
    len_y_w = len(y_w)
    len_z_w = len(z_w)

    x_stream = np.ones((int(t_int/delta_t)+2))*startx
    y_stream = np.ones((int(t_int/delta_t)+2))*starty
    z_stream = np.ones((int(t_int/delta_t)+2))*startz
    t_stream = np.zeros((int(t_int/delta_t)+2))

    t = 0 #set the initial time to be zero
    i=0
    
    # Runge-Kutta fourth order method to estimate next position.
    while t < t_int:
      # Interpolate velocities to initial location
      u_loc = trilinear_interp(startx,starty,startz,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
      v_loc = trilinear_interp(startx,starty,startz,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
      w_loc = trilinear_interp(startx,starty,startz,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
    
      dx1 = delta_t*u_loc
      dy1 = delta_t*v_loc
      dz1 = delta_t*w_loc

      u_loc1 = trilinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
      v_loc1 = trilinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
      w_loc1 = trilinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
      dx2 = delta_t*u_loc1
      dy2 = delta_t*v_loc1
      dz2 = delta_t*w_loc1

      u_loc2 = trilinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
      v_loc2 = trilinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
      w_loc2 = trilinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
      dx3 = delta_t*u_loc2
      dy3 = delta_t*v_loc2
      dz3 = delta_t*w_loc2

      u_loc3 = trilinear_interp(startx + dx3,starty + dy3,startz + dz3,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
      v_loc3 = trilinear_interp(startx + dx3,starty + dy3,startz + dz3,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
      w_loc3 = trilinear_interp(startx + dx3,starty + dy3,startz + dz3,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
      dx4 = delta_t*u_loc3
      dy4 = delta_t*v_loc3
      dz4 = delta_t*w_loc3

      #recycle the "start_" variables to keep the code clean
      startx = startx + (dx1 + 2*dx2 + 2*dx3 + dx4)/6
      starty = starty + (dy1 + 2*dy2 + 2*dy3 + dy4)/6
      startz = startz + (dz1 + 2*dz2 + 2*dz3 + dz4)/6
      t += delta_t
      i += 1

      x_stream[i] = startx
      y_stream[i] = starty
      z_stream[i] = startz
      t_stream[i] = t
      
      
    return x_stream,y_stream,z_stream,t_stream

def streaklines(u_netcdf_filename,v_netcdf_filename,w_netcdf_filename,
            startx,starty,startz,startt,
            t,
            x_u,y_u,z_u,
            x_v='None',y_v='None',z_v='None',
            x_w='None',y_w='None',z_w='None',            
            u_netcdf_variable='UVEL',
            v_netcdf_variable='VVEL',
            w_netcdf_variable='WVEL',
            t_int=6e7,delta_t=3600):
    """A three-dimensional lagrangian particle tracker. The velocity fields must be three dimensional and  vary in time. Because this is a very large amount of data, it is passed as netcdffile handles.
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

    len_x_u = len(x_u)
    len_y_u = len(y_u)
    len_z_u = len(z_u)
    
    len_x_v = len(x_v)
    len_y_v = len(y_v)
    len_z_v = len(z_v)
    
    len_x_w = len(x_w)
    len_y_w = len(y_w)
    len_z_w = len(z_w)
    
    len_t = len(t)

    x_stream = np.ones((int(t_int/delta_t)+2))*startx
    y_stream = np.ones((int(t_int/delta_t)+2))*starty
    z_stream = np.ones((int(t_int/delta_t)+2))*startz
    t_stream = np.zeros((int(t_int/delta_t)+2))

    t_RK = startt #set the initial time to be the given start time
    i=0
    
    u_netcdf_filehandle = netCDF4.Dataset(u_netcdf_filename)
    v_netcdf_filehandle = netCDF4.Dataset(v_netcdf_filename)
    w_netcdf_filehandle = netCDF4.Dataset(w_netcdf_filename)

    
    # Runge-Kutta fourth order method to estimate next position.
    while t_RK < t_int + startt:
        
        # Compute indices at location given
        
        t_index = np.searchsorted(t,startt)
        if t_index == 0:
            raise ValueError('Given time value location is outside the model - too small')
        elif t_index == len_t:
            raise ValueError('Given time value is outside the model - too big')
            
            
        # for u

        u_field,x_index_u,y_index_u,z_index_u = indices_and_field(x_u,y_u,z_u,
                                                    startx,starty,startz,t_index,
                                                    len_x_u,len_y_u,len_z_u,len_t,
                                                    u_netcdf_filehandle,u_netcdf_variable)
        # for v
        v_field,x_index_v,y_index_v,z_index_v = indices_and_field(x_v,y_v,z_v,
                                                    startx,starty,startz,t_index,
                                                    len_x_v,len_y_v,len_z_v,len_t,
                                                    v_netcdf_filehandle,v_netcdf_variable)

        # for w
        w_field,x_index_w,y_index_w,z_index_w = indices_and_field(x_w,y_w,z_w,
                                                    startx,starty,startz,t_index,
                                                    len_x_w,len_y_w,len_z_w,len_t,
                                                    w_netcdf_filehandle,w_netcdf_variable)
        


        # Interpolate velocities to initial location
        u_loc = quadralinear_interp(startx,starty,startz,startt,
                    u_field,
                    x_u,y_u,z_u,t,
                    len_x_u,len_y_u,len_z_u,len_t,
                    x_index_u,y_index_u,z_index_u,t_index)
        v_loc = quadralinear_interp(startx,starty,startz,startt,
                    v_field,
                    x_v,y_v,z_v,t,len_x_v,len_y_v,len_z_v,len_t,
                    x_index_v,y_index_v,z_index_v,t_index)
        w_loc = quadralinear_interp(startx,starty,startz,startt,
                    w_field,
                    x_w,y_w,z_w,t,len_x_w,len_y_w,len_z_w,len_t,
                    x_index_w,y_index_w,z_index_w,t_index)
        dx1 = delta_t*u_loc
        dy1 = delta_t*v_loc
        dz1 = delta_t*w_loc

        u_loc1 = quadralinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1,startt + 0.5*delta_t,
                    u_field,
                    x_u,y_u,z_u,t,len_x_u,len_y_u,len_z_u,len_t,
                    x_index_u,y_index_u,z_index_u,t_index)
        v_loc1 = quadralinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1,startt + 0.5*delta_t,
                    v_field,
                    x_v,y_v,z_v,t,len_x_v,len_y_v,len_z_v,len_t,
                    x_index_v,y_index_v,z_index_v,t_index)
        w_loc1 = quadralinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1,startt + 0.5*delta_t,
                    w_field,
                    x_w,y_w,z_w,t,len_x_w,len_y_w,len_z_w,len_t,
                    x_index_w,y_index_w,z_index_w,t_index)
        dx2 = delta_t*u_loc1
        dy2 = delta_t*v_loc1
        dz2 = delta_t*w_loc1

        u_loc2 = quadralinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2,startt + 0.5*delta_t,
                    u_field,
                    x_u,y_u,z_u,t,len_x_u,len_y_u,len_z_u,len_t,
                    x_index_u,y_index_u,z_index_u,t_index)
        v_loc2 = quadralinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2,startt + 0.5*delta_t,
                    v_field,
                    x_v,y_v,z_v,t,len_x_v,len_y_v,len_z_v,len_t,
                    x_index_v,y_index_v,z_index_v,t_index)
        w_loc2 = quadralinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2,startt + 0.5*delta_t,
                    w_field,
                    x_w,y_w,z_w,t,len_x_w,len_y_w,len_z_w,len_t,
                    x_index_w,y_index_w,z_index_w,t_index)
        dx3 = delta_t*u_loc2
        dy3 = delta_t*v_loc2
        dz3 = delta_t*w_loc2

        u_loc3 = quadralinear_interp(startx + dx3,starty + dy3,startz + dz3,startt + delta_t,
                    u_field,
                    x_u,y_u,z_u,t,len_x_u,len_y_u,len_z_u,len_t,
                    x_index_u,y_index_u,z_index_u,t_index)
        v_loc3 = quadralinear_interp(startx + dx3,starty + dy3,startz + dz3,startt + delta_t,
                    v_field,
                    x_v,y_v,z_v,t,len_x_v,len_y_v,len_z_v,len_t,
                    x_index_v,y_index_v,z_index_v,t_index)
        w_loc3 = quadralinear_interp(startx + dx3,starty + dy3,startz + dz3,startt + delta_t,
                    w_field,
                    x_w,y_w,z_w,t,len_x_w,len_y_w,len_z_w,len_t,
                    x_index_w,y_index_w,z_index_w,t_index)
        dx4 = delta_t*u_loc3
        dy4 = delta_t*v_loc3
        dz4 = delta_t*w_loc3

        #recycle the "start_" variables to keep the code clean
        startx = startx + (dx1 + 2*dx2 + 2*dx3 + dx4)/6
        starty = starty + (dy1 + 2*dy2 + 2*dy3 + dy4)/6
        startz = startz + (dz1 + 2*dz2 + 2*dz3 + dz4)/6
        t_RK += delta_t
        i += 1

        x_stream[i] = startx
        y_stream[i] = starty
        z_stream[i] = startz
        t_stream[i] = t_RK

  
    u_netcdf_filehandle.close()
    v_netcdf_filehandle.close()
    w_netcdf_filehandle.close()

    return x_stream,y_stream,z_stream,t_stream





def bilinear_interp(x0,y0,field,x,y,len_x,len_y):
  """Do bilinear interpolation of a field to get nice accurate streamlines. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles.

  x0,y0 are the points to interpolate to.
  """

  # Compute indeces at location
  x_index = np.searchsorted(x,x0)
  if x_index == 0:
    raise ValueError('Given x location is outside the model grid - too small')
  elif x_index == len_x:
    raise ValueError('Given x location is outside the model grid - too big')
    
  y_index = np.searchsorted(y,y0)
  if y_index == 0:
    raise ValueError('Given y location is outside the model grid - too small')
  elif y_index == len_y:
    raise ValueError('Given y location is outside the model grid - too big')
  
  #print 'x index = ' + str(x_index)
  #print 'y index = ' + str(y_index)
    
  field_interp = actual_bilinear_interp(field,x0,y0,x,y,len_x,len_y,x_index,y_index)
  return field_interp
  
@jit
def actual_bilinear_interp(field,x0,y0,x,y,len_x,len_y,x_index,y_index):
    """This is a numba accelerated bilinear interpolation. The @jit decorator just above this function causes it to be compiled just before it is run. This introduces a small, Order (1 second),overhead the first time, but not on subsequent calls. 
    """
    field_interp = ((field[y_index-1,x_index-1]*(x[x_index] - x0)*(y[y_index] - y0) + 
       field[y_index-1,x_index]*(x0 - x[x_index-1])*(y[y_index] - y0) +
       field[y_index,x_index]*(x[x_index] - x0)*(y0 - y[y_index-1]) + 
       field[y_index,x_index]*(x0 - x[x_index-1])*(y0 - y[y_index-1]))/
       ((y[y_index] - y[y_index-1])*(x[x_index] - x[x_index-1]))) 
    return field_interp

  
def trilinear_interp(x0,y0,z0,field,x,y,z,len_x,len_y,len_z):
  """Do trilinear interpolation of the velocity field in three spatial dimensions to get nice accurate streamlines. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles.

  x0,y0, and z0 represent the point to interpolate to.
  """

  # Compute indices at location given
  x_index = np.searchsorted(x,x0)
  if x_index == 0:
    raise ValueError('Given x location is outside the model grid - too small')
  elif x_index == len_x:
    raise ValueError('Given x location is outside the model grid - too big')
    
  y_index = np.searchsorted(y,y0)
  if y_index == 0:
    raise ValueError('Given y location is outside the model grid - too small')
  elif y_index == len_y:
    raise ValueError('Given y location is outside the model grid - too big')
  
  # np.searchsorted only works for positive arrays :/
  if z0 < 0:
        z0 = -z0
        z = -z
  z_index = np.searchsorted(z,z0)
  if z_index == 0:
    raise ValueError('Given z location is outside the model grid - too small')
  elif z_index == len_z:
    raise ValueError('Given z location is outside the model grid - too big')

  #print 'x index = ' + str(x_index)
  #print 'y index = ' + str(y_index)
    
  field_interp = actual_trilinear_interp(field,x0,y0,z0,x_index,y_index,z_index,x,y,z)

  return field_interp

@jit
def actual_trilinear_interp(field,x0,y0,z0,x_index,y_index,z_index,x,y,z):
    
    field_interp = ((field[z_index-1,y_index-1,x_index-1]*
                (x[x_index] - x0)*(y[y_index] - y0)*(z[z_index] - z0) + 
            field[z_index,y_index-1,x_index-1]*
                (x[x_index] - x0)*(y[y_index] - y0)*(z0 - z[z_index-1]) + 
            field[z_index,y_index,x_index-1]*
                (x[x_index] - x0)*(y0 - y[y_index-1])*(z0 - z[z_index-1]) + 
            field[z_index-1,y_index,x_index-1]*
                (x[x_index] - x0)*(y0 - y[y_index-1])*(z[z_index] - z0) + 
            field[z_index-1,y_index-1,x_index]*
                (x0 - x[x_index-1])*(y[y_index] - y0)*(z[z_index] - z0) + 
            field[z_index,y_index-1,x_index]*
                (x0 - x[x_index-1])*(y[y_index] - y0)*(z0 - z[z_index-1]) + 
            field[z_index,y_index,x_index]*
                (x0 - x[x_index-1])*(y0 - y[y_index-1])*(z0 - z[z_index-1]) + 
            field[z_index-1,y_index,x_index]*
                (x0 - x[x_index-1])*(y0 - y[y_index-1])*(z[z_index] - z0))/
         ((z[z_index] - z[z_index-1])*(y[y_index] - y[y_index-1])*(x[x_index] - x[x_index-1]))) 
    return field_interp

def quadralinear_interp(x0,y0,z0,t0,
                        field,
                        x,y,z,t,
                        len_x,len_y,len_z,len_t,
                        x_index,y_index,z_index,t_index):
  """ Do quadralinear interpolation of the velocity field in three spatial dimensions and one temporal dimension to get nice accurate streaklines. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles.

  location is an array of the point to interpolate to in four dimensional space-time (x_int,y_int,z_int,t_int).

  The velocity field needs to be passed as either a 4D variable, which is big and expensive, or as a handle to the place where it can be obtained from disk.
  
  x,y,z,t are vectors of these dimensions in netcdf_filename.
  """

  # These searchsorted calls are to check if the location has crossed a grid cell. 
  # With very small time steps they're probably irrelevant.
    
  # Compute indices at location
  x_index_shifted = np.searchsorted(x,x0) - x_index + 2
  if x_index == 0:
    raise ValueError('Given x location is outside the truncated field - too small')
  elif x_index == 4:
    raise ValueError('Given x location is outside the truncated field - too big')
    
  y_index_shifted = np.searchsorted(y,y0) - y_index + 2
  if y_index_shifted == 0:
    raise ValueError('Given y location is outside the truncated field - too small')
  elif y_index_shifted == 4:
    raise ValueError('Given y location is outside the truncated field - too big')
  
  # np.searchsorted only works for positive arrays, so z needs to be positive :/
  if z0 < 0:
        z0 = -z0
        z = -z
  z_index_shifted = np.searchsorted(z,z0) - z_index + 2
  if z_index_shifted == 0:
    raise ValueError('Given z location is outside the truncated field - too small')
  elif z_index_shifted == 4:
    raise ValueError('Given z location is outside the truncated field - too big')

  t_index_shifted = np.searchsorted(t,t0) - t_index + 2
  if t_index_shifted == 0:
    raise ValueError('Given t location is outside the truncated field - too small')
  elif t_index_shifted == 4:
    raise ValueError('Given t location is outside the truncated field - too big')
  

  field_interp = actual_quadralinear_interp(field[t_index_shifted-1:t_index_shifted+1,
                       z_index_shifted-1:z_index_shifted+1,
                       y_index_shifted-1:y_index_shifted+1,
                       x_index_shifted-1:x_index_shifted+1],
                        x0,y0,z0,t0,
                        x_index,y_index,z_index,t_index,
                        x,y,z,t)

  return field_interp

@jit
def actual_quadralinear_interp(field,x0,y0,z0,t0,
                               x_index,y_index,z_index,t_index,
                               x,y,z,t):

  field_interp = ((field[0,0,0,0]*
                    (x[x_index] - x0)*(y[y_index] - y0)*(z[z_index] - z0)*(t[t_index] - t0) + 
                field[0,1,0,0]*
                    (x[x_index] - x0)*(y[y_index] - y0)*(z0 - z[z_index-1])*(t[t_index] - t0) + 
                field[0,1,1,0]*
                    (x[x_index] - x0)*(y0 - y[y_index-1])*(z0 - z[z_index-1])*(t[t_index] - t0) + 
                field[0,0,1,0]*
                    (x[x_index] - x0)*(y0 - y[y_index-1])*(z[z_index] - z0)*(t[t_index] - t0) + 
                field[0,0,0,1]*
                    (x0 - x[x_index-1])*(y[y_index] - y0)*(z[z_index] - z0)*(t[t_index] - t0) + 
                field[0,1,0,1]*
                    (x0 - x[x_index-1])*(y[y_index] - y0)*(z0 - z[z_index-1])*(t[t_index] - t0) +        
                field[0,1,1,1]*
                    (x0 - x[x_index-1])*(y0 - y[y_index-1])*(z0 - z[z_index-1])*(t[t_index] - t0) + 
                field[0,0,1,1]*
                    (x0 - x[x_index-1])*(y0 - y[y_index-1])*(z[z_index] - z0)*(t[t_index] - t0) +
                    
                field[1,0,0,0]*
                    (x[x_index] - x0)*(y[y_index] - y0)*(z[z_index] - z0)*(t0 - t[t_index-1]) + 
                field[1,1,0,0]*
                    (x[x_index] - x0)*(y[y_index] - y0)*(z0 - z[z_index-1])*(t0 - t[t_index-1]) + 
                field[1,1,1,0]*
                    (x[x_index] - x0)*(y0 - y[y_index-1])*(z0 - z[z_index-1])*(t0 - t[t_index-1]) + 
                field[1,0,1,0]*
                    (x[x_index] - x0)*(y0 - y[y_index-1])*(z[z_index] - z0)*(t0 - t[t_index-1]) + 
                field[1,0,0,1]*
                    (x0 - x[x_index-1])*(y[y_index] - y0)*(z[z_index] - z0)*(t0 - t[t_index-1]) + 
                field[1,1,0,1]*
                    (x0 - x[x_index-1])*(y[y_index] - y0)*(z0 - z[z_index-1])*(t0 - t[t_index-1]) + 
                field[1,1,1,1]*
                    (x0 - x[x_index-1])*(y0 - y[y_index-1])*(z0 - z[z_index-1])*(t0 - t[t_index-1]) + 
                field[1,0,1,1]*
                    (x0 - x[x_index-1])*(y0 - y[y_index-1])*(z[z_index] - z0)*(t0 - t[t_index-1]))/
             ((t[t_index] - t[t_index-1])*(z[z_index] - z[z_index-1])*
              (y[y_index] - y[y_index-1])*(x[x_index] - x[x_index-1])))
    
  return field_interp





def indices_and_field(x,y,z,
                        startx,starty,startz,t_index,
                        len_x,len_y,len_z,len_t,
                        netcdf_filehandle,variable):
            
            x_index = np.searchsorted(x,startx)
            if x_index == 0:
                raise ValueError('Given x location is outside the model grid - too small')
            elif x_index == len_x:
                raise ValueError('Given x location is outside the model grid - too big')

            y_index = np.searchsorted(y,starty)
            if y_index == 0:
                raise ValueError('Given y location is outside the model grid - too small')
            elif y_index == len_y:
                raise ValueError('Given y location is outside the model grid - too big')

            # np.searchsorted only works for positive arrays :/
            if startz < 0:
                startz = -startz
                z = -z
            z_index = np.searchsorted(z,startz)
            if z_index == 0:
                raise ValueError('Given z location is outside the model grid - too small')
            elif z_index == len_z:
                raise ValueError('Given z location is outside the model grid - too big')



            field = netcdf_filehandle.variables[variable][t_index-2:t_index+3,
                           z_index-2:z_index+3,
                           y_index-2:y_index+3,
                           x_index-2:x_index+3]
            
            return field,x_index,y_index,z_index
            
class InputError(Exception):
	pass