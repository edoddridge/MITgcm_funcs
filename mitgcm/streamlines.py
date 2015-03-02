"""!
A collection of functions for streamlines and related shennanigans.

Streamlines
====================

Functions for creating and analysing streamlines.

Streamlines are defined to be the path that a parcel of fluid would follow when advected by an unchanging velocity field - the velocities are constant in time.

streaklines are defined as the path that a parcel of fluid would follow in the actual flow - the velocity fields change with time.
"""


import numpy as np
import netCDF4
import numba


def stream2(u,v,
            startx,starty,
            grid_object,
            t_int=2592000,delta_t=3600):
    """!A two-dimensional streamline solver. The velocity fields *must* be two dimensional and not vary in time.
    """
    x_u = grid_object['Xp1'][:]
    y_u = grid_object['Y'][:]

    x_v = grid_object['X'][:]
    y_v = grid_object['Yp1'][:]

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
      i += 1

      x_stream[i] = startx
      y_stream[i] = starty
      t_stream[i] = t
      
      
    return x_stream,y_stream,t_stream


def stream3(u,v,w,
            startx,starty,startz,
            grid_object,
            x_v='None',y_v='None',z_v='None',
            x_w='None',y_w='None',z_w='None',
            t_int=2592000,delta_t=3600):
    """!A three-dimensional streamline solver. The velocity fields must be three dimensional and not vary in time.
    """
    x_u = grid_object['Xp1'][:]
    y_u = grid_object['Y'][:]
    z_u = grid_object['Z'][:]

    x_v = grid_object['X'][:]
    y_v = grid_object['Yp1'][:]
    z_v = grid_object['Z'][:]

    x_w = grid_object['X'][:]
    y_w = grid_object['Y'][:]
    z_w = grid_object['Zl'][:]
        
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
            grid_object,            
            u_netcdf_variable='UVEL',
            v_netcdf_variable='VVEL',
            w_netcdf_variable='WVEL',
            t_int=3.1e5,delta_t=3600):
    """!A three-dimensional lagrangian particle tracker. The velocity fields must be four dimensional (three spatial, one temporal) and have units of m/s.
    It should work to track particles forwards or backwards in time (set delta_t <0 for backwards in time). But, be warned, backwards in time hasn't been tested yet.
    
    Because this is a very large amount of data, the fields are passed as netcdffile handles.
    
    The variables are:
    * ?_netcdf_filename = name of the netcdf file with ?'s data in it.
    * start? = intial value for x, y, z, or t.
    * t = vector of time levels that are contained in the velocity data.
    * grid_object is m.grid if you followed the standard naming conventions.
    * ?_netcdf_variable = name of the "?" variable field in the netcdf file.
    * t_int = length of time to track particles for, in seconds
    * delta_t = timestep for particle tracking algorithm, in seconds. This can be positive or negative.
    """

    x_u = grid_object['Xp1'][:]
    y_u = grid_object['Y'][:]
    z_u = grid_object['Z'][:]

    x_v = grid_object['X'][:]
    y_v = grid_object['Yp1'][:]
    z_v = grid_object['Z'][:]

    x_w = grid_object['X'][:]
    y_w = grid_object['Y'][:]
    z_w = grid_object['Zl'][:]

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
    t_stream = np.ones((int(t_int/delta_t)+2))*startt

    t_RK = startt #set the initial time to be the given start time
    z_RK = startz
    y_RK = starty
    x_RK = startx
    
    i=0
    
    u_netcdf_filehandle = netCDF4.Dataset(u_netcdf_filename)
    v_netcdf_filehandle = netCDF4.Dataset(v_netcdf_filename)
    w_netcdf_filehandle = netCDF4.Dataset(w_netcdf_filename)
    
    t_index = np.searchsorted(t,t_RK)
    t_index_new = np.searchsorted(t,t_RK) # this is later used to test if new data needs to be read in.
    if t_index == 0:
        raise ValueError('Given time value is outside the given velocity fields - too small')
    elif t_index == len_t:
        raise ValueError('Given time value is outside the given velocity fields - too big')
    
                
    # load fields in ready for the first run through the loop
    #  u
    u_field,x_index_u,y_index_u,z_index_u = indices_and_field(x_u,y_u,z_u,
                                                x_RK,y_RK,z_RK,t_index,
                                                len_x_u,len_y_u,len_z_u,len_t,
                                                u_netcdf_filehandle,u_netcdf_variable)
    u_field,x_index_u_new,y_index_u_new,z_index_u_new = indices_and_field(x_u,y_u,z_u,
                                                x_RK,y_RK,z_RK,t_index,
                                                len_x_u,len_y_u,len_z_u,len_t,
                                                u_netcdf_filehandle,u_netcdf_variable)
    #  v
    v_field,x_index_v,y_index_v,z_index_v = indices_and_field(x_v,y_v,z_v,
                                                x_RK,y_RK,z_RK,t_index,
                                                len_x_v,len_y_v,len_z_v,len_t,
                                                v_netcdf_filehandle,v_netcdf_variable)
    v_field,x_index_v_new,y_index_v_new,z_index_v_new = indices_and_field(x_v,y_v,z_v,
                                                x_RK,y_RK,z_RK,t_index,
                                                len_x_v,len_y_v,len_z_v,len_t,
                                                v_netcdf_filehandle,v_netcdf_variable)

    #  w
    w_field,x_index_w,y_index_w,z_index_w = indices_and_field(x_w,y_w,z_w,
                                                x_RK,y_RK,z_RK,t_index,
                                                len_x_w,len_y_w,len_z_w,len_t,
                                                w_netcdf_filehandle,w_netcdf_variable)
    
    w_field,x_index_w_new,y_index_w_new,z_index_w_new = indices_and_field(x_w,y_w,z_w,
                                                x_RK,y_RK,z_RK,t_index,
                                                len_x_w,len_y_w,len_z_w,len_t,
                                                w_netcdf_filehandle,w_netcdf_variable)
    
    
    # Runge-Kutta fourth order method to estimate next position.
    while i < np.fabs(t_int/delta_t):
    #t_RK < t_int + startt:
        
        # Compute indices at location given
        
        if (y_index_u_new==y_index_u and 
            x_index_u_new==x_index_u and 
            z_index_u_new==z_index_u and
            
            y_index_v_new==y_index_v and 
            x_index_v_new==x_index_v and 
            z_index_v_new==z_index_v and 
            
            y_index_w_new==y_index_w and 
            x_index_w_new==x_index_w and 
            z_index_w_new==z_index_w and 

            t_index_new == t_index):
            # the particle hasn't moved out of the grid cell it was in.
            # So the loaded field is fine; there's no need to reload it.
            pass
        else:
            t_index = np.searchsorted(t,t_RK)
            if t_index == 0:
                raise ValueError('Given time value is outside the given velocity fields - too small')
            elif t_index == len_t:
                raise ValueError('Given time value is outside the given velocity fields - too big')


            # for u

            u_field,x_index_u,y_index_u,z_index_u = indices_and_field(x_u,y_u,z_u,
                                                        x_RK,y_RK,z_RK,t_index,
                                                        len_x_u,len_y_u,len_z_u,len_t,
                                                        u_netcdf_filehandle,u_netcdf_variable)
            # for v
            v_field,x_index_v,y_index_v,z_index_v = indices_and_field(x_v,y_v,z_v,
                                                        x_RK,y_RK,z_RK,t_index,
                                                        len_x_v,len_y_v,len_z_v,len_t,
                                                        v_netcdf_filehandle,v_netcdf_variable)

            # for w
            w_field,x_index_w,y_index_w,z_index_w = indices_and_field(x_w,y_w,z_w,
                                                        x_RK,y_RK,z_RK,t_index,
                                                        len_x_w,len_y_w,len_z_w,len_t,
                                                        w_netcdf_filehandle,w_netcdf_variable)



        # Interpolate velocities to initial location
        u_loc = quadralinear_interp(x_RK,y_RK,z_RK,t_RK,
                    u_field,
                    x_u,y_u,z_u,t,
                    len_x_u,len_y_u,len_z_u,len_t,
                    x_index_u,y_index_u,z_index_u,t_index)
        v_loc = quadralinear_interp(x_RK,y_RK,z_RK,t_RK,
                    v_field,
                    x_v,y_v,z_v,t,len_x_v,len_y_v,len_z_v,len_t,
                    x_index_v,y_index_v,z_index_v,t_index)
        w_loc = quadralinear_interp(x_RK,y_RK,z_RK,t_RK,
                    w_field,
                    x_w,y_w,z_w,t,len_x_w,len_y_w,len_z_w,len_t,
                    x_index_w,y_index_w,z_index_w,t_index)
        dx1 = delta_t*u_loc
        dy1 = delta_t*v_loc
        dz1 = delta_t*w_loc

        u_loc1 = quadralinear_interp(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,t_RK + 0.5*delta_t,
                    u_field,
                    x_u,y_u,z_u,t,len_x_u,len_y_u,len_z_u,len_t,
                    x_index_u,y_index_u,z_index_u,t_index)
        v_loc1 = quadralinear_interp(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,t_RK + 0.5*delta_t,
                    v_field,
                    x_v,y_v,z_v,t,len_x_v,len_y_v,len_z_v,len_t,
                    x_index_v,y_index_v,z_index_v,t_index)
        w_loc1 = quadralinear_interp(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,t_RK + 0.5*delta_t,
                    w_field,
                    x_w,y_w,z_w,t,len_x_w,len_y_w,len_z_w,len_t,
                    x_index_w,y_index_w,z_index_w,t_index)
        dx2 = delta_t*u_loc1
        dy2 = delta_t*v_loc1
        dz2 = delta_t*w_loc1

        u_loc2 = quadralinear_interp(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,t_RK + 0.5*delta_t,
                    u_field,
                    x_u,y_u,z_u,t,len_x_u,len_y_u,len_z_u,len_t,
                    x_index_u,y_index_u,z_index_u,t_index)
        v_loc2 = quadralinear_interp(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,t_RK + 0.5*delta_t,
                    v_field,
                    x_v,y_v,z_v,t,len_x_v,len_y_v,len_z_v,len_t,
                    x_index_v,y_index_v,z_index_v,t_index)
        w_loc2 = quadralinear_interp(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,t_RK + 0.5*delta_t,
                    w_field,
                    x_w,y_w,z_w,t,len_x_w,len_y_w,len_z_w,len_t,
                    x_index_w,y_index_w,z_index_w,t_index)
        dx3 = delta_t*u_loc2
        dy3 = delta_t*v_loc2
        dz3 = delta_t*w_loc2

        u_loc3 = quadralinear_interp(x_RK + dx3,y_RK + dy3,z_RK + dz3,t_RK + delta_t,
                    u_field,
                    x_u,y_u,z_u,t,len_x_u,len_y_u,len_z_u,len_t,
                    x_index_u,y_index_u,z_index_u,t_index)
        v_loc3 = quadralinear_interp(x_RK + dx3,y_RK + dy3,z_RK + dz3,t_RK + delta_t,
                    v_field,
                    x_v,y_v,z_v,t,len_x_v,len_y_v,len_z_v,len_t,
                    x_index_v,y_index_v,z_index_v,t_index)
        w_loc3 = quadralinear_interp(x_RK + dx3,y_RK + dy3,z_RK + dz3,t_RK + delta_t,
                    w_field,
                    x_w,y_w,z_w,t,len_x_w,len_y_w,len_z_w,len_t,
                    x_index_w,y_index_w,z_index_w,t_index)
        dx4 = delta_t*u_loc3
        dy4 = delta_t*v_loc3
        dz4 = delta_t*w_loc3

        #recycle the variables to keep the code clean
        x_RK = x_RK + (dx1 + 2*dx2 + 2*dx3 + dx4)/6
        y_RK = y_RK + (dy1 + 2*dy2 + 2*dy3 + dy4)/6
        z_RK = z_RK + (dz1 + 2*dz2 + 2*dz3 + dz4)/6
        t_RK += delta_t
        i += 1

        x_stream[i] = x_RK
        y_stream[i] = y_RK
        z_stream[i] = z_RK
        t_stream[i] = t_RK
        
        t_index_new = np.searchsorted(t,t_RK)
        x_index_w_new = np.searchsorted(x_w,x_RK)
        y_index_w_new = np.searchsorted(y_w,y_RK)
        if z_RK < 0:
            z_index_w_new = np.searchsorted(-z_w,-z_RK)
        else:
            z_index_w_new = np.searchsorted(z_w,z_RK)
            
        x_index_v_new = np.searchsorted(x_v,x_RK)
        y_index_v_new = np.searchsorted(y_v,y_RK)
        if z_RK < 0:
            z_index_v_new = np.searchsorted(-z_v,-z_RK)
        else:
            z_index_v_new = np.searchsorted(z_v,z_RK)
            
        x_index_u_new = np.searchsorted(x_u,x_RK)
        y_index_u_new = np.searchsorted(y_u,y_RK)
        if z_RK < 0:
            z_index_u_new = np.searchsorted(-z_u,-z_RK)
        else:
            z_index_u_new = np.searchsorted(z_u,z_RK)

  
    u_netcdf_filehandle.close()
    v_netcdf_filehandle.close()
    w_netcdf_filehandle.close()

    return x_stream,y_stream,z_stream,t_stream






def bilinear_interp(x0,y0,field,x,y,len_x,len_y):
  """!Do bilinear interpolation of a field to get nice accurate streamlines. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles.

  x0,y0 are the points to interpolate to.
  """

  # Compute indeces at location
  x_index = np.searchsorted(x,x0)
  if x_index == 0:
    x_index =1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('x location ', str(x0), ' is outside the model grid - too small')
  elif x_index == len_x:
    x_index =len_x - 1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('x location ', str(x0), ' is outside the model grid - too big')
    
  y_index = np.searchsorted(y,y0)
  if y_index == 0:
    y_index =1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('y location ', str(y0), ' is outside the model grid - too small')
  elif y_index == len_y:
    y_index =len_y - 1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('y location ', str(y0), ' is outside the model grid - too big')
  
  #print 'x index = ' + str(x_index)
  #print 'y index = ' + str(y_index)
    
  field_interp = actual_bilinear_interp(field,x0,y0,x,y,len_x,len_y,x_index,y_index)
  return field_interp
  
@numba.jit
def actual_bilinear_interp(field,x0,y0,x,y,len_x,len_y,x_index,y_index):
    """!This is a numba accelerated bilinear interpolation. The @numba.jit decorator just above this function causes it to be compiled just before it is run. This introduces a small, Order(1 second), overhead the first time, but not on subsequent calls. 
    """
    field_interp = ((field[y_index-1,x_index-1]*(x[x_index] - x0)*(y[y_index] - y0) + 
       field[y_index-1,x_index]*(x0 - x[x_index-1])*(y[y_index] - y0) +
       field[y_index,x_index]*(x[x_index] - x0)*(y0 - y[y_index-1]) + 
       field[y_index,x_index]*(x0 - x[x_index-1])*(y0 - y[y_index-1]))/
       ((y[y_index] - y[y_index-1])*(x[x_index] - x[x_index-1]))) 
    return field_interp

  
def trilinear_interp(x0,y0,z0,field,x,y,z,len_x,len_y,len_z):
  """!Do trilinear interpolation of the velocity field in three spatial dimensions to get nice accurate streamlines. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles. It also requires that the entire field can be held in memory at the same time.

  x0,y0, and z0 represent the point to interpolate to.
  """

  # Compute indices at location given
  x_index = np.searchsorted(x,x0)
  if x_index == 0:
    x_index =1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('x location ', str(x0), ' is outside the model grid - too small')
  elif x_index == len_x:
    x_index =len_x - 1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('x location ', str(x0), ' is outside the model grid - too big')
    
  y_index = np.searchsorted(y,y0)
  if y_index == 0:
    y_index =1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('y location ', str(y0), ' is outside the model grid - too small')
  elif y_index == len_y:
    y_index =len_y - 1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('y location ', str(y0), ' is outside the model grid - too big')
  
  # np.searchsorted only works for positive arrays :/
  if z0 < 0:
        z0 = -z0
        z = -z
  z_index = np.searchsorted(z,z0)
  if z_index == 0:
    z_index =1 # a dirty hack to deal with streamlines coming near the surface
    #raise ValueError('z location ', str(z0), ' is outside the model grid - too small')
  elif z_index == len_z:
    z_index = len_z - 1 # a dirty hack to deal with streamlines coming near the bottom
    #raise ValueError('z location ', str(z0), ' is outside the model grid - too big')

  #print 'x index = ' + str(x_index)
  #print 'y index = ' + str(y_index)
    
  field_interp = actual_trilinear_interp(field,x0,y0,z0,x_index,y_index,z_index,x,y,z)

  return field_interp

@numba.jit
def actual_trilinear_interp(field,x0,y0,z0,x_index,y_index,z_index,x,y,z):
    """!This is a numba accelerated trilinear interpolation. The @numba.jit decorator just above this function causes it to be compiled just before it is run. This introduces a small, Order(1 second), overhead the first time, but not on subsequent calls. 
    """   
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
  """! Do quadralinear interpolation of the velocity field in three spatial dimensions and one temporal dimension to get nice accurate streaklines. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles.

  x0,y0,z0, and t0 represent the point to interpolate to.

  The velocity field, "field", is passed as a truncated 4D field.
  
  x,y,z,t are vectors of these dimensions in netcdf_filename.
  """

  # These searchsorted calls are to check if the location has crossed a grid cell. 
  # With very small time steps they're probably irrelevant.
    
  # Compute indices at location
  x_index_shifted = np.searchsorted(x,x0) - x_index + 2
  #if x_index == 0:
  #  raise ValueError('Given x location is outside the truncated field - too small. This error should never be seen.')
  #elif x_index == 4:
  #  raise ValueError('Given x location is outside the truncated field - too big. This error should never be seen.')
    
  y_index_shifted = np.searchsorted(y,y0) - y_index + 2
  #if y_index_shifted == 0:
  #  raise ValueError('Given y location is outside the truncated field - too small. This error should never be seen.')
  #elif y_index_shifted == 4:
  #  raise ValueError('Given y location is outside the truncated field - too big. This error should never be seen.')
  
  # np.searchsorted only works for positive arrays, so z needs to be positive :/
  if z0 < 0:
        z0 = -z0
        z = -z
  z_index_shifted = np.searchsorted(z,z0) - z_index + 2
  #if z_index_shifted == 0:
  #  raise ValueError('Given z location is outside the truncated field - too small. This error should never be seen.')
  #elif z_index_shifted == 4:
  #  raise ValueError('Given z location is outside the truncated field - too big. This error should never be seen.')

  t_index_shifted = np.searchsorted(t,t0) - t_index + 2
  #if t_index_shifted == 0:
  #  raise ValueError('Given t location is outside the truncated field - too small. This error should never be seen.')
  #elif t_index_shifted == 3:
  #  raise ValueError('Given t location is outside the truncated field - too big. This error should never be seen.')
  

  field_interp = actual_quadralinear_interp(field[t_index_shifted-1:t_index_shifted+1,
                       z_index_shifted-1:z_index_shifted+1,
                       y_index_shifted-1:y_index_shifted+1,
                       x_index_shifted-1:x_index_shifted+1],
                        x0,y0,z0,t0,
                        x_index,y_index,z_index,t_index,
                        x,y,z,t)

  return field_interp

@numba.jit
def actual_quadralinear_interp(field,x0,y0,z0,t0,
                               x_index,y_index,z_index,t_index,
                               x,y,z,t):
  """!This is a numba accelerated quadralinear interpolation. The @numba.jit decorator just above this function causes it to be compiled just before it is run. This introduces a small, Order(1 second), overhead the first time, but not on subsequent
  calls. 
  """   
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
            """!A helper function to extract a small 4D hypercube of data from a netCDF file. This isn't intended to be used on its own."""
            
            # Compute indices at location given
            x_index = np.searchsorted(x,x0)
            if x_index == 0:
            x_index =1 # a dirty hack to deal with streamlines coming near the edge
            #raise ValueError('x location ', str(x0), ' is outside the model grid - too small')
            elif x_index == len_x:
            x_index =len_x - 1 # a dirty hack to deal with streamlines coming near the edge
            #raise ValueError('x location ', str(x0), ' is outside the model grid - too big')

            y_index = np.searchsorted(y,y0)
            if y_index == 0:
            y_index =1 # a dirty hack to deal with streamlines coming near the edge
            #raise ValueError('y location ', str(y0), ' is outside the model grid - too small')
            elif y_index == len_y:
            y_index =len_y - 1 # a dirty hack to deal with streamlines coming near the edge
            #raise ValueError('y location ', str(y0), ' is outside the model grid - too big')

            # np.searchsorted only works for positive arrays :/
            if z0 < 0:
                z0 = -z0
                z = -z
            z_index = np.searchsorted(z,z0)
            if z_index == 0:
            z_index =1 # a dirty hack to deal with streamlines coming near the surface
            #raise ValueError('z location ', str(z0), ' is outside the model grid - too small')
            elif z_index == len_z:
            z_index = len_z - 1 # a dirty hack to deal with streamlines coming near the bottom
            #raise ValueError('z location ', str(z0), ' is outside the model grid - too big')



            field = netcdf_filehandle.variables[variable][t_index-2:t_index+3,
                           z_index-2:z_index+3,
                           y_index-2:y_index+3,
                           x_index-2:x_index+3]
            
            return field,x_index,y_index,z_index
            
            
def extract_along_path4D(path_x,path_y,path_z,path_t,
            t,x,y,z,
            netcdf_filename='netcdf file with variable of interest',
            netcdf_variable='momVort3'):
    """!extract the value of a field along a path through a time varying, 3 dimensional field. The field must be in a NetCDF file, since it is assumed to be 4D and very large.
    """

    t_index = np.searchsorted(t,path_t[0])
    t_index_new = np.searchsorted(t,path_t[0]) # this is later used to test if new data needs to be read in.
        
    len_x = len(x)
    len_y = len(y)
    len_z = len(z)
    len_t = len(t)

    
    netcdf_filehandle = netCDF4.Dataset(netcdf_filename)  
    field,x_index,y_index,z_index = indices_and_field(x,y,z,
                                            path_x[0],path_y[0],path_z[0],t_index,
                                            len_x,len_y,len_z,len_t,
                                            netcdf_filehandle,netcdf_variable)

    field,x_index_new,y_index_new,z_index_new = indices_and_field(x,y,z,
                                            path_x[0],path_y[0],path_z[0],t_index,
                                            len_x,len_y,len_z,len_t,
                                            netcdf_filehandle,netcdf_variable)   



    path_variable = np.zeros((path_x.shape))
    
    i=0
    

    if t_index == 0:
        raise ValueError('Given start time is outside the given variable field - too small')
    elif t_index == len_t:
        raise ValueError('Given start time is outside the given variable field - too big')
    
    
    for i in xrange(0,len(path_t)):
        
        if (y_index_new==y_index and 
            x_index_new==x_index and 
            z_index_new==z_index and

            t_index_new == t_index):
            # the particle hasn't moved out of the grid cell it was in.
            # So the loaded field is fine; there's no need to reload it.
            pass
        else:

            t_index = np.searchsorted(t,path_t[i])
            if t_index == 0:
                raise ValueError('Time value is outside the given variable field - too small')
            elif t_index == len_t:
                raise ValueError('Time value is outside the given variable field - too big')

            field,x_index,y_index,z_index = indices_and_field(x,y,z,
                                            path_x[i],path_y[i],path_z[i],t_index,
                                            len_x,len_y,len_z,len_t,
                                            netcdf_filehandle,netcdf_variable)


        # Interpolate field to  location
        field_at_loc = quadralinear_interp(path_x[i],path_y[i],path_z[i],path_t[i],
                    field,
                    x,y,z,t,
                    len_x,len_y,len_z,len_t,
                    x_index,y_index,z_index,t_index)
      

        path_variable[i] = field_at_loc

        t_index_new = np.searchsorted(t,path_t[i])
        x_index_new = np.searchsorted(x,path_x[i])
        y_index_new = np.searchsorted(y,path_y[i])
        if path_z[i] < 0:
            z_index_new = np.searchsorted(-z,-path_z[i])
        else:
            z_index_new = np.searchsorted(z,path_z[i])
            


    netcdf_filehandle.close()
    
    return path_variable



def extract_along_path3D(path_x,path_y,path_z,
            x,y,z,field):
    """!Extract the value of a field along a path through a 3 dimensional field. The field must be passed as an array. Currently time varying fields are not supported.
    """
        
    len_x = len(x)
    len_y = len(y)
    len_z = len(z)

    path_variable = np.zeros((path_x.shape))
    
    for i in xrange(0,len(path_x)):

        # Interpolate field to  location
        path_variable[i] = trilinear_interp(path_x[i],path_y[i],path_z[i],field,x,y,z,len_x,len_y,len_z)
                
    return path_variable



def extract_along_path2D(path_x,path_y,
            x,y,field):
    """!Extract the value of a field along a path through a 2 dimensional field. The field must be passed as an array. Currently time varying fields are not supported.
    """
        
    len_x = len(x)
    len_y = len(y)

    path_variable = np.zeros((path_x.shape))
    
    for i in xrange(0,len(path_x)):

        # Interpolate field to  location
        path_variable[i] = bilinear_interp(path_x[i],path_y[i],field,x,y,len_x,len_y)
                
    return path_variable