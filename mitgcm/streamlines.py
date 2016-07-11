"""!
A collection of functions for streamlines and related shennanigans.

Streamlines
====================

Functions for creating and analysing streamlines.

These functions work on cartesian and spherical polar grids - other grids, such as cubed sphere, are not supported.

Streamlines are defined to be the path that a parcel of fluid would follow when advected by an unchanging velocity field - the velocities are constant in time.

Streaklines are defined as the path that a parcel of fluid would follow in the actual flow - the velocity fields change with time.

"""


import numpy as np
import netCDF4
import numba
import copy
import scipy.interpolate
from . import functions

def stream2(u,v,
            startx,starty,
            grid_object,
            t_max=2592000,delta_t=3600,
            u_grid_loc='U',v_grid_loc='V'):
    """!A two-dimensional streamline solver. The velocity fields *must* be two dimensional and not vary in time.
    X_grid_loc variables specify where the field "X" is located on the C-grid. Possibles options are, U, V, T and Zeta.
    """

    if u_grid_loc == 'U':
        x_u = grid_object['Xp1'][:]
        y_u = grid_object['Y'][:]
    elif u_grid_loc == 'V':
        x_u = grid_object['X'][:]
        y_u = grid_object['Yp1'][:]  
    elif u_grid_loc == 'T':
        x_u = grid_object['X'][:]
        y_u = grid_object['Y'][:]  
    elif u_grid_loc == 'Zeta':
        x_u = grid_object['Xp1'][:]
        y_u = grid_object['Yp1'][:]
    else:
        print 'u_grid_loc not set correctly. Possible options are: U,V,T, Zeta'
        return

    if v_grid_loc == 'U':
        x_v = grid_object['Xp1'][:]
        y_v = grid_object['Y'][:]
    elif v_grid_loc == 'V':
        x_v = grid_object['X'][:]
        y_v = grid_object['Yp1'][:]  
    elif v_grid_loc == 'T':
        x_v = grid_object['X'][:]
        y_v = grid_object['Y'][:]  
    elif v_grid_loc == 'Zeta':
        x_v = grid_object['Xp1'][:]
        y_v = grid_object['Yp1'][:]
    else:
        print 'v_grid_loc not set correctly. Possible options are: U,V,T, Zeta'
        return





    len_x_u = len(x_u)
    len_y_u = len(y_u)
    
    len_x_v = len(x_v)
    len_y_v = len(y_v)
    
    x_stream = np.ones((int(t_max/delta_t)+2))*startx
    y_stream = np.ones((int(t_max/delta_t)+2))*starty
    t_stream = np.zeros((int(t_max/delta_t)+2))

    t = 0 #set the initial time to be zero
    i=0

    deg_per_m = np.array([1,1])

    # Runge-Kutta fourth order method to estimate next position.
    while t < t_max:
        if grid_object['grid_type']=='polar':
            # use degrees per metre and convert all the velocities to degrees / second# calculate degrees per metre at current location - used to convert the m/s velocities in to degrees/s
            deg_per_m = np.array([1./(1852.*60.),np.cos(starty*np.pi/180.)/(1852.*60.)])

        # Interpolate velocities to initial location
        u_loc = bilinear_interp(startx,starty,u,x_u,y_u,len_x_u,len_y_u)
        v_loc = bilinear_interp(startx,starty,v,x_v,y_v,len_x_v,len_y_v)
        u_loc = u_loc*deg_per_m[1]
        v_loc = v_loc*deg_per_m[0]
        dx1 = delta_t*u_loc
        dy1 = delta_t*v_loc

        u_loc1 = bilinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,u,x_u,y_u,len_x_u,len_y_u)
        v_loc1 = bilinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,v,x_v,y_v,len_x_v,len_y_v)
        u_loc1 = u_loc1*deg_per_m[1]
        v_loc1 = v_loc1*deg_per_m[0]
        dx2 = delta_t*u_loc1
        dy2 = delta_t*v_loc1

        u_loc2 = bilinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,u,x_u,y_u,len_x_u,len_y_u)
        v_loc2 = bilinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,v,x_v,y_v,len_x_v,len_y_v)
        u_loc2 = u_loc2*deg_per_m[1]
        v_loc2 = v_loc2*deg_per_m[0]
        dx3 = delta_t*u_loc2
        dy3 = delta_t*v_loc2

        u_loc3 = bilinear_interp(startx + dx3,starty + dy3,u,x_u,y_u,len_x_u,len_y_u)
        v_loc3 = bilinear_interp(startx + dx3,starty + dy3,v,x_v,y_v,len_x_v,len_y_v)
        u_loc3 = u_loc3*deg_per_m[1]
        v_loc3 = v_loc3*deg_per_m[0]
        dx4 = delta_t*u_loc3
        dy4 = delta_t*v_loc3

        startx = startx + (dx1 + 2*dx2 + 2*dx3 + dx4)/6
        starty = starty + (dy1 + 2*dy2 + 2*dy3 + dy4)/6

        t += delta_t
        i += 1

        x_stream[i] = startx
        y_stream[i] = starty
        t_stream[i] = t

    # if x_stream[0] == x_stream[-1]:
    #     x_stream = np.delete(x_stream,-1) 
    #     y_stream = np.delete(y_stream,-1) 
    #     t_stream = np.delete(t_stream,-1) 

    return x_stream,y_stream,t_stream


def stream3(u,v,w,
            startx,starty,startz,
            grid_object=None,
            x_v='None',y_v='None',z_v='None',
            x_w='None',y_w='None',z_w='None',
            t_max=2592000,delta_t=3600,
            u_grid_loc='U',v_grid_loc='V',w_grid_loc='W'):
    """!A three-dimensional streamline solver. The velocity fields must be three dimensional and not vary in time.
        X_grid_loc variables specify where the field "X" is located on the C-grid. Possibles options are, U, V, W, T and Zeta.
"""
    if grid_object:
        if u_grid_loc == 'U':
            x_u = copy.deepcopy(grid_object['Xp1'][:])
            y_u = copy.deepcopy(grid_object['Y'][:])
            z_u = copy.deepcopy(grid_object['Z'][:])
        elif u_grid_loc == 'V':
            x_u = copy.deepcopy(grid_object['X'][:])
            y_u = copy.deepcopy(grid_object['Yp1'][:]) 
            z_u = copy.deepcopy(grid_object['Z'][:])
        elif u_grid_loc == 'W':
            x_u = copy.deepcopy(grid_object['X'][:])
            y_u = copy.deepcopy(grid_object['Y'][:]) 
            z_u = copy.deepcopy(grid_object['Zl'][:])
        elif u_grid_loc == 'T':
            x_u = copy.deepcopy(grid_object['X'][:])
            y_u = copy.deepcopy(grid_object['Y'][:]) 
            z_u = copy.deepcopy(grid_object['Z'][:])
        elif u_grid_loc == 'Zeta':
            x_u = copy.deepcopy(grid_object['Xp1'][:])
            y_u = copy.deepcopy(grid_object['Yp1'][:])
            z_u = copy.deepcopy(grid_object['Z'][:])
        else:
            print 'u_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
            return

        if v_grid_loc == 'U':
            x_v = copy.deepcopy(grid_object['Xp1'][:])
            y_v = copy.deepcopy(grid_object['Y'][:])
            z_v = copy.deepcopy(grid_object['Z'][:])
        elif v_grid_loc == 'V':
            x_v = copy.deepcopy(grid_object['X'][:])
            y_v = copy.deepcopy(grid_object['Yp1'][:]) 
            z_v = copy.deepcopy(grid_object['Z'][:])
        elif v_grid_loc == 'W':
            x_v = copy.deepcopy(grid_object['X'][:])
            y_v = copy.deepcopy(grid_object['Y'][:]) 
            z_v = copy.deepcopy(grid_object['Zl'][:])
        elif v_grid_loc == 'T':
            x_v = copy.deepcopy(grid_object['X'][:])
            y_v = copy.deepcopy(grid_object['Y'][:]) 
            z_v = copy.deepcopy(grid_object['Z'][:])
        elif v_grid_loc == 'Zeta':
            x_v = copy.deepcopy(grid_object['Xp1'][:])
            y_v = copy.deepcopy(grid_object['Yp1'][:])
            z_v = copy.deepcopy(grid_object['Z'][:])
        else:
            print 'v_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
            return

        if w_grid_loc == 'U':
            x_w = copy.deepcopy(grid_object['Xp1'][:])
            y_w = copy.deepcopy(grid_object['Y'][:])
            z_w = copy.deepcopy(grid_object['Z'][:])
        elif w_grid_loc == 'V':
            x_w = copy.deepcopy(grid_object['X'][:])
            y_w = copy.deepcopy(grid_object['Yp1'][:]) 
            z_w = copy.deepcopy(grid_object['Z'][:])
        elif w_grid_loc == 'W':
            x_w = copy.deepcopy(grid_object['X'][:])
            y_w = copy.deepcopy(grid_object['Y'][:])
            z_w = copy.deepcopy(grid_object['Zl'][:])
        elif w_grid_loc == 'T':
            x_w = copy.deepcopy(grid_object['X'][:])
            y_w = copy.deepcopy(grid_object['Y'][:]) 
            z_w = copy.deepcopy(grid_object['Z'][:])
        elif w_grid_loc == 'Zeta':
            x_w = copy.deepcopy(grid_object['Xp1'][:])
            y_w = copy.deepcopy(grid_object['Yp1'][:])
            z_w = copy.deepcopy(grid_object['Z'][:])
        else:
            print 'w_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
            return

        
    len_x_u = len(x_u)
    len_y_u = len(y_u)
    len_z_u = len(z_u)
    
    len_x_v = len(x_v)
    len_y_v = len(y_v)
    len_z_v = len(z_v)
    
    len_x_w = len(x_w)
    len_y_w = len(y_w)
    len_z_w = len(z_w)

    x_stream = np.ones((int(np.fabs(t_max/delta_t))+2))*startx
    y_stream = np.ones((int(np.fabs(t_max/delta_t))+2))*starty
    z_stream = np.ones((int(np.fabs(t_max/delta_t))+2))*startz
    t_stream = np.zeros((int(np.fabs(t_max/delta_t))+2))

    t = 0 #set the initial time to be zero
    i=0

    deg_per_m = np.array([1,1])

    
    # Runge-Kutta fourth order method to estimate next position.
    while i < np.fabs(t_max/delta_t):
    #t < t_max:
        if grid_object['grid_type']=='polar':
            # use degrees per metre and convert all the velocities to degrees / second# calculate degrees per metre at current location - used to convert the m/s velocities in to degrees/s
            deg_per_m = np.array([1./(1852.*60.),np.cos(starty*np.pi/180.)/(1852.*60.)])

        # Interpolate velocities to initial location
        u_loc = trilinear_interp(startx,starty,startz,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc = trilinear_interp(startx,starty,startz,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc = trilinear_interp(startx,starty,startz,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc = u_loc*deg_per_m[1]
        v_loc = v_loc*deg_per_m[0]
        dx1 = delta_t*u_loc
        dy1 = delta_t*v_loc
        dz1 = delta_t*w_loc

        u_loc1 = trilinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc1 = trilinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc1 = trilinear_interp(startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc1 = u_loc1*deg_per_m[1]
        v_loc1 = v_loc1*deg_per_m[0]
        dx2 = delta_t*u_loc1
        dy2 = delta_t*v_loc1
        dz2 = delta_t*w_loc1

        u_loc2 = trilinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc2 = trilinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc2 = trilinear_interp(startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc2 = u_loc2*deg_per_m[1]
        v_loc2 = v_loc2*deg_per_m[0]
        dx3 = delta_t*u_loc2
        dy3 = delta_t*v_loc2
        dz3 = delta_t*w_loc2

        u_loc3 = trilinear_interp(startx + dx3,starty + dy3,startz + dz3,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc3 = trilinear_interp(startx + dx3,starty + dy3,startz + dz3,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc3 = trilinear_interp(startx + dx3,starty + dy3,startz + dz3,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc3 = u_loc3*deg_per_m[1]
        v_loc3 = v_loc3*deg_per_m[0]
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

###########################################################
def stream3_many(u,v,w,
            startx,starty,startz,
            grid_object=None,
            x_v='None',y_v='None',z_v='None',
            x_w='None',y_w='None',z_w='None',
            t_max=2592000,delta_t=3600,
            u_grid_loc='U',v_grid_loc='V',w_grid_loc='W'):
    """!A three-dimensional streamline solver. The velocity fields must be three dimensional and not vary in time.
        X_grid_loc variables specify where the field "X" is located on the C-grid. Possibles options are, U, V, W, T and Zeta.

        Returns:
        * x_stream, y_stream, z_stream - all with dimensions (particle,time_level)
        * t_stream - with dimensions (time_level)
"""
    if grid_object:
        if u_grid_loc == 'U':
            x_u = copy.deepcopy(grid_object['Xp1'][:])
            y_u = copy.deepcopy(grid_object['Y'][:])
            z_u = copy.deepcopy(grid_object['Z'][:])
        elif u_grid_loc == 'V':
            x_u = copy.deepcopy(grid_object['X'][:])
            y_u = copy.deepcopy(grid_object['Yp1'][:]) 
            z_u = copy.deepcopy(grid_object['Z'][:])
        elif u_grid_loc == 'W':
            x_u = copy.deepcopy(grid_object['X'][:])
            y_u = copy.deepcopy(grid_object['Y'][:]) 
            z_u = copy.deepcopy(grid_object['Zl'][:])
        elif u_grid_loc == 'T':
            x_u = copy.deepcopy(grid_object['X'][:])
            y_u = copy.deepcopy(grid_object['Y'][:]) 
            z_u = copy.deepcopy(grid_object['Z'][:])
        elif u_grid_loc == 'Zeta':
            x_u = copy.deepcopy(grid_object['Xp1'][:])
            y_u = copy.deepcopy(grid_object['Yp1'][:])
            z_u = copy.deepcopy(grid_object['Z'][:])
        else:
            print 'u_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
            return

        if v_grid_loc == 'U':
            x_v = copy.deepcopy(grid_object['Xp1'][:])
            y_v = copy.deepcopy(grid_object['Y'][:])
            z_v = copy.deepcopy(grid_object['Z'][:])
        elif v_grid_loc == 'V':
            x_v = copy.deepcopy(grid_object['X'][:])
            y_v = copy.deepcopy(grid_object['Yp1'][:]) 
            z_v = copy.deepcopy(grid_object['Z'][:])
        elif v_grid_loc == 'W':
            x_v = copy.deepcopy(grid_object['X'][:])
            y_v = copy.deepcopy(grid_object['Y'][:]) 
            z_v = copy.deepcopy(grid_object['Zl'][:])
        elif v_grid_loc == 'T':
            x_v = copy.deepcopy(grid_object['X'][:])
            y_v = copy.deepcopy(grid_object['Y'][:]) 
            z_v = copy.deepcopy(grid_object['Z'][:])
        elif v_grid_loc == 'Zeta':
            x_v = copy.deepcopy(grid_object['Xp1'][:])
            y_v = copy.deepcopy(grid_object['Yp1'][:])
            z_v = copy.deepcopy(grid_object['Z'][:])
        else:
            print 'v_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
            return

        if w_grid_loc == 'U':
            x_w = copy.deepcopy(grid_object['Xp1'][:])
            y_w = copy.deepcopy(grid_object['Y'][:])
            z_w = copy.deepcopy(grid_object['Z'][:])
        elif w_grid_loc == 'V':
            x_w = copy.deepcopy(grid_object['X'][:])
            y_w = copy.deepcopy(grid_object['Yp1'][:]) 
            z_w = copy.deepcopy(grid_object['Z'][:])
        elif w_grid_loc == 'W':
            x_w = copy.deepcopy(grid_object['X'][:])
            y_w = copy.deepcopy(grid_object['Y'][:])
            z_w = copy.deepcopy(grid_object['Zl'][:])
        elif w_grid_loc == 'T':
            x_w = copy.deepcopy(grid_object['X'][:])
            y_w = copy.deepcopy(grid_object['Y'][:]) 
            z_w = copy.deepcopy(grid_object['Z'][:])
        elif w_grid_loc == 'Zeta':
            x_w = copy.deepcopy(grid_object['Xp1'][:])
            y_w = copy.deepcopy(grid_object['Yp1'][:])
            z_w = copy.deepcopy(grid_object['Z'][:])
        else:
            print 'w_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
            return

        
    len_x_u = len(x_u)
    len_y_u = len(y_u)
    len_z_u = len(z_u)
    
    len_x_v = len(x_v)
    len_y_v = len(y_v)
    len_z_v = len(z_v)
    
    len_x_w = len(x_w)
    len_y_w = len(y_w)
    len_z_w = len(z_w)

    x_stream = np.ones((len(startx),int(np.fabs(t_max/delta_t))+2))*startx[:,np.newaxis]
    y_stream = np.ones((len(startx),int(np.fabs(t_max/delta_t))+2))*starty[:,np.newaxis]
    z_stream = np.ones((len(startx),int(np.fabs(t_max/delta_t))+2))*startz[:,np.newaxis]
    t_stream = np.zeros((int(np.fabs(t_max/delta_t))+2))

    t = 0 #set the initial time to be zero
    i=0

    # Prepare for spherical polar grids
    deg_per_m = np.ones((len(startx),2),dtype=float)
    if grid_object['grid_type']=='polar':
        deg_per_m[:,0] = np.ones_like(startx)/(1852.*60.) # multiplier for v
    
    # Runge-Kutta fourth order method to estimate next position.
    while i < np.fabs(t_max/delta_t):
        #t < t_max:
        if grid_object['grid_type']=='polar':
            # use degrees per metre and convert all the velocities to degrees / second# calculate degrees per metre at current location - used to convert the m/s velocities in to degrees/s
            deg_per_m[:,1] = np.cos(starty*np.pi/180.)/(1852.*60.)# multiplier for u

        # Interpolate velocities to initial location
        u_loc = trilinear_interp_arrays(startx,starty,startz,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc = trilinear_interp_arrays(startx,starty,startz,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc = trilinear_interp_arrays(startx,starty,startz,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc = u_loc*deg_per_m[:,1]
        v_loc = v_loc*deg_per_m[:,0]
        dx1 = delta_t*u_loc
        dy1 = delta_t*v_loc
        dz1 = delta_t*w_loc

        u_loc1 = trilinear_interp_arrays(startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc1 = trilinear_interp_arrays(startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc1 = trilinear_interp_arrays(startx + 0.5*dx1,starty + 0.5*dy1,startz + 0.5*dz1,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc1 = u_loc1*deg_per_m[:,1]
        v_loc1 = v_loc1*deg_per_m[:,0]
        dx2 = delta_t*u_loc1
        dy2 = delta_t*v_loc1
        dz2 = delta_t*w_loc1

        u_loc2 = trilinear_interp_arrays(startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc2 = trilinear_interp_arrays(startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc2 = trilinear_interp_arrays(startx + 0.5*dx2,starty + 0.5*dy2,startz + 0.5*dz2,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc2 = u_loc2*deg_per_m[:,1]
        v_loc2 = v_loc2*deg_per_m[:,0]
        dx3 = delta_t*u_loc2
        dy3 = delta_t*v_loc2
        dz3 = delta_t*w_loc2

        u_loc3 = trilinear_interp_arrays(startx + dx3,starty + dy3,startz + dz3,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc3 = trilinear_interp_arrays(startx + dx3,starty + dy3,startz + dz3,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc3 = trilinear_interp_arrays(startx + dx3,starty + dy3,startz + dz3,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc3 = u_loc3*deg_per_m[:,1]
        v_loc3 = v_loc3*deg_per_m[:,0]
        dx4 = delta_t*u_loc3
        dy4 = delta_t*v_loc3
        dz4 = delta_t*w_loc3

        #recycle the "start_" variables to keep the code clean
        startx = startx + (dx1 + 2*dx2 + 2*dx3 + dx4)/6
        starty = starty + (dy1 + 2*dy2 + 2*dy3 + dy4)/6
        startz = startz + (dz1 + 2*dz2 + 2*dz3 + dz4)/6
        t += delta_t
        i += 1

        x_stream[:,i] = startx
        y_stream[:,i] = starty
        z_stream[:,i] = startz
        t_stream[i] = t


    return x_stream,y_stream,z_stream,t_stream
#####################################################


def pathlines(u_netcdf_filename,v_netcdf_filename,w_netcdf_filename,
            startx,starty,startz,startt,
            t,
            grid_object,            
            t_max,delta_t,
            u_netcdf_variable='UVEL',
            v_netcdf_variable='VVEL',
            w_netcdf_variable='WVEL',
            u_grid_loc='U',v_grid_loc='V',w_grid_loc='W',
            u_bias_field=None,
            v_bias_field=None,
            w_bias_field=None):
    """!A three-dimensional lagrangian particle tracker. The velocity fields must be four dimensional (three spatial, one temporal) and have units of m/s.
    It should work to track particles forwards or backwards in time (set delta_t <0 for backwards in time). But, be warned, backwards in time hasn't been thoroughly tested yet.
    
    Because this is a very large amount of data, the fields are passed as netcdffile handles.
    
    The variables are:
    * ?_netcdf_filename = name of the netcdf file with ?'s data in it.
    * start? = intial value for x, y, z, or t.
    * t = vector of time levels that are contained in the velocity data.
    * grid_object is m.grid if you followed the standard naming conventions.
    * ?_netcdf_variable = name of the "?" variable field in the netcdf file.
    * t_max = length of time to track particles for, in seconds. This is always positive
    * delta_t = timestep for particle tracking algorithm, in seconds. This can be positive or negative.
    * ?_grid_loc = where the field "?" is located on the C-grid. Possibles options are, U, V, W, T and Zeta.
    * ?_bias_field = bias to add to that velocity field omponent. If set to -mean(velocity component), then only the time varying portion of that field will be used.
    """

    if u_grid_loc == 'U':
        x_u = grid_object['Xp1'][:]
        y_u = grid_object['Y'][:]
        z_u = grid_object['Z'][:]
    elif u_grid_loc == 'V':
        x_u = grid_object['X'][:]
        y_u = grid_object['Yp1'][:]  
        z_u = grid_object['Z'][:]
    elif u_grid_loc == 'W':
        x_u = grid_object['X'][:]
        y_u = grid_object['Y'][:]  
        z_u = grid_object['Zl'][:]
    elif u_grid_loc == 'T':
        x_u = grid_object['X'][:]
        y_u = grid_object['Y'][:]  
        z_u = grid_object['Z'][:]
    elif u_grid_loc == 'Zeta':
        x_u = grid_object['Xp1'][:]
        y_u = grid_object['Yp1'][:]
        z_u = grid_object['Z'][:]
    else:
        print 'u_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
        return

    if v_grid_loc == 'U':
        x_v = grid_object['Xp1'][:]
        y_v = grid_object['Y'][:]
        z_v = grid_object['Z'][:]
    elif v_grid_loc == 'V':
        x_v = grid_object['X'][:]
        y_v = grid_object['Yp1'][:]  
        z_v = grid_object['Z'][:]
    elif v_grid_loc == 'W':
        x_v = grid_object['X'][:]
        y_v = grid_object['Y'][:]  
        z_v = grid_object['Zl'][:]
    elif v_grid_loc == 'T':
        x_v = grid_object['X'][:]
        y_v = grid_object['Y'][:]  
        z_v = grid_object['Z'][:]
    elif v_grid_loc == 'Zeta':
        x_v = grid_object['Xp1'][:]
        y_v = grid_object['Yp1'][:]
        z_v = grid_object['Z'][:]
    else:
        print 'v_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
        return

    if w_grid_loc == 'U':
        x_w = grid_object['Xp1'][:]
        y_w = grid_object['Y'][:]
        z_w = grid_object['Z'][:]
    elif w_grid_loc == 'V':
        x_w = grid_object['X'][:]
        y_w = grid_object['Yp1'][:]  
        z_w = grid_object['Z'][:]
    elif w_grid_loc == 'W':
        x_w = grid_object['X'][:]
        y_w = grid_object['Y'][:]  
        z_w = grid_object['Zl'][:]
    elif w_grid_loc == 'T':
        x_w = grid_object['X'][:]
        y_w = grid_object['Y'][:]  
        z_w = grid_object['Z'][:]
    elif w_grid_loc == 'Zeta':
        x_w = grid_object['Xp1'][:]
        y_w = grid_object['Yp1'][:]
        z_w = grid_object['Z'][:]
    else:
        print 'w_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
        return

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

    if u_bias_field is None:
        u_bias_field = np.zeros_like(grid_object['wet_mask_U'][:])
    if v_bias_field is None:
        v_bias_field = np.zeros_like(grid_object['wet_mask_V'][:])
    if w_bias_field is None:
        w_bias_field = np.zeros_like(grid_object['wet_mask_W'][:])

    x_stream = np.ones((int(np.fabs(t_max/delta_t))+2))*startx
    y_stream = np.ones((int(np.fabs(t_max/delta_t))+2))*starty
    z_stream = np.ones((int(np.fabs(t_max/delta_t))+2))*startz
    t_stream = np.ones((int(np.fabs(t_max/delta_t))+2))*startt

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
                                                u_netcdf_filehandle,u_netcdf_variable,u_bias_field)
    u_field,x_index_u_new,y_index_u_new,z_index_u_new = indices_and_field(x_u,y_u,z_u,
                                                x_RK,y_RK,z_RK,t_index,
                                                len_x_u,len_y_u,len_z_u,len_t,
                                                u_netcdf_filehandle,u_netcdf_variable,u_bias_field)
    #  v
    v_field,x_index_v,y_index_v,z_index_v = indices_and_field(x_v,y_v,z_v,
                                                x_RK,y_RK,z_RK,t_index,
                                                len_x_v,len_y_v,len_z_v,len_t,
                                                v_netcdf_filehandle,v_netcdf_variable,v_bias_field)
    v_field,x_index_v_new,y_index_v_new,z_index_v_new = indices_and_field(x_v,y_v,z_v,
                                                x_RK,y_RK,z_RK,t_index,
                                                len_x_v,len_y_v,len_z_v,len_t,
                                                v_netcdf_filehandle,v_netcdf_variable,v_bias_field)

    #  w
    w_field,x_index_w,y_index_w,z_index_w = indices_and_field(x_w,y_w,z_w,
                                                x_RK,y_RK,z_RK,t_index,
                                                len_x_w,len_y_w,len_z_w,len_t,
                                                w_netcdf_filehandle,w_netcdf_variable,w_bias_field)
    w_field,x_index_w_new,y_index_w_new,z_index_w_new = indices_and_field(x_w,y_w,z_w,
                                                x_RK,y_RK,z_RK,t_index,
                                                len_x_w,len_y_w,len_z_w,len_t,
                                                w_netcdf_filehandle,w_netcdf_variable,w_bias_field)
    

    # Prepare for spherical polar grids
    deg_per_m = np.array([1,1])

    # Runge-Kutta fourth order method to estimate next position.
    while i < np.fabs(t_max/delta_t):
    #t_RK < t_max + startt:
        
        if grid_object['grid_type']=='polar':
            # use degrees per metre and convert all the velocities to degrees / second# calculate degrees per metre at current location - used to convert the m/s velocities in to degrees/s
            deg_per_m = np.array([1./(1852.*60.),np.cos(starty*np.pi/180.)/(1852.*60.)])
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
                                                        u_netcdf_filehandle,u_netcdf_variable,u_bias_field)
            # for v
            v_field,x_index_v,y_index_v,z_index_v = indices_and_field(x_v,y_v,z_v,
                                                        x_RK,y_RK,z_RK,t_index,
                                                        len_x_v,len_y_v,len_z_v,len_t,
                                                        v_netcdf_filehandle,v_netcdf_variable,v_bias_field)

            # for w
            w_field,x_index_w,y_index_w,z_index_w = indices_and_field(x_w,y_w,z_w,
                                                        x_RK,y_RK,z_RK,t_index,
                                                        len_x_w,len_y_w,len_z_w,len_t,
                                                        w_netcdf_filehandle,w_netcdf_variable,w_bias_field)



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
        u_loc = u_loc*deg_per_m[1]
        v_loc = v_loc*deg_per_m[0]
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
        u_loc1 = u_loc1*deg_per_m[1]
        v_loc1 = v_loc1*deg_per_m[0]
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
        u_loc2 = u_loc2*deg_per_m[1]
        v_loc2 = v_loc2*deg_per_m[0]
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
        u_loc3 = u_loc3*deg_per_m[1]
        v_loc3 = v_loc3*deg_per_m[0]
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


   ######################################################################

def pathlines_many(u_netcdf_filename,v_netcdf_filename,w_netcdf_filename,
            startx,starty,startz,startt,
            t,
            grid_object, 
            t_max,delta_t,           
            u_netcdf_variable='UVEL',
            v_netcdf_variable='VVEL',
            w_netcdf_variable='WVEL',
            u_grid_loc='U',v_grid_loc='V',w_grid_loc='W',
            u_bias_field=None,
            v_bias_field=None,
            w_bias_field=None):
    """!A three-dimensional lagrangian particle tracker designed for tracking many particles at once. If you're tracking fewer than O(10) - use the pathlines function. 

    The velocity fields must be four dimensional (three spatial, one temporal) and have units of m/s.
    It should work to track particles forwards or backwards in time (set delta_t <0 for backwards in time). But, be warned, backwards in time hasn't been thoroughly tested yet.
    
    Because this is a very large amount of data, the fields are passed as netcdffile handles.


    ## Returns:
    * x_stream, y_stream, z_stream - all with dimensions (particle,time_level)
    * t_stream - with dimensions (time_level)
    
    ## The variables are:
    * ?_netcdf_filename = name of the netcdf file with ?'s data in it.
    * start? = (nx1) arrays of initial values for x, y, or z.
    * startt = start time
    * t = vector of time levels that are contained in the velocity data.
    * grid_object is m.grid if you followed the standard naming conventions.
    * ?_netcdf_variable = name of the "?" variable field in the netcdf file.
    * t_max = length of time to track particles for, in seconds. This is always positive
    * delta_t = timestep for particle tracking algorithm, in seconds. This can be positive or negative.
    * ?_grid_loc = where the field "?" is located on the C-grid. Possibles options are, U, V, W, T and Zeta.
    * ?_bias_field = bias to add to that velocity field component. If set to -mean(velocity component), then only the time varying portion of that field will be used.
    """

    if u_grid_loc == 'U':
        x_u = grid_object['Xp1'][:]
        y_u = grid_object['Y'][:]
        z_u = grid_object['Z'][:]
    elif u_grid_loc == 'V':
        x_u = grid_object['X'][:]
        y_u = grid_object['Yp1'][:]  
        z_u = grid_object['Z'][:]
    elif u_grid_loc == 'W':
        x_u = grid_object['X'][:]
        y_u = grid_object['Y'][:]  
        z_u = grid_object['Zl'][:]
    elif u_grid_loc == 'T':
        x_u = grid_object['X'][:]
        y_u = grid_object['Y'][:]  
        z_u = grid_object['Z'][:]
    elif u_grid_loc == 'Zeta':
        x_u = grid_object['Xp1'][:]
        y_u = grid_object['Yp1'][:]
        z_u = grid_object['Z'][:]
    else:
        print 'u_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
        return

    if v_grid_loc == 'U':
        x_v = grid_object['Xp1'][:]
        y_v = grid_object['Y'][:]
        z_v = grid_object['Z'][:]
    elif v_grid_loc == 'V':
        x_v = grid_object['X'][:]
        y_v = grid_object['Yp1'][:]  
        z_v = grid_object['Z'][:]
    elif v_grid_loc == 'W':
        x_v = grid_object['X'][:]
        y_v = grid_object['Y'][:]  
        z_v = grid_object['Zl'][:]
    elif v_grid_loc == 'T':
        x_v = grid_object['X'][:]
        y_v = grid_object['Y'][:]  
        z_v = grid_object['Z'][:]
    elif v_grid_loc == 'Zeta':
        x_v = grid_object['Xp1'][:]
        y_v = grid_object['Yp1'][:]
        z_v = grid_object['Z'][:]
    else:
        print 'v_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
        return

    if w_grid_loc == 'U':
        x_w = grid_object['Xp1'][:]
        y_w = grid_object['Y'][:]
        z_w = grid_object['Z'][:]
    elif w_grid_loc == 'V':
        x_w = grid_object['X'][:]
        y_w = grid_object['Yp1'][:]  
        z_w = grid_object['Z'][:]
    elif w_grid_loc == 'W':
        x_w = grid_object['X'][:]
        y_w = grid_object['Y'][:]  
        z_w = grid_object['Zl'][:]
    elif w_grid_loc == 'T':
        x_w = grid_object['X'][:]
        y_w = grid_object['Y'][:]  
        z_w = grid_object['Z'][:]
    elif w_grid_loc == 'Zeta':
        x_w = grid_object['Xp1'][:]
        y_w = grid_object['Yp1'][:]
        z_w = grid_object['Z'][:]
    else:
        print 'w_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
        return

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

    if u_bias_field is None:
        u_bias_field = np.zeros_like(grid_object['wet_mask_U'][:])
    if v_bias_field is None:
        v_bias_field = np.zeros_like(grid_object['wet_mask_V'][:])
    if w_bias_field is None:
        w_bias_field = np.zeros_like(grid_object['wet_mask_W'][1:,...])

    x_stream = np.ones((len(startx),int(np.fabs(t_max/delta_t))+2))*startx[:,np.newaxis]
    y_stream = np.ones((len(startx),int(np.fabs(t_max/delta_t))+2))*starty[:,np.newaxis]
    z_stream = np.ones((len(startx),int(np.fabs(t_max/delta_t))+2))*startz[:,np.newaxis]
    t_stream = np.ones((int(np.fabs(t_max/delta_t))+2))*startt

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
    

    u_field_before = u_netcdf_filehandle.variables[u_netcdf_variable][t_index,...]
    u_field_after = u_netcdf_filehandle.variables[u_netcdf_variable][t_index+1,...]
    u_field = (u_field_before + ((u_field_before - u_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - 
                u_bias_field)

    v_field_before = v_netcdf_filehandle.variables[v_netcdf_variable][t_index,...]
    v_field_after = v_netcdf_filehandle.variables[v_netcdf_variable][t_index+1,...]
    v_field = (v_field_before + ((v_field_before - v_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - 
                v_bias_field)

    w_field_before = w_netcdf_filehandle.variables[w_netcdf_variable][t_index,...]
    w_field_after = w_netcdf_filehandle.variables[w_netcdf_variable][t_index+1,...]
    w_field = (w_field_before + ((w_field_before - w_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - 
                w_bias_field)
    
    
    # Prepare for spherical polar grids
    deg_per_m = np.ones((len(startx),2),dtype=float)
    if grid_object['grid_type']=='polar':
        deg_per_m[:,0] = np.ones_like(startx)/(1852.*60.) # multiplier for v

    # Runge-Kutta fourth order method to estimate next position.
    while i < np.fabs(t_max/delta_t):
    #t_RK < t_max + startt:
        
        if grid_object['grid_type']=='polar':
            # use degrees per metre and convert all the velocities to degrees / second# calculate degrees per metre at current location - used to convert the m/s velocities in to degrees/s
            deg_per_m[:,1] = np.cos(starty*np.pi/180.)/(1852.*60.)# multiplier for u
            # Compute indices at location given
        
        if (t_index_new == t_index):
            # time hasn't progressed beyond the loaded time slices
            pass
        else: 
            t_index = np.searchsorted(t,t_RK)
            if t_index == 0:
                raise ValueError('Given time value is outside the given velocity fields - too small')
            elif t_index == len_t:
                raise ValueError('Given time value is outside the given velocity fields - too big')


            u_field_before = u_netcdf_filehandle.variables[u_netcdf_variable][t_index,...]
            u_field_after = u_netcdf_filehandle.variables[u_netcdf_variable][t_index+1,...]
            u_field = (u_field_before + ((u_field_before - u_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - u_bias_field)

            v_field_before = v_netcdf_filehandle.variables[v_netcdf_variable][t_index,...]
            v_field_after = v_netcdf_filehandle.variables[v_netcdf_variable][t_index+1,...]
            v_field = (v_field_before + ((v_field_before - v_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - v_bias_field)

            w_field_before = w_netcdf_filehandle.variables[w_netcdf_variable][t_index,...]
            w_field_after = w_netcdf_filehandle.variables[w_netcdf_variable][t_index+1,...]
            w_field = (w_field_before + ((w_field_before - w_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - w_bias_field)

            # Interpolate velocities at initial location        
        u_loc = np.ones_like(startx)        
        v_loc = np.ones_like(startx)
        w_loc = np.ones_like(startx)

        u_loc = trilinear_interp_arrays(x_RK,y_RK,z_RK,u_field,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc = trilinear_interp_arrays(x_RK,y_RK,z_RK,v_field,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc = trilinear_interp_arrays(x_RK,y_RK,z_RK,w_field,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
            
        u_loc = u_loc*deg_per_m[:,1]
        v_loc = v_loc*deg_per_m[:,0]

        dx1 = delta_t*u_loc
        dy1 = delta_t*v_loc
        dz1 = delta_t*w_loc
        
        u_loc1 = np.ones_like(startx)       
        v_loc1 = np.ones_like(startx)   
        w_loc1 = np.ones_like(startx)

        u_loc1 = trilinear_interp_arrays(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,
                    u_field,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc1 = trilinear_interp_arrays(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,
                    v_field,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc1 = trilinear_interp_arrays(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,
                    w_field,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc1 = u_loc1*deg_per_m[:,1]
        v_loc1 = v_loc1*deg_per_m[:,0]
        dx2 = delta_t*u_loc1
        dy2 = delta_t*v_loc1
        dz2 = delta_t*w_loc1
                    
        u_loc2 = np.ones_like(startx)       
        v_loc2 = np.ones_like(startx)   
        w_loc2 = np.ones_like(startx)
        u_loc2 = trilinear_interp_arrays(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,
                    u_field,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc2 = trilinear_interp_arrays(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,
                    v_field,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc2 = trilinear_interp_arrays(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,
                    w_field,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)

        u_loc2 = u_loc2*deg_per_m[:,1]
        v_loc2 = v_loc2*deg_per_m[:,0]
        dx3 = delta_t*u_loc2
        dy3 = delta_t*v_loc2
        dz3 = delta_t*w_loc2

        u_loc3 = np.ones_like(startx)
        v_loc3 = np.ones_like(startx)
        w_loc3 = np.ones_like(startx)
        u_loc3 = trilinear_interp_arrays(x_RK + dx3,y_RK + dy3,z_RK + dz3,
                    u_field,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc3 = trilinear_interp_arrays(x_RK + dx3,y_RK + dy3,z_RK + dz3,
                    v_field,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc3 = trilinear_interp_arrays(x_RK + dx3,y_RK + dy3,z_RK + dz3,
                    w_field,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        
        u_loc3 = u_loc3*deg_per_m[:,1]
        v_loc3 = v_loc3*deg_per_m[:,0]
        dx4 = delta_t*u_loc3
        dy4 = delta_t*v_loc3
        dz4 = delta_t*w_loc3

        #recycle the variables to keep the code clean
        x_RK = x_RK + (dx1 + 2*dx2 + 2*dx3 + dx4)/6
        y_RK = y_RK + (dy1 + 2*dy2 + 2*dy3 + dy4)/6
        z_RK = z_RK + (dz1 + 2*dz2 + 2*dz3 + dz4)/6
        t_RK += delta_t
        i += 1

        x_stream[:,i] = x_RK
        y_stream[:,i] = y_RK
        z_stream[:,i] = z_RK
        t_stream[i] = t_RK
        
        t_index_new = np.searchsorted(t,t_RK)

  
    u_netcdf_filehandle.close()
    v_netcdf_filehandle.close()
    w_netcdf_filehandle.close()

    return x_stream,y_stream,z_stream,t_stream


    #################################################################################


def pathlines_for_OLIC_xyzt_ani(u_netcdf_filename,v_netcdf_filename,w_netcdf_filename,
            n_particles,startt,
            t,
            grid_object, 
            t_tracking,delta_t, trace_length,
            u_netcdf_variable='UVEL',
            v_netcdf_variable='VVEL',
            w_netcdf_variable='WVEL',
            u_grid_loc='U',v_grid_loc='V',w_grid_loc='W',
            u_bias_field=None,
            v_bias_field=None,
            w_bias_field=None):
    """!A three-dimensional lagrangian particle tracker designed for tracking many particles at once. If you're tracking fewer than O(10) - use the streaklines function. 

    The velocity fields must be four dimensional (three spatial, one temporal) and have units of m/s.
    It should work to track particles forwards or backwards in time (set delta_t <0 for backwards in time). But, be warned, backwards in time hasn't been thoroughly tested yet.
    
    Because this is a very large amount of data, the fields are passed as netcdffile handles.


    ## Returns:
    * x_stream, y_stream, z_stream - all with dimensions (particle,time_level)
    * t_stream - with dimensions (time_level)
    
    ## The variables are:
    * ?_netcdf_filename = name of the netcdf file with ?'s data in it.
    * n_particles = number of particles to track
    * startt = start time
    * t = vector of time levels that are contained in the velocity data.
    * grid_object is m.grid if you followed the standard naming conventions.
    * ?_netcdf_variable = name of the "?" variable field in the netcdf file.
    * t_tracking = length of time to track particles for, in seconds. This is always positive
    * delta_t = timestep for particle tracking algorithm, in seconds. This can be positive or negative.
    * trace_length = length of time for each individual trace
    * ?_grid_loc = where the field "?" is located on the C-grid. Possibles options are, U, V, W, T and Zeta.
    * ?_bias_field = bias to add to that velocity field component. If set to -mean(velocity component), then only the time varying portion of that field will be used.
    """

    if u_grid_loc == 'U':
        x_u = grid_object['Xp1'][:]
        y_u = grid_object['Y'][:]
        z_u = grid_object['Z'][:]
    elif u_grid_loc == 'V':
        x_u = grid_object['X'][:]
        y_u = grid_object['Yp1'][:]  
        z_u = grid_object['Z'][:]
    elif u_grid_loc == 'W':
        x_u = grid_object['X'][:]
        y_u = grid_object['Y'][:]  
        z_u = grid_object['Zl'][:]
    elif u_grid_loc == 'T':
        x_u = grid_object['X'][:]
        y_u = grid_object['Y'][:]  
        z_u = grid_object['Z'][:]
    elif u_grid_loc == 'Zeta':
        x_u = grid_object['Xp1'][:]
        y_u = grid_object['Yp1'][:]
        z_u = grid_object['Z'][:]
    else:
        print 'u_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
        return

    if v_grid_loc == 'U':
        x_v = grid_object['Xp1'][:]
        y_v = grid_object['Y'][:]
        z_v = grid_object['Z'][:]
    elif v_grid_loc == 'V':
        x_v = grid_object['X'][:]
        y_v = grid_object['Yp1'][:]  
        z_v = grid_object['Z'][:]
    elif v_grid_loc == 'W':
        x_v = grid_object['X'][:]
        y_v = grid_object['Y'][:]  
        z_v = grid_object['Zl'][:]
    elif v_grid_loc == 'T':
        x_v = grid_object['X'][:]
        y_v = grid_object['Y'][:]  
        z_v = grid_object['Z'][:]
    elif v_grid_loc == 'Zeta':
        x_v = grid_object['Xp1'][:]
        y_v = grid_object['Yp1'][:]
        z_v = grid_object['Z'][:]
    else:
        print 'v_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
        return

    if w_grid_loc == 'U':
        x_w = grid_object['Xp1'][:]
        y_w = grid_object['Y'][:]
        z_w = grid_object['Z'][:]
    elif w_grid_loc == 'V':
        x_w = grid_object['X'][:]
        y_w = grid_object['Yp1'][:]  
        z_w = grid_object['Z'][:]
    elif w_grid_loc == 'W':
        x_w = grid_object['X'][:]
        y_w = grid_object['Y'][:]  
        z_w = grid_object['Zl'][:]
    elif w_grid_loc == 'T':
        x_w = grid_object['X'][:]
        y_w = grid_object['Y'][:]  
        z_w = grid_object['Z'][:]
    elif w_grid_loc == 'Zeta':
        x_w = grid_object['Xp1'][:]
        y_w = grid_object['Yp1'][:]
        z_w = grid_object['Z'][:]
    else:
        print 'w_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
        return

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

    if u_bias_field is None:
        u_bias_field = np.zeros_like(grid_object['wet_mask_U'][:])
    if v_bias_field is None:
        v_bias_field = np.zeros_like(grid_object['wet_mask_V'][:])
    if w_bias_field is None:
        w_bias_field = np.zeros_like(grid_object['wet_mask_W'][1:,...])


    steps_per_trace = int(trace_length/delta_t)
    time_steps_until_jitter = np.random.randint(steps_per_trace, size=n_particles)

    startx = (np.random.rand(n_particles)*
        (np.max(grid_object['X'][:]) - np.min(grid_object['X'][:]))) + np.min(grid_object['X'][:])
    starty = (np.random.rand(n_particles)*
        (np.max(grid_object['Y'][:]) - np.min(grid_object['Y'][:]))) + np.min(grid_object['Y'][:])
    startz = (np.random.rand(n_particles)*grid_object['Z'][1])
        #(np.max(grid_object['Z'][:]) - np.min(grid_object['X'][:]))) + np.min(grid_object['X'][:])

    x_stream = np.ones((len(startx),int(np.fabs(t_tracking/delta_t))+2))*startx[:,np.newaxis]
    y_stream = np.ones((len(startx),int(np.fabs(t_tracking/delta_t))+2))*starty[:,np.newaxis]
    z_stream = np.ones((len(startx),int(np.fabs(t_tracking/delta_t))+2))*startz[:,np.newaxis]
    t_stream = np.ones((int(np.fabs(t_tracking/delta_t))+2))*startt

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
    

    u_field_before = u_netcdf_filehandle.variables[u_netcdf_variable][t_index,...]
    u_field_after = u_netcdf_filehandle.variables[u_netcdf_variable][t_index+1,...]
    u_field = (u_field_before + ((u_field_before - u_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - 
                u_bias_field)

    v_field_before = v_netcdf_filehandle.variables[v_netcdf_variable][t_index,...]
    v_field_after = v_netcdf_filehandle.variables[v_netcdf_variable][t_index+1,...]
    v_field = (v_field_before + ((v_field_before - v_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - 
                v_bias_field)

    w_field_before = w_netcdf_filehandle.variables[w_netcdf_variable][t_index,...]
    w_field_after = w_netcdf_filehandle.variables[w_netcdf_variable][t_index+1,...]
    w_field = (w_field_before + ((w_field_before - w_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - 
                w_bias_field)
    
    
    # Prepare for spherical polar grids
    deg_per_m = np.ones((len(startx),2),dtype=float)
    if grid_object['grid_type']=='polar':
        deg_per_m[:,0] = np.ones_like(startx)/(1852.*60.) # multiplier for v

    # Runge-Kutta fourth order method to estimate next position.
    while i < np.fabs(t_tracking/delta_t):
    #t_RK < t_tracking + startt:
        
        if grid_object['grid_type']=='polar':
            # use degrees per metre and convert all the velocities to degrees / second# calculate degrees per metre at current location - used to convert the m/s velocities in to degrees/s
            deg_per_m[:,1] = np.cos(starty*np.pi/180.)/(1852.*60.)# multiplier for u
            # Compute indices at location given
        
        if (t_index_new == t_index):
            # time hasn't progressed beyond the loaded time slices
            pass
        else: 
            t_index = np.searchsorted(t,t_RK)
            if t_index == 0:
                raise ValueError('Given time value is outside the given velocity fields - too small')
            elif t_index == len_t:
                raise ValueError('Given time value is outside the given velocity fields - too big')


            u_field_before = u_netcdf_filehandle.variables[u_netcdf_variable][t_index,...]
            u_field_after = u_netcdf_filehandle.variables[u_netcdf_variable][t_index+1,...]
            u_field = (u_field_before + ((u_field_before - u_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - u_bias_field)

            v_field_before = v_netcdf_filehandle.variables[v_netcdf_variable][t_index,...]
            v_field_after = v_netcdf_filehandle.variables[v_netcdf_variable][t_index+1,...]
            v_field = (v_field_before + ((v_field_before - v_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - v_bias_field)

            w_field_before = w_netcdf_filehandle.variables[w_netcdf_variable][t_index,...]
            w_field_after = w_netcdf_filehandle.variables[w_netcdf_variable][t_index+1,...]
            w_field = (w_field_before + ((w_field_before - w_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - w_bias_field)

        # check if any particles need moving, and then move them
        for particle in xrange(0,n_particles):
            if time_steps_until_jitter[particle] <= 0:
                    x_RK[particle] = (np.random.rand(1)*
                        (np.max(grid_object['X'][:]) - np.min(grid_object['X'][:]))) + np.min(grid_object['X'][:])
                    y_RK[particle] = (np.random.rand(1)*
                        (np.max(grid_object['Y'][:]) - np.min(grid_object['Y'][:]))) + np.min(grid_object['Y'][:])
                    z_RK[particle] = grid_object['Z'][1]
                        #(np.random.rand(1)*
                        #(np.max(grid_object['Z'][:]) - np.min(grid_object['Z'][:]))) + np.min(grid_object['Z'][:])
                    time_steps_until_jitter[particle] = steps_per_trace


        # Interpolate velocities at initial location 
        u_loc = np.ones_like(startx)        
        v_loc = np.ones_like(startx)
        w_loc = np.ones_like(startx)

        u_loc = trilinear_interp_arrays(x_RK,y_RK,z_RK,u_field,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc = trilinear_interp_arrays(x_RK,y_RK,z_RK,v_field,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc = trilinear_interp_arrays(x_RK,y_RK,z_RK,w_field,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
            
        u_loc = u_loc*deg_per_m[:,1]
        v_loc = v_loc*deg_per_m[:,0]

        dx1 = delta_t*u_loc
        dy1 = delta_t*v_loc
        dz1 = delta_t*w_loc
        
        u_loc1 = np.ones_like(startx)       
        v_loc1 = np.ones_like(startx)   
        w_loc1 = np.ones_like(startx)

        u_loc1 = trilinear_interp_arrays(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,
                    u_field,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc1 = trilinear_interp_arrays(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,
                    v_field,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc1 = trilinear_interp_arrays(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,
                    w_field,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc1 = u_loc1*deg_per_m[:,1]
        v_loc1 = v_loc1*deg_per_m[:,0]
        dx2 = delta_t*u_loc1
        dy2 = delta_t*v_loc1
        dz2 = delta_t*w_loc1
                    
        u_loc2 = np.ones_like(startx)       
        v_loc2 = np.ones_like(startx)   
        w_loc2 = np.ones_like(startx)
        u_loc2 = trilinear_interp_arrays(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,
                    u_field,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc2 = trilinear_interp_arrays(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,
                    v_field,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc2 = trilinear_interp_arrays(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,
                    w_field,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)

        u_loc2 = u_loc2*deg_per_m[:,1]
        v_loc2 = v_loc2*deg_per_m[:,0]
        dx3 = delta_t*u_loc2
        dy3 = delta_t*v_loc2
        dz3 = delta_t*w_loc2

        u_loc3 = np.ones_like(startx)
        v_loc3 = np.ones_like(startx)
        w_loc3 = np.ones_like(startx)
        u_loc3 = trilinear_interp_arrays(x_RK + dx3,y_RK + dy3,z_RK + dz3,
                    u_field,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc3 = trilinear_interp_arrays(x_RK + dx3,y_RK + dy3,z_RK + dz3,
                    v_field,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc3 = trilinear_interp_arrays(x_RK + dx3,y_RK + dy3,z_RK + dz3,
                    w_field,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        
        u_loc3 = u_loc3*deg_per_m[:,1]
        v_loc3 = v_loc3*deg_per_m[:,0]
        dx4 = delta_t*u_loc3
        dy4 = delta_t*v_loc3
        dz4 = delta_t*w_loc3

        #recycle the variables to keep the code clean
        x_RK = x_RK + (dx1 + 2*dx2 + 2*dx3 + dx4)/6
        y_RK = y_RK + (dy1 + 2*dy2 + 2*dy3 + dy4)/6
        z_RK = z_RK + (dz1 + 2*dz2 + 2*dz3 + dz4)/6
        t_RK += delta_t
        i += 1
        time_steps_until_jitter = time_steps_until_jitter - 1

        x_stream[:,i] = x_RK
        y_stream[:,i] = y_RK
        z_stream[:,i] = z_RK
        t_stream[i] = t_RK
        
        t_index_new = np.searchsorted(t,t_RK)

  
    u_netcdf_filehandle.close()
    v_netcdf_filehandle.close()
    w_netcdf_filehandle.close()

    return x_stream,y_stream,z_stream,t_stream

##########################

# include xyz version

def numeric_GLM_xyzt(u_netcdf_filename,v_netcdf_filename,w_netcdf_filename,
            n_particles,
            startx,starty,startz,startt,
            total_time,timestep,
            r_x,r_y,r_z,
            t,
            grid_object, 
            r_cutoff_factor=3,
            u_netcdf_variable='UVEL',
            v_netcdf_variable='VVEL',
            w_netcdf_variable='WVEL',
            u_grid_loc='U',v_grid_loc='V',w_grid_loc='W',
            u_bias_field=None,
            v_bias_field=None,
            w_bias_field=None):
    """!A three-dimensional lagrangian particle tracker designed for tracking many particles at once as a method of estimating Generalised Lagrangian-Mean velocities. 

    The algorithm is a little slow because it spends time checking that the particles are within a given distance of the centre of mass, that they are in the fluid not the bathymetry and that they are below the surface. This means that the cloud of particles should remain within the fluid and not get trapped by the bathymetry.

    The velocity fields must be four dimensional (three spatial, one temporal) and have units of m/s.
    It should work to track particles forwards or backwards in time (set timestep <0 for backwards in time). But, be warned, backwards in time hasn't been thoroughly tested yet.
    
    Because this is a very large amount of data, the fields are passed as netcdffile handles.


    ## Returns:
    * x_stream, y_stream, z_stream - all with dimensions (particle,time_level)
    * t_stream - with dimensions (time_level)
    * com_stream - with dimensions (time_level)
    
    ## The variables are:
    * ?_netcdf_filename = name of the netcdf file with ?'s data in it.
    * n_particles = number of particles to track
    * start? = starting value for   x, y, z, or time
    * total_time = length of time to track particles for, in seconds. This is always positive
    * timestep = timestep for particle tracking algorithm, in seconds. This can be positive or negative.
    * r_? = radius of sedding ellipsoid in x, y, or  z direction
    * t = vector of time levels that are contained in the velocity data.
    * grid_object is m.grid if you followed the standard naming conventions.
    * ?_netcdf_variable = name of the "?" variable field in the netcdf file.
    * ?_grid_loc = where the field "?" is located on the C-grid. Possibles options are, U, V, W, T and Zeta.
    * ?_bias_field = bias to add to that velocity field component. If set to -mean(velocity component), then only the time varying portion of that field will be used.
    """

    if u_grid_loc == 'U':
        x_u = grid_object['Xp1'][:]
        y_u = grid_object['Y'][:]
        z_u = grid_object['Z'][:]
    elif u_grid_loc == 'V':
        x_u = grid_object['X'][:]
        y_u = grid_object['Yp1'][:]  
        z_u = grid_object['Z'][:]
    elif u_grid_loc == 'W':
        x_u = grid_object['X'][:]
        y_u = grid_object['Y'][:]  
        z_u = grid_object['Zl'][:]
    elif u_grid_loc == 'T':
        x_u = grid_object['X'][:]
        y_u = grid_object['Y'][:]  
        z_u = grid_object['Z'][:]
    elif u_grid_loc == 'Zeta':
        x_u = grid_object['Xp1'][:]
        y_u = grid_object['Yp1'][:]
        z_u = grid_object['Z'][:]
    else:
        print 'u_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
        return

    if v_grid_loc == 'U':
        x_v = grid_object['Xp1'][:]
        y_v = grid_object['Y'][:]
        z_v = grid_object['Z'][:]
    elif v_grid_loc == 'V':
        x_v = grid_object['X'][:]
        y_v = grid_object['Yp1'][:]  
        z_v = grid_object['Z'][:]
    elif v_grid_loc == 'W':
        x_v = grid_object['X'][:]
        y_v = grid_object['Y'][:]  
        z_v = grid_object['Zl'][:]
    elif v_grid_loc == 'T':
        x_v = grid_object['X'][:]
        y_v = grid_object['Y'][:]  
        z_v = grid_object['Z'][:]
    elif v_grid_loc == 'Zeta':
        x_v = grid_object['Xp1'][:]
        y_v = grid_object['Yp1'][:]
        z_v = grid_object['Z'][:]
    else:
        print 'v_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
        return

    if w_grid_loc == 'U':
        x_w = grid_object['Xp1'][:]
        y_w = grid_object['Y'][:]
        z_w = grid_object['Z'][:]
    elif w_grid_loc == 'V':
        x_w = grid_object['X'][:]
        y_w = grid_object['Yp1'][:]  
        z_w = grid_object['Z'][:]
    elif w_grid_loc == 'W':
        x_w = grid_object['X'][:]
        y_w = grid_object['Y'][:]  
        z_w = grid_object['Zl'][:]
    elif w_grid_loc == 'T':
        x_w = grid_object['X'][:]
        y_w = grid_object['Y'][:]  
        z_w = grid_object['Z'][:]
    elif w_grid_loc == 'Zeta':
        x_w = grid_object['Xp1'][:]
        y_w = grid_object['Yp1'][:]
        z_w = grid_object['Z'][:]
    else:
        print 'w_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
        return

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

    if u_bias_field is None:
        u_bias_field = np.zeros_like(grid_object['wet_mask_U'][:])
    if v_bias_field is None:
        v_bias_field = np.zeros_like(grid_object['wet_mask_V'][:])
    if w_bias_field is None:
        w_bias_field = np.zeros_like(grid_object['wet_mask_W'][1:,...])


    ini_x = ((np.random.rand(n_particles)-0.5)*2*r_x + startx)
    ini_y = ((np.random.rand(n_particles)-0.5)*2*r_y + starty)
    ini_z = ((np.random.rand(n_particles)-0.5)*2*r_z + startz)

    x_stream = np.ones((n_particles,int(np.fabs(total_time/timestep))+2))*startx
    y_stream = np.ones((n_particles,int(np.fabs(total_time/timestep))+2))*starty
    z_stream = np.ones((n_particles,int(np.fabs(total_time/timestep))+2))*startz
    t_stream = np.ones((int(np.fabs(total_time/timestep))+2))*startt
    com_stream = np.ones((3,int(np.fabs(total_time/timestep))+2))

    t_RK = startt #set the initial time to be the given start time
    z_RK = ini_z
    y_RK = ini_y
    x_RK = ini_x

    # define centre of mass variable
    x_com = startx
    y_com = starty
    z_com = startz

    normed_radius = ( ((ini_x - startx)**2)/(r_x**2) +  
                      ((ini_y - starty)**2)/(r_y**2) + 
                      ((ini_z - startz)**2)/(r_z**2) )

    wet_test = trilinear_interp_arrays(x_RK,y_RK,z_RK,grid_object['wet_mask_U'][:],x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)

    for particle in xrange(n_particles): # check if any particles outside cutoffs
        if (normed_radius[particle] > r_cutoff_factor or wet_test[particle] < 0.5 or z_RK[particle]*grid_object['Z'][2] < 0): # find specific particles
            while (normed_radius[particle] > 1 or wet_test[particle] < 0.5 or z_RK[particle]*grid_object['Z'][2] < 0): # reseed inside ellipsoid
                x_RK[particle] = ((np.random.rand(1)-0.5)*r_x*2 + x_com)
                y_RK[particle] = ((np.random.rand(1)-0.5)*r_y*2 + y_com)
                z_RK[particle] = ((np.random.rand(1)-0.5)*r_z*2 + z_com)
                normed_radius[particle] = ( ((x_RK[particle] - x_com)**2)/(r_x**2) +  
                                  ((y_RK[particle] - y_com)**2)/(r_y**2) + 
                                  ((z_RK[particle] - z_com)**2)/(r_z**2) )
                wet_test[particle] = trilinear_interp(x_RK[particle],y_RK[particle],z_RK[particle],
                    grid_object['wet_mask_U'][:],x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)




    
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
    

    u_field_before = u_netcdf_filehandle.variables[u_netcdf_variable][t_index,...]
    u_field_after = u_netcdf_filehandle.variables[u_netcdf_variable][t_index+1,...]
    u_field = (u_field_before + ((u_field_before - u_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - 
                u_bias_field)

    v_field_before = v_netcdf_filehandle.variables[v_netcdf_variable][t_index,...]
    v_field_after = v_netcdf_filehandle.variables[v_netcdf_variable][t_index+1,...]
    v_field = (v_field_before + ((v_field_before - v_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - 
                v_bias_field)

    w_field_before = w_netcdf_filehandle.variables[w_netcdf_variable][t_index,...]
    w_field_after = w_netcdf_filehandle.variables[w_netcdf_variable][t_index+1,...]
    w_field = (w_field_before + ((w_field_before - w_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - 
                w_bias_field)
    
    
    # Prepare for spherical polar grids
    deg_per_m = np.ones((n_particles,2),dtype=float)
    if grid_object['grid_type']=='polar':
        deg_per_m[:,0] = np.ones((n_particles))/(1852.*60.) # multiplier for v

    # Runge-Kutta fourth order method to estimate next position.
    while i < np.fabs(total_time/timestep):
    #t_RK < total_time + startt:
        
        if grid_object['grid_type']=='polar':
            # use degrees per metre and convert all the velocities to degrees / second# calculate degrees per metre at current location - used to convert the m/s velocities in to degrees/s
            deg_per_m[:,1] = np.cos(np.ones((n_particles))*starty*np.pi/180.)/(1852.*60.)# multiplier for u
            # Compute indices at location given
        
        if (t_index_new == t_index):
            # time hasn't progressed beyond the loaded time slices
            pass
        else: 
            t_index = np.searchsorted(t,t_RK)
            if t_index == 0:
                raise ValueError('Given time value is outside the given velocity fields - too small')
            elif t_index == len_t:
                raise ValueError('Given time value is outside the given velocity fields - too big')


            u_field_before = u_netcdf_filehandle.variables[u_netcdf_variable][t_index,...]
            u_field_after = u_netcdf_filehandle.variables[u_netcdf_variable][t_index+1,...]
            u_field = (u_field_before + ((u_field_before - u_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - u_bias_field)

            v_field_before = v_netcdf_filehandle.variables[v_netcdf_variable][t_index,...]
            v_field_after = v_netcdf_filehandle.variables[v_netcdf_variable][t_index+1,...]
            v_field = (v_field_before + ((v_field_before - v_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - v_bias_field)

            w_field_before = w_netcdf_filehandle.variables[w_netcdf_variable][t_index,...]
            w_field_after = w_netcdf_filehandle.variables[w_netcdf_variable][t_index+1,...]
            w_field = (w_field_before + ((w_field_before - w_field_after)*
                                (t_RK - t[t_index])/(t[t_index+1] - t[t_index])) - w_bias_field)


        # check if any particles need moving, and then reseed them inside an ellipsoid around centre of mass with the original size
        for particle in xrange(n_particles): # check if any particles outside cutoff
            if (normed_radius[particle] > r_cutoff_factor or wet_test[particle] < 0.5 or z_RK[particle]*grid_object['Z'][2] < 0): # find specific particles
                while (normed_radius[particle] > 1 or wet_test[particle] < 0.5 or z_RK[particle]*grid_object['Z'][2] < 0): # reseed inside ellipsoid
                    x_RK[particle] = ((np.random.rand(1)-0.5)*r_x*2 + x_com)
                    y_RK[particle] = ((np.random.rand(1)-0.5)*r_y*2 + y_com)
                    z_RK[particle] = ((np.random.rand(1)-0.5)*r_z*2 + z_com)
                    normed_radius[particle] = ( ((x_RK[particle] - x_com)**2)/(r_x**2) +  
                                      ((y_RK[particle] - y_com)**2)/(r_y**2) + 
                                      ((z_RK[particle] - z_com)**2)/(r_z**2) )
                    wet_test[particle] = trilinear_interp(x_RK[particle],y_RK[particle],z_RK[particle],
                        grid_object['wet_mask_U'][:],x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)


        # Interpolate velocities at initial location 

        u_loc = trilinear_interp_arrays(x_RK,y_RK,z_RK,u_field,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc = trilinear_interp_arrays(x_RK,y_RK,z_RK,v_field,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc = trilinear_interp_arrays(x_RK,y_RK,z_RK,w_field,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc = u_loc*deg_per_m[:,1]
        v_loc = v_loc*deg_per_m[:,0]
        dx1 = timestep*u_loc
        dy1 = timestep*v_loc
        dz1 = timestep*w_loc
        

        u_loc1 = trilinear_interp_arrays(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,
                    u_field,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc1 = trilinear_interp_arrays(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,
                    v_field,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc1 = trilinear_interp_arrays(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,
                    w_field,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc1 = u_loc1*deg_per_m[:,1]
        v_loc1 = v_loc1*deg_per_m[:,0]
        dx2 = timestep*u_loc1
        dy2 = timestep*v_loc1
        dz2 = timestep*w_loc1
                    

        u_loc2 = trilinear_interp_arrays(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,
                    u_field,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc2 = trilinear_interp_arrays(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,
                    v_field,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc2 = trilinear_interp_arrays(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,
                    w_field,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc2 = u_loc2*deg_per_m[:,1]
        v_loc2 = v_loc2*deg_per_m[:,0]
        dx3 = timestep*u_loc2
        dy3 = timestep*v_loc2
        dz3 = timestep*w_loc2


        u_loc3 = trilinear_interp_arrays(x_RK + dx3,y_RK + dy3,z_RK + dz3,
                    u_field,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc3 = trilinear_interp_arrays(x_RK + dx3,y_RK + dy3,z_RK + dz3,
                    v_field,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc3 = trilinear_interp_arrays(x_RK + dx3,y_RK + dy3,z_RK + dz3,
                    w_field,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        
        u_loc3 = u_loc3*deg_per_m[:,1]
        v_loc3 = v_loc3*deg_per_m[:,0]
        dx4 = timestep*u_loc3
        dy4 = timestep*v_loc3
        dz4 = timestep*w_loc3

        #recycle the variables to keep the code clean
        x_RK = x_RK + (dx1 + 2*dx2 + 2*dx3 + dx4)/6
        y_RK = y_RK + (dy1 + 2*dy2 + 2*dy3 + dy4)/6
        z_RK = z_RK + (dz1 + 2*dz2 + 2*dz3 + dz4)/6
        t_RK += timestep
        i += 1

        x_com = np.mean(x_RK)
        y_com = np.mean(y_RK)
        z_com = np.mean(z_RK)

        x_stream[:,i] = x_RK
        y_stream[:,i] = y_RK
        z_stream[:,i] = z_RK
        t_stream[i] = t_RK

        com_stream[0,i] = x_com
        com_stream[1,i] = y_com
        com_stream[2,i] = z_com

        t_index_new = np.searchsorted(t,t_RK)
        normed_radius = ( ((x_RK - x_com)**2)/(r_x**2) +  
                          ((y_RK - y_com)**2)/(r_y**2) + 
                          ((z_RK - z_com)**2)/(r_z**2) )
        wet_test = trilinear_interp_arrays(x_RK,y_RK,z_RK,grid_object['wet_mask_U'][:],x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
  
    u_netcdf_filehandle.close()
    v_netcdf_filehandle.close()
    w_netcdf_filehandle.close()

    return x_stream,y_stream,z_stream,t_stream, com_stream
################################################

def numeric_GLM_xyz(u,v,w,
            n_particles,
            startx,starty,startz,startt,
            total_time,timestep,
            r_x,r_y,r_z,
            grid_object,
            r_cutoff_factor=3,
            x_v='None',y_v='None',z_v='None',
            x_w='None',y_w='None',z_w='None',
            u_grid_loc='U',v_grid_loc='V',w_grid_loc='W'):


    """!A three-dimensional streamline solver. The velocity fields must be three dimensional and not vary in time.
        X_grid_loc variables specify where the field "X" is located on the C-grid. Possibles options are, U, V, W, T and Zeta.

        Returns:
        * x_stream, y_stream, z_stream - all with dimensions (particle,time_level)
        * t_stream - with dimensions (time_level)
        * com_stream - with dimensions (3(x,y,z),time_level)
"""
    if grid_object:
        if u_grid_loc == 'U':
            x_u = copy.deepcopy(grid_object['Xp1'][:])
            y_u = copy.deepcopy(grid_object['Y'][:])
            z_u = copy.deepcopy(grid_object['Z'][:])
        elif u_grid_loc == 'V':
            x_u = copy.deepcopy(grid_object['X'][:])
            y_u = copy.deepcopy(grid_object['Yp1'][:]) 
            z_u = copy.deepcopy(grid_object['Z'][:])
        elif u_grid_loc == 'W':
            x_u = copy.deepcopy(grid_object['X'][:])
            y_u = copy.deepcopy(grid_object['Y'][:]) 
            z_u = copy.deepcopy(grid_object['Zl'][:])
        elif u_grid_loc == 'T':
            x_u = copy.deepcopy(grid_object['X'][:])
            y_u = copy.deepcopy(grid_object['Y'][:]) 
            z_u = copy.deepcopy(grid_object['Z'][:])
        elif u_grid_loc == 'Zeta':
            x_u = copy.deepcopy(grid_object['Xp1'][:])
            y_u = copy.deepcopy(grid_object['Yp1'][:])
            z_u = copy.deepcopy(grid_object['Z'][:])
        else:
            print 'u_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
            return

        if v_grid_loc == 'U':
            x_v = copy.deepcopy(grid_object['Xp1'][:])
            y_v = copy.deepcopy(grid_object['Y'][:])
            z_v = copy.deepcopy(grid_object['Z'][:])
        elif v_grid_loc == 'V':
            x_v = copy.deepcopy(grid_object['X'][:])
            y_v = copy.deepcopy(grid_object['Yp1'][:]) 
            z_v = copy.deepcopy(grid_object['Z'][:])
        elif v_grid_loc == 'W':
            x_v = copy.deepcopy(grid_object['X'][:])
            y_v = copy.deepcopy(grid_object['Y'][:]) 
            z_v = copy.deepcopy(grid_object['Zl'][:])
        elif v_grid_loc == 'T':
            x_v = copy.deepcopy(grid_object['X'][:])
            y_v = copy.deepcopy(grid_object['Y'][:]) 
            z_v = copy.deepcopy(grid_object['Z'][:])
        elif v_grid_loc == 'Zeta':
            x_v = copy.deepcopy(grid_object['Xp1'][:])
            y_v = copy.deepcopy(grid_object['Yp1'][:])
            z_v = copy.deepcopy(grid_object['Z'][:])
        else:
            print 'v_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
            return

        if w_grid_loc == 'U':
            x_w = copy.deepcopy(grid_object['Xp1'][:])
            y_w = copy.deepcopy(grid_object['Y'][:])
            z_w = copy.deepcopy(grid_object['Z'][:])
        elif w_grid_loc == 'V':
            x_w = copy.deepcopy(grid_object['X'][:])
            y_w = copy.deepcopy(grid_object['Yp1'][:]) 
            z_w = copy.deepcopy(grid_object['Z'][:])
        elif w_grid_loc == 'W':
            x_w = copy.deepcopy(grid_object['X'][:])
            y_w = copy.deepcopy(grid_object['Y'][:])
            z_w = copy.deepcopy(grid_object['Zl'][:])
        elif w_grid_loc == 'T':
            x_w = copy.deepcopy(grid_object['X'][:])
            y_w = copy.deepcopy(grid_object['Y'][:]) 
            z_w = copy.deepcopy(grid_object['Z'][:])
        elif w_grid_loc == 'Zeta':
            x_w = copy.deepcopy(grid_object['Xp1'][:])
            y_w = copy.deepcopy(grid_object['Yp1'][:])
            z_w = copy.deepcopy(grid_object['Z'][:])
        else:
            print 'w_grid_loc not set correctly. Possible options are: U,V,W,T, and Zeta'
            return

        
    len_x_u = len(x_u)
    len_y_u = len(y_u)
    len_z_u = len(z_u)
    
    len_x_v = len(x_v)
    len_y_v = len(y_v)
    len_z_v = len(z_v)
    
    len_x_w = len(x_w)
    len_y_w = len(y_w)
    len_z_w = len(z_w)


    ini_x = ((np.random.rand(n_particles)-0.5)*2*r_x + startx)
    ini_y = ((np.random.rand(n_particles)-0.5)*2*r_y + starty)
    ini_z = ((np.random.rand(n_particles)-0.5)*2*r_z + startz)

    x_stream = np.ones((n_particles,int(np.fabs(total_time/timestep))+2))*startx
    y_stream = np.ones((n_particles,int(np.fabs(total_time/timestep))+2))*starty
    z_stream = np.ones((n_particles,int(np.fabs(total_time/timestep))+2))*startz
    t_stream = np.ones((int(np.fabs(total_time/timestep))+2))*startt
    com_stream = np.ones((3,int(np.fabs(total_time/timestep))+2))

    t_RK = startt #set the initial time to be the given start time
    z_RK = ini_z
    y_RK = ini_y
    x_RK = ini_x

    # define centre of mass variable
    x_com = startx
    y_com = starty
    z_com = startz
    t = startt
    
    normed_radius = ( ((ini_x - startx)**2)/(r_x**2) +  
                      ((ini_y - starty)**2)/(r_y**2) + 
                      ((ini_z - startz)**2)/(r_z**2) )

    wet_test = trilinear_interp_arrays(x_RK,y_RK,z_RK,grid_object['wet_mask_U'][:],x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)

    for particle in xrange(n_particles): # check if any particles outside cutoff
        if (normed_radius[particle] > r_cutoff_factor or wet_test[particle] < 0.5 or z_RK[particle]*grid_object['Z'][2] < 0): # find specific particles
            while (normed_radius[particle] > 1 or wet_test[particle] < 0.5 or z_RK[particle]*grid_object['Z'][2] < 0): # reseed inside ellipsoid
                x_RK[particle] = ((np.random.rand(1)-0.5)*r_x*2 + x_com)
                y_RK[particle] = ((np.random.rand(1)-0.5)*r_y*2 + y_com)
                z_RK[particle] = ((np.random.rand(1)-0.5)*r_z*2 + z_com)
                normed_radius[particle] = ( ((x_RK[particle] - x_com)**2)/(r_x**2) +  
                                  ((y_RK[particle] - y_com)**2)/(r_y**2) + 
                                  ((z_RK[particle] - z_com)**2)/(r_z**2) )
                wet_test[particle] = trilinear_interp(x_RK[particle],y_RK[particle],z_RK[particle],
                    grid_object['wet_mask_U'][:],x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)


    i=0

    # Prepare for spherical polar grids
    deg_per_m = np.ones((n_particles,2),dtype=float)
    if grid_object['grid_type']=='polar':
        deg_per_m[:,0] = np.ones((n_particles))/(1852.*60.) # multiplier for v

    # Runge-Kutta fourth order method to estimate next position.
    while i < np.fabs(total_time/timestep):
        #t_RK < total_time:
        if grid_object['grid_type']=='polar':
            # use degrees per metre and convert all the velocities to degrees / second# calculate degrees per metre at current location - used to convert the m/s velocities in to degrees/s
            deg_per_m[:,1] = np.cos(np.ones((n_particles))*starty*np.pi/180.)/(1852.*60.)# multiplier for u
            # Compute indices at location given

        # check if any particles need moving, and then reseed them inside an ellipsoid around centre of mass with the original size
        for particle in xrange(n_particles): # check if any particles outside cutoff
            if (normed_radius[particle] > r_cutoff_factor or wet_test[particle] < 0.5 or z_RK[particle]*grid_object['Z'][2] < 0): # find specific particles
                while (normed_radius[particle] > 1 or wet_test[particle] < 0.5 or z_RK[particle]*grid_object['Z'][2] < 0): # reseed inside ellipsoid
                    x_RK[particle] = ((np.random.rand(1)-0.5)*r_x*2 + x_com)
                    y_RK[particle] = ((np.random.rand(1)-0.5)*r_y*2 + y_com)
                    z_RK[particle] = ((np.random.rand(1)-0.5)*r_z*2 + z_com)
                    normed_radius[particle] = ( ((x_RK[particle] - x_com)**2)/(r_x**2) +  
                                      ((y_RK[particle] - y_com)**2)/(r_y**2) + 
                                      ((z_RK[particle] - z_com)**2)/(r_z**2) )
                    wet_test[particle] = trilinear_interp(x_RK[particle],y_RK[particle],z_RK[particle],
                        grid_object['wet_mask_U'][:],x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)

        # Interpolate velocities to initial location
        u_loc = trilinear_interp_arrays(x_RK,y_RK,z_RK,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc = trilinear_interp_arrays(x_RK,y_RK,z_RK,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc = trilinear_interp_arrays(x_RK,y_RK,z_RK,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc = u_loc*deg_per_m[:,1]
        v_loc = v_loc*deg_per_m[:,0]
        dx1 = timestep*u_loc
        dy1 = timestep*v_loc
        dz1 = timestep*w_loc

        u_loc1 = trilinear_interp_arrays(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc1 = trilinear_interp_arrays(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc1 = trilinear_interp_arrays(x_RK + 0.5*dx1,y_RK + 0.5*dy1,z_RK + 0.5*dz1,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc1 = u_loc1*deg_per_m[:,1]
        v_loc1 = v_loc1*deg_per_m[:,0]
        dx2 = timestep*u_loc1
        dy2 = timestep*v_loc1
        dz2 = timestep*w_loc1

        u_loc2 = trilinear_interp_arrays(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc2 = trilinear_interp_arrays(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc2 = trilinear_interp_arrays(x_RK + 0.5*dx2,y_RK + 0.5*dy2,z_RK + 0.5*dz2,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc2 = u_loc2*deg_per_m[:,1]
        v_loc2 = v_loc2*deg_per_m[:,0]
        dx3 = timestep*u_loc2
        dy3 = timestep*v_loc2
        dz3 = timestep*w_loc2

        u_loc3 = trilinear_interp_arrays(x_RK + dx3,y_RK + dy3,z_RK + dz3,u,x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)
        v_loc3 = trilinear_interp_arrays(x_RK + dx3,y_RK + dy3,z_RK + dz3,v,x_v,y_v,z_v,len_x_v,len_y_v,len_z_v)
        w_loc3 = trilinear_interp_arrays(x_RK + dx3,y_RK + dy3,z_RK + dz3,w,x_w,y_w,z_w,len_x_w,len_y_w,len_z_w)
        u_loc3 = u_loc3*deg_per_m[:,1]
        v_loc3 = v_loc3*deg_per_m[:,0]
        dx4 = timestep*u_loc3
        dy4 = timestep*v_loc3
        dz4 = timestep*w_loc3

        #recycle the "start_" variables to keep the code clean
        x_RK = x_RK + (dx1 + 2*dx2 + 2*dx3 + dx4)/6
        y_RK = y_RK + (dy1 + 2*dy2 + 2*dy3 + dy4)/6
        z_RK = z_RK + (dz1 + 2*dz2 + 2*dz3 + dz4)/6
        t_RK += timestep
        i += 1

        x_com = np.mean(x_RK)
        y_com = np.mean(y_RK)
        z_com = np.mean(z_RK)

        x_stream[:,i] = x_RK
        y_stream[:,i] = y_RK
        z_stream[:,i] = z_RK
        t_stream[i] = t_RK

        com_stream[0,i] = x_com
        com_stream[1,i] = y_com
        com_stream[2,i] = z_com

        normed_radius = ( ((x_RK - x_com)**2)/(r_x**2) +  
                          ((y_RK - y_com)**2)/(r_y**2) + 
                          ((z_RK - z_com)**2)/(r_z**2) )

        wet_test = trilinear_interp_arrays(x_RK,y_RK,z_RK,grid_object['wet_mask_U'][:],x_u,y_u,z_u,len_x_u,len_y_u,len_z_u)


    return x_stream,y_stream,z_stream,t_stream, com_stream
#####################################################



def bilinear_interp(x0,y0,field,x,y,len_x,len_y):
  """!Do bilinear interpolation of a field. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles.

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
  """!Do trilinear interpolation of the field in three spatial dimensions. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles. It also requires that the entire field can be held in memory at the same time.

  x0,y0, and z0 represent the point to interpolate to.
  """

  # Compute indices at location given
  x_index = x.searchsorted(x0)
  if x_index == 0:
    x_index =1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('x location ', str(x0), ' is outside the model grid - too small')
  elif x_index == len_x:
    x_index =len_x - 1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('x location ', str(x0), ' is outside the model grid - too big')
    
  y_index = y.searchsorted(y0)
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
  z_index = z.searchsorted(z0)
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


  
def trilinear_interp_arrays(x0,y0,z0,field,x,y,z,len_x,len_y,len_z):
  """!Do trilinear interpolation of the field in three spatial dimensions. This function assumes that the grid can locally be regarded as cartesian, with everything at right angles. It also requires that the entire field can be held in memory at the same time.

  x0,y0, and z0 represent the point to interpolate to.
  """

  # Compute indices at location given
  x0[x0<x[0]] = (x[0]+x[1])/2
  x0[x0>x[-1]] = (x[-1]+x[-2])/2
  x_index = x.searchsorted(x0)
  #x_index[x_index == 0] = 1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('x location ', str(x0), ' is outside the model grid - too small')
  #x_index[x_index == len_x] = len_x - 1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('x location ', str(x0), ' is outside the model grid - too big')
    
  y0[y0<y[0]] = (y[0]+y[1])/2
  y0[y0>y[-1]] = (y[-1]+y[-2])/2
  y_index = y.searchsorted(y0)
  #y_index[y_index == 0] = 1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('y location ', str(y0), ' is outside the model grid - too small')
  #y_index[y_index == len_y] = len_y - 1 # a dirty hack to deal with streamlines coming near the edge
    #raise ValueError('y location ', str(y0), ' is outside the model grid - too big')
  
  # np.searchsorted only works for positive arrays :/
  if any(z < 0):
        z0 = -z0
        z = -z
  z0[z0<z[0]] = (z[0]+z[1])/2
  z0[z0>z[-1]] = (z[-1]+z[-2])/2
  z_index = z.searchsorted(z0)
  #z_index[z_index == 0] = 1# a dirty hack to deal with streamlines coming near the surface
    #raise ValueError('z location ', str(z0), ' is outside the model grid - too small')
  #z_index[z_index == len_z] = len_z - 1 # a dirty hack to deal with streamlines coming near the bottom
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
  
  x,y,z,t are vectors of these dimensions.
  """

  field_interp = actual_quadralinear_interp(field[1:3,1:3,1:3],
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
                        x0,y0,z0,t_index,
                        len_x,len_y,len_z,len_t,
                        netcdf_filehandle,variable, bias_field):
            """!A helper function to extract a small 4D hypercube of data from a netCDF file. This isn't intended to be used on its own."""
            
            # Compute indices at location given
            x_index = np.searchsorted(x,x0)
            y_index = np.searchsorted(y,y0)
            # np.searchsorted only works for positive arrays :/
            if z0 < 0:
                z0 = -z0
                z = -z
            z_index = np.searchsorted(z,z0)

            # a bunch of dirty hacks to deal with streamlines coming near the edge
            x_index = max(x_index,1)
            x_index = min(x_index,len_x-3)
            y_index = max(y_index,1)
            y_index = min(y_index,len_y-3)
            z_index = max(z_index,1)
            z_index = min(z_index,len_z-3)
            
            field = netcdf_filehandle.variables[variable][t_index-1:t_index+2,
                           z_index-1:z_index+3,
                           y_index-1:y_index+3,
                           x_index-1:x_index+3]


            field = (field + 
                        bias_field[np.newaxis,z_index-1:z_index+3,y_index-1:y_index+3,x_index-1:x_index+3])
            
            return field,x_index,y_index,z_index


def extract_along_path4D(path_x,path_y,path_z,path_t,
            x_axis,y_axis,z_axis,t_axis,
            netcdf_filename='netcdf file with variable of interest',
            netcdf_variable='momVort3'):
    """!extract the value of a field along a path through a time varying, 3 dimensional field. The field must be in a NetCDF file, since it is assumed to be 4D and very large.

    This can also be used to pull out values at specific locations and times.
    """

    t_index = np.searchsorted(t_axis,path_t[0])
    t_index_new = np.searchsorted(t_axis,path_t[0]) # this is later used to test if new data needs to be read in.
        
    len_x = len(x_axis)
    len_y = len(y_axis)
    len_z = len(z_axis)
    len_t = len(t_axis)

    
    netcdf_filehandle = netCDF4.Dataset(netcdf_filename)  
    field,x_index,y_index,z_index = indices_and_field(x_axis,y_axis,z_axis,
                                            path_x[0],path_y[0],path_z[0],t_index,
                                            len_x,len_y,len_z,len_t,
                                            netcdf_filehandle,netcdf_variable)

    field,x_index_new,y_index_new,z_index_new = indices_and_field(x_axis,y_axis,z_axis,
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

            field,x_index,y_index,z_index = indices_and_field(x_axis,y_axis,z_axis,
                                            path_x[i],path_y[i],path_z[i],t_index,
                                            len_x,len_y,len_z,len_t,
                                            netcdf_filehandle,netcdf_variable)


        # Interpolate field to  location
        field_at_loc = quadralinear_interp(path_x[i],path_y[i],path_z[i],path_t[i],
                    field,
                    x_axis,y_axis,z_axis,t_axis,
                    len_x,len_y,len_z,len_t,
                    x_index,y_index,z_index,t_index)
      

        path_variable[i] = field_at_loc

        t_index_new = np.searchsorted(t_axis,path_t[i])
        x_index_new = np.searchsorted(x_axis,path_x[i])
        y_index_new = np.searchsorted(y_axis,path_y[i])
        if path_z[i] < 0:
            z_index_new = np.searchsorted(-z_axis,-path_z[i])
        else:
            z_index_new = np.searchsorted(z_axis,path_z[i])
            


    netcdf_filehandle.close()
    
    return path_variable



def extract_along_path3D(path_x,path_y,path_z,
            x_axis,y_axis,z_axis,field):
    """!Extract the value of a field along a path through a 3 dimensional field. The field must be passed as an array. Time varying fields are not supported.
    """
        
    len_x = len(x_axis)
    len_y = len(y_axis)
    len_z = len(z_axis)

    path_variable = np.zeros((path_x.shape))
    
    #if np.min(path_z) == np.max(path_z):
        # the path is along a 2D depth surface
    #    surf_loc_array = path_z[0]*np.ones((field.shape))
    #    field_at_depth = functions.extract_on_surface(field,surf_loc_array,z_axis,direction='up')


        # FINISH THIS

    for i in xrange(0,len(path_x)):

        # Interpolate field to  location
        path_variable[i] = trilinear_interp(path_x[i],path_y[i],path_z[i],field,x_axis,y_axis,z_axis,len_x,len_y,len_z)
                
    return path_variable



def extract_along_path2D(path_x,path_y,
            x_axis,y_axis,field,order=3,dx=0,dy=0):
    """!Extract the value of a field along a path through a 2 dimensional field. The field must be passed as an array. Time varying fields are not supported.
    
    ------
    ##Parameters##
    * path_x - x-coordinate of the path.
    * path_y - y-coordinate of the path.
    * x_axis - vector of the x-coordinate of the input field.
    * y_axis - vector of the y-coordinate of the input field.
    * field - the input field, the values of which will be extracted along the path.
    * order - the order for the interpolation function, must be between 1 and 5 inclusive. 1 -> linear, 3 -> cubic. 
    * dx, dy - order of the derivative to take in x or y. Optional arguments that are passed to the scipy cubic interpolation function.
    """
        


    path_variable = np.zeros((path_x.shape))

    if order == 1:
        len_x = len(x_axis)
        len_y = len(y_axis)
        # use linear interpolation function from this module
        
        for i in xrange(0,len(path_x)):
            # Interpolate field to  location
            path_variable[i] = bilinear_interp(path_x[i],path_y[i],field,x_axis,y_axis,len_x,len_y)

    else:
        interp_field = scipy.interpolate.RectBivariateSpline(y_axis,x_axis,field,kx=order,ky=order)

        for i in xrange(0,len(path_x)):

            # Interpolate field to  location
            path_variable[i] = interp_field(path_y[i],path_x[i],dx=dx,dy=dy)

    return path_variable


