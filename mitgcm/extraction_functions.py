"""
extraction_functions
====================

Functions for extracting surfaces of constant value, or values on arbitrary surfaces.

Each function has a detailed docstring.
"""

import numpy as np
import numba

def extract_surface(input_field,surface_value,axis_vector,direction='down',max_depth=-4000):
    """Extract a surface (2 dimensions) from the input_field (3 dimensions). 
    The surface represents the location at which input_field == hypersurface_value. 
    Specifying an axis_vector means that it is possible to use this function with non-uniform spaced grids.
    
    The function returns output_array, a 2D array of axis values.
    
    The values of input_field *must* be monotonic in the specified dimension.
    
    input_field: 3 dimensional matrix
    output_array: 2 dimensional hypersurface
    surface_value: value of input_field on the extracted surface
    axis_vector: one dimensional vector specifying the distance between elements of input_field
    direction: optional argument to specify which direction the field increases in. Default is down

    Oceanography example: extract depths for a given temperature.
    
    Some arbitrary surface temperatures
    temp = np.array([[10,11,10],[11,11,13],[12,11,10]])
    input_field = np.zeros((temp.shape[0],temp.shape[1],4))
    
    Make them decrease with depth
    for i in xrange(0,4):
        input_field[:,:,i] = temp - i
        
    print input_field[:,:,0] returns
    [[ 10.  11.  10.]
     [ 11.  11.  13.]
     [ 12.  11.  10.]]

    Pick out the depth at which the temperature should be 10.2
    surface_value = 10.2
    
    Depth axis
    axis_vector = np.array([1,2,4,7])
    
    depth_temp_10point2 = extract_surface(input_field, surface_value,axis_vector)
    
    print depth_temp_10.2 returns
    [[ nan     1.8         nan]
     [ 1.8     1.8  4.26666667]
     [ 2.4     1.8         nan]]
    """
    if direction == 'down':
        dummy_direction = 1
    elif direction =='up':
        dummy_direction = -1
    else:
        print "direction of decreasing values not defined properly. Should be 'down' or 'up'"
    
    ind = np.zeros((input_field.shape[1], input_field.shape[2]))

    # Find the index at which the value becomes larger than surface_value.
    for i in xrange(0,input_field.shape[1]):
        for j in xrange(0,input_field.shape[2]):            
            ind[i,j] = dummy_direction*np.searchsorted(input_field[::dummy_direction,i,j],surface_value)
            # Need to set nans if this happens at one of the extrema - the value 
            # we're after doesn't exist in this vector
            #if ind[i,j] == 0 or ind[i,j] == len(input_field[:,i,j]):
            #    ind[i,j] = np.nan

    #output_array[:] = np.NAN

    # do linear interpolation to get the value of axis_vector at which the transition occours.
    output_array = linear_interp(input_field,surface_value,ind,axis_vector)

    return output_array, ind
    
@numba.jit
def linear_interp(input_field,surface_value,ind,axis_vector):
    output_array = np.zeros((input_field.shape[1], input_field.shape[2]))
    for i in xrange(0,input_field.shape[1]):
        for j in xrange(0,input_field.shape[2]):
            if ind[i,j] == 0:
                output_array[i,j] = 0#np.nan
            elif ind[i,j] == len(input_field[:,i,j]):
		output_array[i,j] = max_depth
            else:
                output_array[i,j] = ((surface_value - input_field[ind[i,j]-1,i,j])*
                                     (axis_vector[ind[i,j]] - axis_vector[ind[i,j]-1])/
                                     (input_field[ind[i,j],i,j] - input_field[ind[i,j]-1,i,j])
                                     ) + axis_vector[ind[i,j]-1]
    return output_array
    



def extract_on_surface(input_field,surf_loc_array,axis_values,direction='up'):
    """This function takes an 3 dimensional matrix 'input_field' and an 2 dimensional
    matrix 'surf_loc_array' and returns a 2 dimensional matrix that contains the
    values of input_field at the location specified by surf_loc_array along the third dimension using
    the values for that axis contained in 'axis_values'
    
    direction: optional argument to specify which direction the axis increases in. Default is up

    """
    
    if np.max(surf_loc_array) > np.max(axis_values):
        print 'At least one value in surf_loc_array is larger than the largest value in axis_values'
        
    if np.min(surf_loc_array) < np.min(axis_values):
        print 'At least one value in surf_loc_array is smaller than the smallest value in axis_values'
        
    if not (all(np.diff(axis_values))<0 or all(np.diff(axis_values))>0):
        print 'axis_vector is not monotonic. This is a problem.'
        
    if direction == 'down':
        dummy_direction = 1
    elif direction =='up':
        dummy_direction = -1
    else:
        print "direction of decreasing values not defined properly. Should be 'down' or 'up'"
    
    value_on_surf = np.zeros((surf_loc_array.shape))
    value_on_surf[:] = np.nan

    for i in xrange(0,surf_loc_array.shape[0]):
        for j in xrange(0,surf_loc_array.shape[1]):
            if np.isnan(surf_loc_array[i,j]):
                value_on_surf[i,j] = np.nan
            elif surf_loc_array[i,j] == 0:
                value_on_surf[i,j] = np.nan
            else:
                k = dummy_direction*np.searchsorted(axis_values[::dummy_direction],surf_loc_array[i,j]) - 1
                
                value_on_surf[i,j] = (input_field[k,i,j] + 
                                      ((surf_loc_array[i,j] - axis_values[k])/
                                      (axis_values[k] - axis_values[k+1]))*
                                      (input_field[k,i,j] - input_field[k+1,i,j])
                                      )
    return value_on_surf