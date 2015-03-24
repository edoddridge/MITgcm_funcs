"""!
Useful functions that don't belong elsewhere.

A collection of functions for analysing model output. These include functions to extract surfaces at a constant value of some variable and extract a variable on a surface.

Each function has a detailed docstring.
"""

import numpy as np
import netCDF4
import numba
import sys
import matplotlib.pyplot as plt
import glob
import scipy.interpolate


def extract_surface(input_field,surface_value,axis_vector,direction='down',max_depth=-4000):
    """!Extract a surface (2 dimensions) from the input_field (3 dimensions). 
    The surface represents the location at which input_field == surface_value. 
    Specifying an axis_vector means that it is possible to use this function with non-uniform spaced grids.
    
    The function returns output_array, a 2D array of axis values.
    
    The values of input_field *must* be monotonic in the specified dimension.
    
    input_field: 3 dimensional matrix
    output_array: 2 dimensional hypersurface
    surface_value: value of input_field on the extracted surface
    axis_vector: one dimensional vector specifying the distance between elements of input_field
    direction: optional argument to specify which direction the field increases in. Default is down

    ----------------------

    ##Oceanography example:##
    extract depths for a given temperature.

    
    Some arbitrary surface temperatures

        temp = np.array([[10,11,10],[11,11,13],[12,11,10]])
        input_field = np.zeros((temp.shape[0],temp.shape[1],4))
    
    Make them decrease with depth

        for i in xrange(0,4):
            input_field[:,:,i] = temp - i
        
        print input_field[:,:,0]
        > [[ 10.  11.  10.]
        >  [ 11.  11.  13.]
        >  [ 12.  11.  10.]]

    Define a depth axis

        axis_vector = np.array([1,2,4,7])
    
    Pick out the depth at which the temperature should be 10.2

        surface_value = 10.2
    
        depth_temp_10point2 = extract_surface(input_field, surface_value,axis_vector)
    
        print depth_temp_10point2
        > [[ nan     1.8         nan]
        >  [ 1.8     1.8  4.26666667]
        >  [ 2.4     1.8         nan]]
    """
    if direction == 'down':
        dummy_direction = 1
    elif direction =='up':
        dummy_direction = -1
    else:
        print "direction of decreasing values not defined properly. Should be 'down' or 'up'"
    
    ind = np.zeros((input_field.shape[1], input_field.shape[2]))
    output_array = np.zeros((input_field.shape[1], input_field.shape[2]))

    for i in xrange(0,input_field.shape[1]):
        for j in xrange(0,input_field.shape[2]):
            ind[i,j] = dummy_direction*np.searchsorted(input_field[::dummy_direction,i,j],surface_value)
            
            if ind[i,j] == 0:
                output_array[i,j] = 0#np.nan
            elif ind[i,j] == len(input_field[:,i,j]):
                output_array[i,j] = max_depth
            else:
                output_array[i,j] = ((surface_value - input_field[ind[i,j]-1,i,j])*
                                     (axis_vector[ind[i,j]] - axis_vector[ind[i,j]-1])/
                                     (input_field[ind[i,j],i,j] - input_field[ind[i,j]-1,i,j])
                                     ) + axis_vector[ind[i,j]-1]


    #output_array = linear_interp(input_field,surface_value,ind,axis_vector,dummy_direction)

    return output_array
    
@numba.jit
def linear_interp(input_field,surface_value,ind,axis_vector,dummy_direction):
    """!Numba accelerated linear interpolation function. This was seperate from the extract_surface function, to allow numba to work its magic. But now it's not being used."""
    
    output_array = np.zeros((input_field.shape[1], input_field.shape[2]))

    for i in xrange(0,input_field.shape[1]):
        for j in xrange(0,input_field.shape[2]):
            ind[i,j] = dummy_direction*np.searchsorted(input_field[::dummy_direction,i,j],surface_value)
        
            if ind[i,j] == 0:
                output_array[i,j] = 0#np.nan
            elif ind[i,j] == len(input_field[:,i,j]):
                output_array[i,j] = max_depth
            else:
                output_array[i,j] = ((surface_value - input_field[ind[i,j]-1,i,j])*
                                 (axis_vector[ind[i,j]] - axis_vector[ind[i,j]-1])/
                                 (input_field[ind[i,j],i,j] - input_field[ind[i,j]-1,i,j])
                                 ) + axis_vector[ind[i,j]-1]

  #   output_array = np.zeros((input_field.shape[1], input_field.shape[2]))
  #   for i in xrange(0,input_field.shape[1]):
  #       for j in xrange(0,input_field.shape[2]):
  #           if ind[i,j] == 0:
  #               output_array[i,j] = 0#np.nan
  #           elif ind[i,j] == len(input_field[:,i,j]):
		# output_array[i,j] = max_depth
  #           else:
  #               output_array[i,j] = ((surface_value - input_field[ind[i,j]-1,i,j])*
  #                                    (axis_vector[ind[i,j]] - axis_vector[ind[i,j]-1])/
  #                                    (input_field[ind[i,j],i,j] - input_field[ind[i,j]-1,i,j])
  #                                    ) + axis_vector[ind[i,j]-1]
    return output_array
    



def extract_on_surface(input_field,surface_values,axis_values,direction='up'):
    """!Extract the value of a 3D field on a 2D surface.
    This function takes an 3 dimensional matrix 'input_field' and an 2 dimensional
    matrix 'surface_values' and returns a 2 dimensional matrix that contains the
    values of input_field at the location specified by surf_loc_array along the third dimension using
    the values for that axis contained in 'axis_values'. Linear interpolation is used to find the values.
    
    direction: optional argument to specify which direction the axis increases in. Default is up

    """
    
    #if np.max(surf_loc_array) > np.max(axis_values):
    #    print 'At least one value in surf_loc_array is larger than the largest value in axis_values'
        
    #if np.min(surf_loc_array) < np.min(axis_values):
    #    print 'At least one value in surf_loc_array is smaller than the smallest value in axis_values'
        
    if not (all(np.diff(axis_values))<0 or all(np.diff(axis_values))>0):
        print 'axis_vector is not monotonic. This is a problem.'
        
    if direction == 'down':
        dummy_direction = 1
    elif direction =='up':
        dummy_direction = -1
    else:
        print "direction of decreasing values not defined properly. Should be 'down' or 'up'"

    # broadcast depth to the shape of the field
    try:
        surf_loc_array = (surface_values*
                    np.ones((input_field.shape[1],input_field.shape[2])))
    except ValueError:
        print "input_field and surface_values have incompatible shapes"
        return
   
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


@numba.jit
def numerics_extract_on_surface(input_field,surf_loc_array,axis_values,dummy_direction,value_on_surf):

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


def layer_integrate(upper_contour, lower_contour, axis, integrand = 'none', axis_sign = 'negative'): 
    """!Integrate between two non-trivial surfaces, 'upper_contour' and 'lower_contour'. The arrays ind_upper and ind_lower come from the function extract_surface.
    At the moment this only works if all the inputs are defined at the same location.
    
    In MITgcm world, the axis needs to be Zl - 'the lower interface locations'. It needs to include the surface, but the lowest grid face is not required.

    The input array 'integrand' is optional. If it is not included then the output is the volume (per unit area) between the two surfaces at each grid point, 

    ##Examples:##

        contour_10375, ind_10375 = extract_surface(density_diags_mean.rho[:],1037.5,grid.Z[:])
        contour_1038, ind_1038 = extract_surface(density_diags_mean.rho[:],1038,grid.Z[:])

    Volume between two stratification surfaces. In this case the optional argument 'integrand' is not required.
    
        volume = integrate_layerwise(contour_10375, ind_10375,contour_1038, ind_1038,grid.Z[:])

    Kinetic energy between two stratification surfaces. The integrand is the kinetic energy.
    
        EKE_10375_1038 = integrate_layerwise(contour_10375, ind_10375,contour_1038, ind_1038,grid.Z[:],EKE)


        upper = -1 * np.array([[1,1,1],[1,1,1],[1,1,1]])
        lower = -1 * (np.array([[-0.9,1,1],[1,1,1],[1,1,1]]) + 2)
        axis = -1 * np.array([0.5,1.2,1.6,2.1,2.6,3.1])
        test = layer_integrate(upper,lower,axis)
        print test
        > [[ 0.1  2.   2. ]
        >  [ 2.   2.   2. ]
        >  [ 2.   2.   2. ]]
        print np.sum(test)
        > 16.1
        print np.sum(upper - lower)
        > 16.1
    """

    if axis_sign == 'positive':
        dummy_sign = 1
    elif axis_sign =='negative':
        dummy_sign = -1
    else:
        print "axis_sign not specified correctly. Should be 'positive' or 'negative'."
    
    total = np.zeros((upper_contour.shape))
    
    
    if integrand == 'none':
        total = (upper_contour - lower_contour)

    else:
        for i in xrange(0,upper_contour.shape[1]):
            for j in xrange(0,upper_contour.shape[0]):
                ind_upper = np.searchsorted(dummy_sign*axis[:],dummy_sign*upper_contour[j,i],side='right')

                ind_lower = np.searchsorted(dummy_sign*axis[:],dummy_sign*lower_contour[j,i],side='right')

                if ind_lower == ind_upper:
                    total[j,i] = (upper_contour[j,i] - lower_contour[j,i])*integrand[ind_upper-1,j,i]
                    #print 'equal indicies, contribution is ', total[j,i]
                else:
                    total[j,i] = ((upper_contour[j,i] - axis[ind_upper])*integrand[ind_upper-1,j,i] + #upper partial
                                 (axis[ind_lower-1] - lower_contour[j,i])*integrand[ind_lower-1,j,i]) # lower partial
                    #print 'uuper partial = ', (upper_contour[j,i] - axis[ind_upper])*integrand[ind_upper-1,j,i]
                    #print 'lower partial = ', (axis[ind_lower-1] - lower_contour[j,i])*integrand[ind_lower-1,j,i]

                    #sum the full cells in between
                    for k in xrange(ind_upper,ind_lower-1): #goes up to the ind_lower-2 to ind_lower-1 cell
                        total[j,i] += (axis[k] - axis[k+1]) * integrand[k,j,i]
    return total
    
    
def test_layer_integrate():
    integrand = np.ones((len(axis),upper_contour.shape[0],upper_contour.shape[1]))
    upper = -1 * np.array([[1,1,1],[1,1,1],[1,1,1]])
    lower = -1 * (np.array([[-0.9,1,1],[1,1,1],[1,1,1]]) + 2)
    axis = -1 * np.array([0.5,1.2,1.6,2.1,2.6,3.1])
    assert layer_integrate(upper,lower,axis,integrand=integrand) == np.sum(upper - lower)







def interp_field(field,old_x,old_y,new_x,new_y,interp_order):
    """!Interpolate a given field onto a different grid. Only performs interpolation in the horizontal directions.
    
    None of the grids need to be specified - the zoom factor determines the resolution of the returned field.
    
    ----
    ##Parameters##
    * field - the variable to be interpolated
    * old_x, old_y - the axis on which the original field is defined.
    * new_x, new_y - the axis onto which the field will be interpolated.
    * interp_order - the order of the interpolation function, integer between 0 and 5 inclusive. 1 -> linear, 3 -> cubic, &c.."""


    mask = np.ones((np.shape(field)))
    mask[field == 0.] = 0.

    field_interp = np.zeros((field.shape[0],
                     len(new_y),
                     len(new_x)))

    kx = order
    ky = order

    interp_object = scipy.interpolate.RectBivariateSpline(y_axis,x_axis,field,kx,ky)


    for k in xrange(0,field.shape[0]):
        field_interp[k,:,:] = interp_object(new_y,new_x)


    return field_interp




def export_binary(filename,field,dtype='float64'):
    """!Export binary files that can be imported into the MITgcm.
    The files are big endian, and the datatype can either be 'float64' (= double precision), or 'float32' (=single precision).

    Might not work for very large datasets."""
    
    data = np.array(field,dtype=dtype) # with defined precision, either float32 or float64
    if sys.byteorder == 'little': data.byteswap(True)
    fid = open(filename,"wb")
    data.tofile(fid) # this does not work for very large data sets
    fid.close()


     
def show_variables(netcdf_filename):
    """!A shortcut function to display all of the variables contained within a netcdf file.

    If the filename contains wildcards, they'll be expanded and only the first file used."""
    files = glob.glob(netcdf_filename)
    netcdf_file = netCDF4.Dataset(files[0])
    print netcdf_file.variables.keys()
    netcdf_file.close()


  
  
def plt_mon_stats(netcdf_filename,
                    variables=['advcfl_uvel_max','advcfl_vvel_max','advcfl_wvel_max'],
                    time_units='days'):
    """!Plot some monitor file variables. 'netcdf_filename' can contain shell wild cards, but only the first matching file will be used.
    
    Options include:
    * advcfl_uvel_max
    * advcfl_vvel_max
    * advcfl_wvel_max
    * ke_mean
    * dynstat_theta_mean
    * dynstat_sst_mean
    * dynstat_sst_sd
    * dynstat_salt_max
    * dynstat_salt_min
    * dynstat_uvel_mean
    * dynstat_vvel_mean
    * dynstat_wvel_mean
    * ...
    * and many others.
    
    """
    files = glob.glob(netcdf_filename)
    monitor_output = netCDF4.Dataset(files[0])
    
    if time_units == 'days':
        time = monitor_output.variables['time_secondsf'][:]/(86400)
    elif time_units == 'years':
        time = monitor_output.variables['time_secondsf'][:]/(86400*365)
    else:
        raise ValueError(str(time_units) + ' is not a valid option for time_units')
    data = {}
    for stat in variables:
        data[stat] = monitor_output.variables[stat][:]
        plt.plot(time,data[stat],label=stat)
    plt.xlabel('Model '+ time_units)
    plt.legend()

    return data


