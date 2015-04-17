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
import warnings
import mitgcm

    
def calc_surface(input_array,surface_value,axis_values,method='linear'):
    """ nearest finds the index jusut before the search value. linear uses linear interpolation to find the location between grid points.
    May give silly answers if the input_array is not monotonic in the search direction."""

    axis=0
    monoton_test = np.diff(input_array,axis=axis)
    
    if np.all(monoton_test <= 0) or np.all(monoton_test >= 0):
        pass
    else:
        warnings.warn("input field is not strictly monotonic in search direction. Strange results may occur.", RuntimeWarning)
    
    dist = (input_array - surface_value)
    
    indsz = np.repeat(np.arange(input_array.shape[0]-1).reshape((input_array.shape[0]-1,1,1)),input_array.shape[1],axis=1)
    indsz = np.repeat(indsz.reshape((input_array.shape[0]-1,input_array.shape[1],1)),input_array.shape[2],axis=2)

    sign= np.sign(dist)  
    sign[sign==0] = -1     # replace zeros with -1  
    indices_min = np.where(np.diff(sign,axis=axis),indsz,0)
    indices_min = (np.argmax(indices_min,axis=axis)) # to deal with multiple crossings

    if method =='nearest':
        z_surface = np.take(axis_values,indices_min[:,:])

    elif method == 'linear':
        z_nearest = np.take(axis_values,indices_min[:,:])

        indsy = np.repeat(np.arange(indices_min.shape[0]).reshape((indices_min.shape[0],1)),indices_min.shape[1],axis=1)
        indsx = np.repeat(np.arange(indices_min.shape[1]).reshape((1,indices_min.shape[1])),indices_min.shape[0],axis=0)

        above = input_array[indices_min[:,:]-1,indsy,indsx]
        nearest = input_array[indices_min[:,:],indsy,indsx]
        below = input_array[indices_min[:,:]+1,indsy,indsx]

        direction = np.zeros(indices_min.shape,dtype='int64')

        # python refuses to index with a two-component conditional, so it needs to be done in two parts.
        test1 = above > surface_value
        test2 = surface_value > nearest
        direction[test1 == test2] = -1 

        test1 = above < surface_value
        test2 = surface_value < nearest
        direction[test1 == test2] = -1
        
        test1 = nearest > surface_value
        test2 = surface_value > below
        direction[test1 == test2] = 1
        
        test1 = nearest < surface_value
        test2 = surface_value < below
        direction[test1 == test2] = 1



        z_surface =  np.take(axis_values,indices_min[:,:] + direction) + np.nan_to_num(
                                    ((z_nearest - np.take(axis_values,indices_min[:,:] + direction))/
                                 (input_array[indices_min[:,:],indsy,indsx] - 
                                    input_array[indices_min[:,:] + direction,indsy,indsx]))*
                                 (surface_value - input_array[indices_min[:,:] + direction,indsy,indsx]))



    input_array_masked = np.ma.masked_where(input_array==0,input_array)
    

    test1 = np.nanmax(input_array_masked,axis=0) < surface_value
    test2 = np.nanmin(input_array_masked,axis=0) > surface_value

    mask_condition = test1 + test2

    z_surface = np.ma.masked_where(mask_condition, z_surface)

    return z_surface


def extract_on_surface(input_array,surface_locations,axis_values):
    """!Extract the value of a 3D field on a 2D surface.
    This function takes an 3 dimensional matrix 'input_array' and an 2 dimensional
    matrix 'surface_locations' and returns a 2 dimensional matrix that contains the
    values of input_array at the location specified by surface_locations along the third dimension using
    the values for that axis contained in 'axis_values'. Linear interpolation is used to find the values.
    
    """
    
    axis_array = np.repeat(axis_values.reshape((axis_values.shape[0],1,1)),surface_locations.shape[0],axis=1)
    axis_array = np.repeat(axis_array.reshape((axis_array.shape[0],axis_array.shape[1],1)),surface_locations.shape[1],axis=2)
    
    indsz = np.repeat(np.arange(input_array.shape[0]-1).reshape((input_array.shape[0]-1,1,1)),input_array.shape[1],axis=1)
    indsz = np.repeat(indsz.reshape((input_array.shape[0]-1,input_array.shape[1],1)),input_array.shape[2],axis=2)

    dist = (axis_array - surface_locations)

    sign= np.sign(dist)  
    sign[sign==0] = -1     # replace zeros with -1  
    indices_above = np.where(np.diff(sign,axis=0),indsz,0)
    indices_above = (np.nanmax(indices_above,axis=0)) # to deal with multiple crossings (shouldn't be needed)

    indsy = np.repeat(np.arange(indices_above.shape[0]).reshape((indices_above.shape[0],1)),indices_above.shape[1],axis=1)
    indsx = np.repeat(np.arange(indices_above.shape[1]).reshape((1,indices_above.shape[1])),indices_above.shape[0],axis=0)

    values_above = input_array[indices_above[:,:],indsy,indsx]
    values_below = input_array[indices_above[:,:]+1,indsy,indsx]

    axis_above = axis_array[indices_above[:,:],indsy,indsx]
    axis_below = axis_array[indices_above[:,:]+1,indsy,indsx]

    surface_values = values_above + ((values_below - values_above)/(axis_below - axis_above))*(surface_locations - axis_above)
    # value above + linear interpolation of final partial cell.

    return surface_values


def layer_integrate(upper_contour, lower_contour, axis_values, integrand = 'none'): 
    """!Integrate between two non-trivial surfaces, 'upper_contour' and 'lower_contour'. 
    At the moment this only works if all the inputs are defined at the same grid location.
    
    In MITgcm world, the axis_values needs to be Z - the tracer levels.

    The input array 'integrand' is optional. If it is not included then the output is the volume per unit area (difference in depth) between the two surfaces at each grid point, 
    """
    
    total = np.zeros((upper_contour.shape))

    
    if integrand == 'none':
        total = np.absolute(upper_contour - lower_contour)

    else:
        axis_array = np.repeat(axis_values.reshape((axis_values.shape[0],1,1)),upper_contour.shape[0],axis=1)
        axis_array = np.repeat(axis_array.reshape((axis_array.shape[0],axis_array.shape[1],1)),upper_contour.shape[1],axis=2)
        
        indsz = np.repeat(np.arange(integrand.shape[0]-1).reshape((integrand.shape[0]-1,1,1)),integrand.shape[1],axis=1)
        indsz = np.repeat(indsz.reshape((integrand.shape[0]-1,integrand.shape[1],1)),integrand.shape[2],axis=2)

        dist = (axis_array - upper_contour)
        sign= np.sign(dist)  
        sign[sign==0] = -1     # replace zeros with -1  
        indices_upper = np.where(np.diff(sign,axis=0),indsz,0)
        indices_upper = (np.nanmax(indices_upper,axis=0)) # to flatten array and deal with multiple crossings

        dist = (axis_array - lower_contour)
        sign= np.sign(dist)  
        sign[sign==0] = -1     # replace zeros with -1  
        indices_lower = np.where(np.diff(sign,axis=0),indsz,0)
        indices_lower = (np.nanmax(indices_lower,axis=0)) # to flatten array and deal with multiple crossings

        values_upper = mitgcm.functions.extract_on_surface(integrand,upper_contour,axis_values)
        values_lower = mitgcm.functions.extract_on_surface(integrand,lower_contour,axis_values)

        for j in xrange(0,upper_contour.shape[0]):
            for i in xrange(0,upper_contour.shape[1]):
                if (values_upper[j,i] is not np.ma.masked) and (values_lower[j,i] is not np.ma.masked):
                    if indices_upper[j,i] == indices_lower[j,i]:
                        # in the same cell. find midvalue and mulitply by thickness
                        total[j,i] = (values_upper[j,i] + values_lower[j,i])*(upper_contour[j,i] - lower_contour[j,i])/2
                    else:
                        # not in the same cell. Have at least an upper and lower partial cell to compute
                        upper_partial = ((values_upper[j,i] + integrand[indices_upper[j,i]+1,j,i])*
                                        (upper_contour[j,i] - axis_values[indices_upper[j,i]+1]))/2
                        lower_partial = ((values_lower[j,i] + integrand[indices_lower[j,i],j,i])*
                                        (axis_values[indices_lower[j,i]] - lower_contour[j,i]))/2

                        total[j,i] = upper_partial + lower_partial

                        if indices_lower[j,i] - indices_upper[j,i] > 1:
                             # have at least one whole cell between them (the same cell case has already been captured)
                            for k in xrange(indices_upper[j,i]+1,indices_lower[j,i]):
                                total[j,i] += (integrand[k,j,i] + integrand[k+1,j,i])*(axis_values[k] - axis_values[k+1])/2 #

    return total



    
def test_layer_integrate():
    """Test function to check that the layer integrate function is working correctly."""
    upper = -1 * np.array([[1,1,1],[1,1,1],[1,1,1]])
    lower = -1 * (np.array([[-0.9,1,1],[1,1,1],[1,1,1]]) + 2)
    axis = -1 * np.array([0.5,1.2,1.6,2.1,2.6,3.1])
    integrand = np.ones((len(axis),upper.shape[0],upper.shape[1]))
    assert np.sum(layer_integrate(upper,lower,axis,integrand=integrand)) == np.sum(upper - lower)







def interp_field(field,old_x,old_y,new_x,new_y,interp_order):
    """!Interpolate a given field onto a different grid. Only performs interpolation in the horizontal directions.
        
    ----
    ##Parameters##
    * field - the variable to be interpolated
    * old_x, old_y - the axis on which the original field is defined.
    * new_x, new_y - the axis onto which the field will be interpolated.
    * interp_order - the order of the interpolation function, integer between 1 and 5 inclusive. 1 -> linear, 3 -> cubic, &c.."""


    mask = np.ones((np.shape(field)))
    mask[field == 0.] = 0.

    field_interp = np.zeros((field.shape[0],
                     len(new_y),
                     len(new_x)))

    

    for k in xrange(0,field.shape[0]):
        interp_object = scipy.interpolate.RectBivariateSpline(old_y,old_x,field[k,:,:],kx=interp_order,ky=interp_order)
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
                    time_units='days',
                    output_filename=None):
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

    data['time'] = time

    if output_filename != None:
        plt.savefig(output_filename)

    return data


