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
import copy
import mitgcm

    
def calc_surface(input_array,surface_value,axis_values,method='linear'):
    """ nearest finds the index just before the search value. linear uses linear interpolation to find the location between grid points.
    May give silly answers if the input_array is not monotonic in the search direction."""


    axis=0
    monoton_test = np.diff(input_array,axis=axis)
    
    if np.all(monoton_test <= 0) or np.all(monoton_test >= 0):
        pass
    else:
        warnings.warn("input field is not strictly monotonic in search direction. Strange results may occur.", RuntimeWarning)
    
    indsz = np.repeat(np.arange(input_array.shape[0]-1).reshape((input_array.shape[0]-1,1,1)),input_array.shape[1],axis=1)
    indsz = np.repeat(indsz.reshape((input_array.shape[0]-1,input_array.shape[1],1)),input_array.shape[2],axis=2)

    dist = (input_array - surface_value)

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
    else:
        raise RuntimeError(str(method), ' not set correctly. Must be "nearest" or "linear".')


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
    if hasattr(surface_locations, "__len__"):
        pass
    else:
        surface_locations = surface_locations*np.ones((input_array.shape[1],input_array.shape[2]))

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


def layer_integrate(upper_contour, lower_contour, grid_obect, integrand = 'none', interp_method='none'): 
    """!Integrate between two non-trivial surfaces, 'upper_contour' and 'lower_contour'. 
    At the moment this only works if all the inputs are defined at the same grid location.
    
    The input array 'integrand' is optional. If it is not included then the output is the volume per unit area (difference in depth) between the two surfaces at each grid point, 

    -------------------
    ## Parameters
    * upper_contour - the higher surface
    * lower_contour - the lower surface
    * axis_values - the vertical axis.
      ** defined at the cell centres if using interp_method = 'linear'
      ** defined at the cell faces if using interp_method = 'none'
    * interp_method - defines whether the function interpolates values of the integrand. Possible options are 'none' or 'linear'.
    * integrand - the field to be integrated between the contours
    """

    if interp_method == 'none':
        axis_values = grid_obect['Zl'][:]
    elif interp_method == 'linear':
        axis_values = grid_obect['Z'][:]

    total = np.ma.zeros((upper_contour.shape))

    
    if integrand == 'none':
        total = np.absolute(upper_contour - lower_contour)

    else:
        axis_array = np.repeat(axis_values.reshape((axis_values.shape[0],1,1)),upper_contour.shape[0],axis=1)
        axis_array = np.repeat(axis_array.reshape((axis_array.shape[0],axis_array.shape[1],1)),upper_contour.shape[1],axis=2)
        
        indsz = np.repeat(np.arange(integrand.shape[0]-1).reshape((integrand.shape[0]-1,1,1)),integrand.shape[1],axis=1)
        indsz = np.repeat(indsz.reshape((integrand.shape[0]-1,integrand.shape[1],1)),integrand.shape[2],axis=2)

        # these code blocks always return the index of the cell above the zero crossing
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


        if interp_method == 'none':
            for j in xrange(0,upper_contour.shape[0]):
                for i in xrange(0,upper_contour.shape[1]):
                    if (values_upper[j,i] is not np.ma.masked) and (values_lower[j,i] is not np.ma.masked):
                        if indices_upper[j,i] == indices_lower[j,i]:
                            # in the same cell. find midvalue and mulitply by thickness
                            total[j,i] = (integrand[indices_upper[j,i],j,i])*(upper_contour[j,i] - lower_contour[j,i])
                            #if total[j,i]<0:
                            #    print 'Problem! total less than zero from same cell'
                        else:
                            # not in the same cell. Have at least an upper and lower partial cell to compute
                            upper_partial = (integrand[indices_upper[j,i],j,i]*
                                            (upper_contour[j,i] - axis_values[indices_upper[j,i]+1]))
                            lower_partial = (integrand[indices_lower[j,i],j,i]*
                                            (axis_values[indices_lower[j,i]] - lower_contour[j,i]))

                            total[j,i] = upper_partial + lower_partial
                            #if total[j,i]<0:
                            #    print 'Problem! value less than zero from two partials'

                            if indices_lower[j,i] - indices_upper[j,i] > 1:
                                 # have at least one whole cell between them (the same cell case has already been captured)
                                for k in xrange(indices_upper[j,i],indices_lower[j,i]-1):
                                    total[j,i] += integrand[k,j,i]*(axis_values[k] - axis_values[k+1]) #
                                    #if (integrand[k,j,i]*(axis_values[k] - axis_values[k+1]))<0:
                                    #    print 'Problem! intermediate cell contribution is less than zero.'
        elif interp_method == 'linear':
            for j in xrange(0,upper_contour.shape[0]):
                for i in xrange(0,upper_contour.shape[1]):
                    if (values_upper[j,i] is not np.ma.masked) and (values_lower[j,i] is not np.ma.masked):
                        # both contours defined at this location
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
                    
                    elif (values_upper[j,i] is  np.ma.masked) and (values_lower[j,i] is not np.ma.masked):
                        # upper contour undefined at this location (has outcropped)

                        total[j,i] += integrand[0,j,i]*grid_obect['drF'][0]/2 # surface half-cell

                        lower_partial = ((values_lower[j,i] + integrand[indices_lower[j,i],j,i])*
                                        (axis_values[indices_lower[j,i]] - lower_contour[j,i]))/2

                        total[j,i] += lower_partial
                            
                        for k in xrange(indices_lower[j,i]):
                            # whole cells
                            total[j,i] += (integrand[k,j,i] + integrand[k+1,j,i])*(axis_values[k] - axis_values[k+1])/2 


                    elif (values_upper[j,i] is not np.ma.masked) and (values_lower[j,i] is np.ma.masked):
                        # lower contour is undefined at this location (has incropped)
                        total[j,i] += integrand[-1,j,i]*grid_obect['drF'][-1]/2 # bottom half-cell


                        upper_partial = ((values_upper[j,i] + integrand[indices_upper[j,i]+1,j,i])*
                                        (upper_contour[j,i] - axis_values[indices_upper[j,i]+1]))/2

                        total[j,i] += upper_partial

                        for k in xrange(indices_upper[j,i],len(axis_values)-1):
                            total[j,i] += (integrand[k,j,i] + integrand[k+1,j,i])*(axis_values[k] - axis_values[k+1])/2 

    return total



    
def test_layer_integrate():
    """Test function to check that the layer integrate function is working correctly."""
    upper = -1 * np.array([[1,1,1],[1,1,1],[1,1,1]])
    lower = -1 * (np.array([[-0.9,1,1],[1,1,1],[1,1,1]]) + 2)
    axis = -1 * np.array([0.5,1.2,1.6,2.1,2.6,3.1])
    integrand = np.ones((len(axis),upper.shape[0],upper.shape[1]))
    assert np.sum(layer_integrate(upper,lower,axis,integrand=integrand)) == np.sum(upper - lower)







def interp_field(field,old_x,old_y,new_x,new_y,interp_order,fill_nans='no',max_its=5):
    """!Interpolate a given field onto a different grid. Only performs interpolation in the horizontal directions.
        
    ----
    ##Parameters##
    * field - the variable to be interpolated
    * old_x, old_y - the axis on which the original field is defined.
    * new_x, new_y - the axis onto which the field will be interpolated.
    * interp_order - the order of the interpolation function, integer between 1 and 5 inclusive. 1 -> linear, 3 -> cubic, &c..
    * fill_nans - if 'no' values in field are not altered. If 'yes', then NaNs in 'field'
    are replace with the mean of surrounding non-NaNs.
    * max_its - maximum number of iterations to perform when healing NaNs in field."""

    field_interp = np.zeros((field.shape[0],
                     len(new_y),
                     len(new_x)))

    for k in xrange(field.shape[0]):
        if fill_nans == 'yes':
            field_slice = replace_nans(field[k,:,:], max_its,0.5,1,'localmean')
            n = 0
            while (field_slice != field_slice).any():
                    field_slice = replace_nans(field_slice[:,:], max_its,0.5,1,'localmean')
                    # repeat the replace_nans call since it can sometimes miss ones in the corners.
                    # need a way to prevent hanging in the while loop
                    if n > max_its:
                        error_message = 'Tried ' + str(max_its) +   ' iterations to heal NaNs in the input field, and failed.'
                        raise RuntimeError(error_message)

                    n += 1
        elif fill_nans == 'no':
            field_slice = field[k,:,:]
        else:
            raise ValueError('fill_nans not set correctly. Should be "yes" or "no". You gave "' + str(fill_nans) + '"')

        interp_object = scipy.interpolate.RectBivariateSpline(old_y,old_x,field_slice,kx=interp_order,ky=interp_order)
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


def replace_nans(array, max_iter, tol, kernel_size, method='localmean'):
    """!Replace NaN elements in an array using an iterative image inpainting algorithm.
    
    The algorithm is the following:
    
    1) For each element in the input array, replace it by a weighted average
       of the neighbouring elements which are not NaN themselves. The weights depends
       of the method type. If ``method=localmean`` weight are equal to 1/( (2*kernel_size+1)**2 -1 )
       
    2) Several iterations are needed if there are adjacent NaN elements.
       If this is the case, information is "spread" from the edges of the missing
       regions iteratively, until the variation is below a certain threshold. 
    
    Parameters
    ----------
    
    array : 2d np.ndarray
        an array containing NaN elements that have to be replaced
    
    max_iter : int
        the number of iterations

    tol: float
        the difference between subsequent iterations for stopping the algorithm
    
    kernel_size : int
        the size of the kernel, default is 1
        
    method : str
        the method used to replace invalid values. Valid options are
        `localmean`.
        
    Returns
    --------
    
    filled : 2d np.ndarray
        a copy of the input array, where NaN elements have been replaced.



    ---------
    ##Acknowledgements
    
    Code for this function is (very slightly modified) from 
    https://github.com/gasagna/openpiv-python/commit/81038df6d218b893b044193a739026630238fb22#diff-9b2f4f9bb8180e4451e8f85164df7217
    which is part of the OpenPIV project.
    * docs here: http://alexlib.github.io/openpiv-python/index.html 
    * code here: https://github.com/gasagna/openpiv-python
        
    """

    filled = np.empty( [array.shape[0], array.shape[1]], dtype=np.float64)
    kernel = np.empty( (2*kernel_size+1, 2*kernel_size+1), dtype=np.float64 ) 

    # indices where array is NaN
    inans, jnans = np.nonzero( np.isnan(array) )
    
    # number of NaN elements
    n_nans = len(inans)
    
    # arrays which contain replaced values to check for convergence
    replaced_new = np.zeros( n_nans, dtype=np.float64)
    replaced_old = np.zeros( n_nans, dtype=np.float64)
    
    # depending on kernel type, fill kernel array

    if method == 'localmean':
        for i in range(2*kernel_size+1):
            for j in range(2*kernel_size+1):
                kernel[i,j] = 1.0
    else:
        raise ValueError( 'method not valid. Should be one of `localmean`.')
    
    # fill new array with input elements
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            filled[i,j] = array[i,j]


    #filled = numerics_replace_nans(max_iter,n_nans,inans,jnans,filled,kernel,kernel_size,tol,replaced_new,replaced_old)
    # make several passes
    # until we reach convergence
    for it in range(max_iter):
        
        # for each NaN element
        for k in range(n_nans):
            i = inans[k]
            j = jnans[k]
            
            # initialize to zero
            filled[i,j] = 0.0
            n = 0
            
            # loop over the kernel
            for I in range(2*kernel_size+1):
                for J in range(2*kernel_size+1):
                   
                    # if we are not out of the boundaries
                    if i+I-kernel_size < filled.shape[0] and i+I-kernel_size >= 0:
                        if j+J-kernel_size < filled.shape[1] and j+J-kernel_size >= 0:
                                                
                            # if the neighbour element is not NaN itself.
                            if filled[i+I-kernel_size, j+J-kernel_size] == filled[i+I-kernel_size, j+J-kernel_size] :
                                
                                # do not sum itself
                                if I-kernel_size != 0 and J-kernel_size != 0:
                                    
                                    # convolve kernel with original array
                                    filled[i,j] += filled[i+I-kernel_size, j+J-kernel_size]*kernel[I, J]
                                    n = n + 1

            # divide value by effective number of added elements
            if n != 0:
                filled[i,j] = filled[i,j] / n
                replaced_new[k] = filled[i,j]
            else:
                filled[i,j] = np.nan
                
        # check if mean square difference between values of replaced 
        #elements is below a certain tolerance
        if np.mean( (replaced_new-replaced_old)**2 ) < tol:
            break
        else:
            for l in range(n_nans):
                replaced_old[l] = replaced_new[l]


    # replace remaining nans with global mean
    inans, jnans = np.nonzero( np.isnan(filled) )
    n_nans = len(inans)
    # for each NaN element
    fill_value = np.nanmean(filled)
    for k in range(n_nans):
        i = inans[k]
        j = jnans[k]
            
        filled[i,j] = fill_value

    # Check the corners - these are the tricky bits
    #if np.nonzero(np.isnan(filled[0,0])):
    #    filled[0,0] = (filled[0,1] + filled[1,0] + filled[1,1])/3

    #if np.nonzero(np.isnan(filled[0,-1])):
    #    filled[0,-1] = (filled[0,-2] + filled[1,-1] + filled[1,-2])/3

    #if np.nonzero(np.isnan(filled[0,0])):
    #    filled[-1,0] = (filled[-1,1] + filled[-2,0] + filled[-2,1])/3

    #if np.nonzero(np.isnan(filled[0,0])):
    #    filled[-1,-1] = (filled[-1,-2] + filled[-2,-1] + filled[-2,-2])/3


    # now go around the edge and fill in any nans found there
    # indices where array is NaN
    #jnans = np.nonzero( np.isnan(filled[0,:]) )
    # number of NaN elements
    #n_nans = len(jnans)
    # for each NaN element
    #for k in range(n_nans):
    #    j = jnans[k]
    #    filled[0,j] = np.nanmean(filled[0,j-1:j+2])

    #jnans = np.nonzero( np.isnan(filled[-1,:]) )
    #n_nans = len(jnans)
    #for k in range(n_nans):
    #    j = jnans[k]
    #    filled[-1,j] = np.nanmean(filled[-1,j-1:j+2])

    #inans = np.nonzero( np.isnan(filled[:,0]) )
    # number of NaN elements
    #n_nans = len(inans)
    # for each NaN element
    #for k in range(n_nans):
    #    i = inans[k]
    #    filled[i,0] = np.nanmean(filled[i-1:i+2,0])

    #inans = np.nonzero( np.isnan(filled[:,-1]) )
    # number of NaN elements
    #n_nans = len(inans)
    # for each NaN element
    #for k in range(n_nans):
    #    i = inans[k]
    #    filled[i,-1] = np.nanmean(filled[i-1:i+2,-1])

    return filled


#@numba.jit
def numerics_replace_nans(max_iter,n_nans,inans,jnans,filled,kernel,kernel_size,tol,replaced_new,replaced_old):
    # make several passes
    # until we reach convergence
    for it in range(max_iter):
        
        # for each NaN element
        for k in range(n_nans):
            i = inans[k]
            j = jnans[k]
            
            # initialize to zero
            filled[i,j] = 0.0
            n = 0
            
            # loop over the kernel
            for I in range(2*kernel_size+1):
                for J in range(2*kernel_size+1):
                   
                    # if we are not out of the boundaries
                    if i+I-kernel_size < filled.shape[0] and i+I-kernel_size >= 0:
                        if j+J-kernel_size < filled.shape[1] and j+J-kernel_size >= 0:
                                                
                            # if the neighbour element is not NaN itself.
                            if filled[i+I-kernel_size, j+J-kernel_size] == filled[i+I-kernel_size, j+J-kernel_size] :
                                
                                # do not sum itself
                                if I-kernel_size != 0 and J-kernel_size != 0:
                                    
                                    # convolve kernel with original array
                                    filled[i,j] += filled[i+I-kernel_size, j+J-kernel_size]*kernel[I, J]
                                    n = n + 1

            # divide value by effective number of added elements
            if n != 0:
                filled[i,j] = filled[i,j] / n
                replaced_new[k] = filled[i,j]
            else:
                filled[i,j] = np.nan
                
        # check if mean square difference between values of replaced 
        #elements is below a certain tolerance
        if np.mean( (replaced_new-replaced_old)**2 ) < tol:
            break
        else:
            for l in range(n_nans):
                replaced_old[l] = replaced_new[l]

    return filled




def shift_vort_to_T(array):
    """! Shift the array from vorticity points to the corresponding tracer point."""
    shifted = (array[...,0:-1] + array[...,1:])/2
    shifted = (shifted[...,0:-1,:] + shifted[...,1:,:])/2
    return shifted

def shift_U_to_T(array):
    """! Shift the array from UVEL points to the corresponding tracer point."""
    shifted = (array[...,0:-1] + array[...,1:])/2

    return shifted

def shift_V_to_T(array):
    """! Shift the array from VVEL points to the corresponding tracer point."""
    shifted = (array[...,0:-1,:] + array[...,1:,:])/2

    return shifted

def shift_W_to_T(array):
    """! Shift the array from WVEL points to the corresponding tracer point."""
    shifted = (array[...,0:-1,:,:] + array[...,1:,:,:])/2

    return shifted

