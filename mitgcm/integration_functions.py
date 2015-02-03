"""
integration_functions
====================

Functions for integrating between arbitrary surfaces.

Each function has a detailed docstring.
"""
import numpy as np

def layer_integrate(upper_contour, lower_contour, axis, integrand = 'none', axis_sign = 'negative'): 
    """Integrate between two non-trivial surfaces, 'upper_contour' and 'lower_contour'. The arrays ind_upper and ind_lower come from the function extract_surface.
    At the moment this only works if all the inputs are defined at the same location.
    
    In MITgcm world, the axis needs to be Zl - 'the lower interface locations'. It needs to include the surface, but the lowest grid face is not required.

    The input array 'integrand' is optional. If it is not included then the output is the volume (per unit area) between the two surfaces at each grid point, 

    Examples:
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
    print np.sum(test)
    print np.sum(upper - lower)

    [[ 0.1  2.   2. ]
     [ 2.   2.   2. ]
     [ 2.   2.   2. ]]
    16.1
    16.1
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