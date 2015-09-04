"""!
Visualisation functions.

Each function has a detailed docstring.
"""

import numpy as np
import netCDF4
import numba
import sys
import matplotlib.pyplot as plt
import matplotlib.colors
import glob
import scipy.interpolate
import warnings
import copy
import mitgcm


def LIC2_sparse(u,v,points,grid_object,trace_length,kernel='anisotropic_linear',delta_t=3600.):
    """!Line integral convolution with a sparse noise field. This produces discrete points that flow around the visualisation.

    LIC is a method for visualising flow fields.
    
    -------------
    ##Parameters
    * kernel - the convolution kernel. Options are: 
      * 'box'
      * 'anisotropic_linear'



    -------------
    ##Example call

    \code{.py}
    output,myCM = LIC2_sparse(m.zonal_velocity['UVEL'][3,:,:],m.meridional_velocity['VVEL'][3,:,:],
                                                        5000,m.grid,15*86400,kernel='anisotropic_linear',delta_t=10*3600)
	\endcode
	 
	Then plot with

	\code{.py}
	plt.pcolormesh(m.grid['X'][:],m.grid['Yp1'][:],m.meridional_velocity['VVEL'][level,:,:],cmap='winter',vmin=-1,vmax=1)
	plt.colorbar()
	for i in xrange(output.shape[1]):
    	plt.scatter(output[0,i,:],output[1,i,:],
                c=output[2,i,:],cmap=myCM,lw=0,vmin=0,vmax=1,s=4)
	\endcode

	\callgraph
	
    """
    
    
 # temporary variable for setting up the convolution kernel and noise field.
    trace_length = float(trace_length)
    delta_t = float(delta_t)
    steps_per_trace = int(trace_length/delta_t)
    

    if kernel == 'anisotropic_linear':
        k = np.ones(steps_per_trace)
        for i in xrange(steps_per_trace):
            k[i] = 1 - float(i)/float(steps_per_trace-1)
        #k[:int(steps_per_trace/2.)] = 0
        #k[int(2*kernel_length/4.):] = 0
        #k = k/np.sum(delta_t*k)
        
        #plt.plot(k)
        
        noise = np.zeros(2*steps_per_trace)
        noise[steps_per_trace] = 1
        
        intensity = np.zeros((steps_per_trace))
        for i in xrange(steps_per_trace):
            intensity[i] = np.sum(k*noise[i:steps_per_trace+i])
        intensity = intensity/np.max(intensity)
        #plt.plot(intensity)
            
    elif kernel == 'box':
        k = np.ones(kernel_length)
        # k[:int(kernel_length/4.)] = 0
        # k[int(3*kernel_length/4.):] = 0
        #k = k/np.sum(delta_t*k)

        noise = np.zeros(2*steps_per_trace)
        noise[steps_per_trace] = 1
        
        intensity = np.zeros((steps_per_trace))
        for i in xrange(steps_per_trace):
            intensity[i] = np.sum(k*noise[i:steps_per_trace+i])
        intensity = intensity/np.max(intensity)

    else:
        raise ValueError('Valid options for kernel are: anisotropic_linear, box')
        
    
    x_start = (np.random.random_sample(points)*(np.max(grid_object['Xp1'][:]) - np.min(grid_object['Xp1'][:])) +
                                                    np.min(grid_object['Xp1'][:]))
    y_start = (np.random.random_sample(points)*(np.max(grid_object['Yp1'][:]) - np.min(grid_object['Yp1'][:])) +
                                                    np.min(grid_object['Yp1'][:]))
    
    output = np.zeros((3,points,steps_per_trace))
    
    for i in xrange(points):
        x_stream,y_stream,t_stream = mitgcm.streamlines.stream2(u,v,x_start[i],y_start[i],
                                                            grid_object,trace_length,delta_t)

        output[0,i,:steps_per_trace] = x_stream[:steps_per_trace]
        output[1,i,:steps_per_trace] = y_stream[:steps_per_trace]
        output[2,i,:steps_per_trace] = intensity[:steps_per_trace]


            
            
    # define a colour map with alpha varying
    newCM = plt.cm.get_cmap('bone_r')
    newCM._init()

    alphas = np.abs(np.linspace(0, 1.0, newCM.N))
    newCM._lut[:-3,-1] = alphas
    newCM._lut[:,0] = 1             
    newCM._lut[:,1] = 1             
    newCM._lut[:,2] = 1  

    return output,newCM


def LIC2_sparse_animate(u,v,points,grid_object,animation_length,trace_length,kernel='anisotropic_linear',
                 delta_t=3600.):
    """!Line integral convolution with a sparse noise field. The sparse noise field produces discrete points that travel around with the flow field.

    This function produces data that can be used to animate a static flow field.

    LIC is a method for visualising flow fields.
    
    returns:

    * output_matrix = [Variables (this axis has three values: x,y,intensity_ramp),trace_number ,time]
	* intensity = vector containing the convolved kernel and noise field


    -------------
    ##Parameters
    * kernel - the convolution kernel. Options are: 
      * 'box' - same intensity for the entire trace
      * 'anisotropic_linear' - intensity ramps up over the trace


    ------------
    ##Example call

    \code{.py}
    output,intensity,myCM = LIC2_sparse_animate(m.zonal_velocity['UVEL'][3,:,:],m.meridional_velocity['VVEL'][3,:,:],
                                            10000,m.grid,100*86400,30*86400,kernel='anisotropic_linear',delta_t=5*3600)
	\endcode

	and then plot with

	\code{.py}
	n = len(intensity)
	for t in xrange(output.shape[2]-len(intensity)):
	    plt.pcolormesh(m.grid['X'][:],m.grid['Y'][:],m.temperature['THETA'][level,:,:],cmap='winter')
	    plt.colorbar()
	    
	    for i in xrange(output.shape[1]):
	        plt.scatter(output[0,i,t:t+n],output[1,i,t:t+n],
	                c=output[2,i,t:t+n],cmap=myCM,lw=0,vmin=0,vmax=1,s=4)
	        

	    filename = '../b animated LIC test ' + str(t) + '.png'
	    plt.savefig(filename)
	    plt.clf()
    \endcode

    \callgraph
    """
    

    # temporary variable for setting up the convolution kernel and noise field.
    trace_length = float(trace_length)
    kernel_length = float(trace_length/4)
    delta_t = float(delta_t)
    animation_length = float(animation_length)
    steps_per_trace = int(trace_length/delta_t)
    steps_per_kernel = int(kernel_length/delta_t)
    steps_per_ani = int(animation_length/delta_t)
    

    intensity_ramp = np.ones((steps_per_trace))
    for i in xrange(int(steps_per_trace/8.)):
        intensity_ramp[i] = float(i)/float(steps_per_trace/8.)
        intensity_ramp[-i] = float(i)/float(steps_per_trace/8.)

    #plt.plot(intensity_ramp)
    #print intensity_ramp

    if kernel == 'anisotropic_linear':
        k = np.ones(steps_per_kernel)
        for i in xrange(steps_per_kernel):
            k[i] = 1 - float(i)/float(steps_per_kernel-1)
        k[:int(steps_per_kernel/2.)] = 0
        #k[int(2*kernel_length/4.):] = 0
        #k = k/np.sum(delta_t*k)
        
        #plt.plot(k)
        
        noise = np.zeros(steps_per_ani)
        noise[steps_per_kernel] = 1
        
        intensity = np.zeros((steps_per_ani))
        for i in xrange(steps_per_ani - steps_per_kernel):
            intensity[i] = np.sum(k*noise[i:steps_per_kernel+i])
        intensity = intensity/np.max(intensity)
        #plt.plot(intensity)
            
    elif kernel == 'box':
        k = np.ones(kernel_length)
        # k[:int(kernel_length/4.)] = 0
        # k[int(3*kernel_length/4.):] = 0
        #k = k/np.sum(delta_t*k)

        noise = np.zeros_like(intensity_ramp)
        noise[int(len(noise)/2)] = 1
        
        intensity = np.zeros((steps_per_ani))
        for i in xrange(kernel_length):
            intensity[i] = np.sum(k*noise[i:kernel_length+i])
        intensity = intensity/np.max(intensity)

    else:
        raise ValueError('Valid options for kernel are: anisotropic_linear, box')
        

    x_start = (np.random.random_sample(points)*(np.max(grid_object['Xp1'][:]) - np.min(grid_object['Xp1'][:])) +
                                                    np.min(grid_object['Xp1'][:]))
    y_start = (np.random.random_sample(points)*(np.max(grid_object['Yp1'][:]) - np.min(grid_object['Yp1'][:])) +
                                                    np.min(grid_object['Yp1'][:]))
    
    output = np.zeros((3,points,steps_per_ani+int(2*steps_per_trace)))
    #intensity = np.zeros((2,points,steps_per_ani+steps_per_trace))
    
    for i in xrange(points):
        x_stream,y_stream,t_stream = mitgcm.streamlines.stream2(u,v,x_start[i],y_start[i],
                                                            grid_object,trace_length,delta_t)

        t0 = int(np.random.random_sample(1)*(steps_per_ani+int(1*steps_per_trace)))

        output[0,i,t0:t0+steps_per_trace] = x_stream[:steps_per_trace]
        output[1,i,t0:t0+steps_per_trace] = y_stream[:steps_per_trace]
        output[2,i,t0:t0+steps_per_trace] = intensity_ramp[:steps_per_trace]#*intensity[:steps_per_trace]
                        
    # define a colour map with alpha varying
    newCM = plt.cm.get_cmap('bone_r')
    newCM._init()

    alphas = np.abs(np.linspace(0, 1.0, newCM.N))
    newCM._lut[:-3,-1] = alphas
    newCM._lut[:,0] = 1             
    newCM._lut[:,1] = 1             
    newCM._lut[:,2] = 1  

    return output[:,:,steps_per_trace:-steps_per_trace],intensity[:steps_per_kernel],newCM


def create_cmap_vary_alpha(colour='white'):
    """create a colour map with variable alpha. This can be used to sketch out particles as they move.

    The only input variable 'coour' defines the colour that is used to create the colour map. It can be any colour code that matplotlib recognises: single letter codes, hex colour string, a standard colour name, or a string representation of a float (e.g. '0.4') for gray on a 0-1 scale."""
    
    rgb = matplotlib.colors.colorConverter.to_rgb(colour) # convert the input colour to an rgb tuple

    # take a predefined colour map and hack it.
    newCM = copy.deepcopy(plt.cm.get_cmap('bone_r'))
    newCM._init()

    alphas = np.abs(np.linspace(0, 1.0, newCM.N))
    newCM._lut[:-3,-1] = alphas
    newCM._lut[:,0] = rgb[0]            
    newCM._lut[:,1] = rgb[1]        
    newCM._lut[:,2] = rgb[2]

    return newCM
