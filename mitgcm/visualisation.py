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
    output = LIC2_sparse(m.zonal_velocity['UVEL'][3,:,:],m.meridional_velocity['VVEL'][3,:,:],
                                                        5000,m.grid,15*86400,kernel='anisotropic_linear',delta_t=10*3600)

    white_var_alpha = mitgcm.visualisation.create_cmap_vary_alpha(colour='k')
	\endcode
	 
	Then plot with

	\code{.py}
	plt.pcolormesh(m.grid['X'][:],m.grid['Yp1'][:],m.meridional_velocity['VVEL'][level,:,:],cmap='winter',vmin=-1,vmax=1)
	plt.colorbar()
	for i in xrange(output.shape[1]):
    	plt.scatter(output[0,i,:],output[1,i,:],
                c=output[2,i,:],cmap=white_var_alpha,lw=0,vmin=0,vmax=1,s=4)
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

    return output


def LIC2_sparse_animate(u,v,nparticles,grid_object,animation_length,trace_length,kernel='anisotropic_linear',
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
    output,intensity = LIC2_sparse_animate(m.zonal_velocity['UVEL'][1,:,:],
                                                                 m.meridional_velocity['VVEL'][1,:,:],
                                                                 15,m.grid,
                                                                 300*86400,60*86400,
                                                                 kernel='anisotropic_linear',delta_t=4*3600)
    white_var_alpha = mitgcm.visualisation.create_cmap_vary_alpha(colour='k')
    \endcode

    and then plot with

    \code{.py}
    trace_length= len(intensity)

    for t0 in xrange(output.shape[2]-len(intensity)):
        for i in xrange(output.shape[0]):
                    plt.scatter(output[i,0,t0:t0+trace_length],output[i,1,t0:t0+trace_length],
                        c = intensity,cmap='winter',lw=0,vmin=0,vmax=1,s=20)

        filename = '../OLIC animation{:04g}.png'.format(t0)
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

        noise = np.zeros((steps_per_trace))
        noise[int(len(noise)/2)] = 1
        
        intensity = np.zeros((steps_per_ani))
        for i in xrange(kernel_length):
            intensity[i] = np.sum(k*noise[i:kernel_length+i])
        intensity = intensity/np.max(intensity)

    else:
        raise ValueError('Valid options for kernel are: anisotropic_linear, box')
        


    
    output = np.zeros((nparticles,2,steps_per_ani))
    #intensity = np.zeros((2,nparticles,steps_per_ani+steps_per_trace))
    
    x_start = (np.random.random_sample(nparticles)*(np.max(grid_object['Xp1'][:]) - np.min(grid_object['Xp1'][:])) +
                                                    np.min(grid_object['Xp1'][:]))
    y_start = (np.random.random_sample(nparticles)*(np.max(grid_object['Yp1'][:]) - np.min(grid_object['Yp1'][:])) +
                                                    np.min(grid_object['Yp1'][:]))
    # first lot of particle tracks 
    for i in xrange(nparticles):


        x_stream,y_stream,t_stream = mitgcm.streamlines.stream2(u,v,x_start[i],y_start[i],
                                                            grid_object,trace_length,delta_t)


        output[i,0,:steps_per_trace] = x_stream[:steps_per_trace]
        output[i,1,:steps_per_trace] = y_stream[:steps_per_trace]

    
    # second lot of particle tracks 
    # these ones start randomly during the first lot
    x_start = (np.random.random_sample(nparticles)*(np.max(grid_object['Xp1'][:]) - np.min(grid_object['Xp1'][:])) +
                                                    np.min(grid_object['Xp1'][:]))
    y_start = (np.random.random_sample(nparticles)*(np.max(grid_object['Yp1'][:]) - np.min(grid_object['Yp1'][:])) +
                                                    np.min(grid_object['Yp1'][:]))
    for i in xrange(nparticles):

        x_stream,y_stream,t_stream = mitgcm.streamlines.stream2(u,v,x_start[i],y_start[i],
                                                            grid_object,trace_length,delta_t)
        
        t0 = int(np.random.random_sample(1)*(steps_per_trace))


        output[i,0,t0:t0+steps_per_trace] = x_stream[:steps_per_trace]
        output[i,1,t0:t0+steps_per_trace] = y_stream[:steps_per_trace]


    # Now have two sets of particle tracks for each particle
    # and the second lot have starting times that are random
    # Now, go through and put in the rest of the animations worth of tracks
    # Do this by finding places in the output array where there aren't yet particle tracks
    for n in xrange(nparticles):
        
        for trace in xrange(int(np.floor(animation_length/trace_length)-2)):
        
            indices = np.where(output[n,0,:]==output[n,1,:])
            t_ind = indices[0][0]
            
            if t_ind+steps_per_trace < steps_per_ani:

                x_start = (np.random.random_sample(1)*(np.max(grid_object['Xp1'][:]) - np.min(grid_object['Xp1'][:])) +
                                                            np.min(grid_object['Xp1'][:]))
                y_start = (np.random.random_sample(1)*(np.max(grid_object['Yp1'][:]) - np.min(grid_object['Yp1'][:])) +
                                                            np.min(grid_object['Yp1'][:])) 

                x_stream,y_stream,t_stream = mitgcm.streamlines.stream2(u,v,x_start,y_start,
                                                                    grid_object,trace_length,delta_t)

                output[n,0,t_ind:t_ind+steps_per_trace] = x_stream[:steps_per_trace]
                output[n,1,t_ind:t_ind+steps_per_trace] = y_stream[:steps_per_trace]


    return output,intensity[:int(steps_per_kernel/2.)]


def create_cmap_vary_alpha(colour='white'):
    """!Create a colour map with variable alpha. This can be used to sketch out particles as they move.

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
