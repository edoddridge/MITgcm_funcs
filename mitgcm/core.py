"""!
The main file with the class definitions

Core
==============

This file contains all of the classes for the module. It has the base MITgcm_Simulation class, and all of the subclasses for different types of fields. Each class has methods for taking derivatives and doing useful manipulations. It also has some functions that might be of use.
"""

import numpy as np
import netCDF4
import copy
import os
import numba
import matplotlib.pyplot as plt
import glob
import collections
    
class MITgcm_Simulation(dict):
    """!The simulation class is the main class of this package, and an instance of this class is a model object. All fields are associated with the model object - either directly (it is a dict), or indirectly through one of its subobjects (which are also dicts).

    """
    def __init__(self,output_dir,grid_netcdf_filename,EOS_type='linear',g=9.8,
                    ntiles_x=1,ntiles_y=1):
        """!Instantiate an MITgcm model instance.

        ----
        # Parameters #

        * output_dir - the location in which the grid netcdf file is, and the rest of the data can be found
        * grid_netcdf_filename - name of the grid file
        * EOS_type - the equation of state used. 'linear' is the only supported equation of state at the moment
        * g - gravity
        * ntiles_x - number of tiles in the x direction (nPx in MITgcm speak)
        * ntiles_y - number of tiles in the y direction (nPy in MITgcm speak)
        """
        
        os.chdir(output_dir)
        self['output_dir'] = output_dir

        self.grid = Grid(grid_netcdf_filename)
        

        self['g'] = g
        self['EOS_type'] = EOS_type
        if EOS_type != 'linear':
            raise ValueError('Only linear equation of state is currently supported')
            
        self['ntiles_x'] = ntiles_x
        self['ntiles_y'] = ntiles_y

    def load_field(self,model_instance, netcdf_filename,variable,time_level=-1,field_name=None,grid_loc='T'):
        """!Load a model field from NetCDF output. This function associates the field with the object it is called on.

        time_level can be an integer or an array of integers. If it's an array, then multiple time levels will be returned as a higher dimensional array.

        netcdf_filename can be a string with shell wildcards - they'll be expanded and all of the files loaded. BUT, this is intended as a way to take a quick look at the model. I would *strongly* recommend using something like "gluemncbig" to join the multiple tiles into one file before doing proper analysis."""
        if field_name == None:
            field_name = variable

        file_list = glob.glob(netcdf_filename)

        self[field_name] = self.load_from_file(model_instance,file_list,variable,time_level,grid_loc)
        return 

    def load_from_file(self,model_instance, file_list,variable,time_level,grid_loc):
        """!Internal function to pull the data from the file(s). It is called by "load_field", and probably shouldn't be used independently."""

        # remove overlapping regions from tiles
        if grid_loc == 'T':
            clip_x = None
            clip_y = None
        elif grid_loc == 'W':
            clip_x = None
            clip_y = None
        elif grid_loc == 'V':
            clip_x = None
            clip_y = -1
        elif grid_loc == 'U':
            clip_x = -1
            clip_y = None
        elif grid_loc == 'Zeta':
            clip_x = -1
            clip_y = -1
        else:
            print "grid_loc variable not set correctly"
            return

        data = {}

        if time_level == 'All':
            print 'Loading all available time levels in ' + str(variable) + '. This could take a while.'

            for files in file_list:
                netcdf_file = netCDF4.Dataset(files)
                index = files.find('.t')
                tile = files[index+2:-3]
                if variable in netcdf_file.variables.keys():
                    data[tile] = netcdf_file.variables[variable][:]
                else:
                    print variable, 'not in', files, '- aborting import'
                    netcdf_file.close()
                    return

                netcdf_file.close()

        else:
            for files in file_list:
                netcdf_file = netCDF4.Dataset(files)
                index = files.find('.t')
                tile = files[index+2:-3]
                if variable in netcdf_file.variables.keys():
                    data[tile] = netcdf_file.variables[variable][time_level,...]
                else:
                    print variable, 'not in', files, '- aborting import'
                    netcdf_file.close()
                    return
                netcdf_file.close()


        if len(file_list) == 1:
            loaded_field = data[tile]

        else:
            data2 = collections.OrderedDict(sorted(data.items(), key=lambda t: t[0]))

            del data #since the fields can be big, might as well get rid of the duplicate.

            # for tile in data2.keys():
            #     data2[tile] = 0*data2[tile] + float(tile)

            strip_x = {}
            tiles = data2.keys()

            if model_instance['ntiles_x'] > 1:

                for i in xrange(0,model_instance['ntiles_y']):
                    strip_x[i] = np.concatenate([data2[tiles[n]][...,:clip_x]
                                        for n in xrange(i*model_instance['ntiles_x'],
                                        (i+1)*model_instance['ntiles_x'] - 1) ],axis=-1)

                    strip_x[i] = np.concatenate((strip_x[i],data2[tiles[n+1]][...,:,:]),axis=-1)
            else:
                strip_x = data2

            strip_x_keys = strip_x.keys()

            if model_instance['ntiles_y'] > 1:

                loaded_field = np.concatenate([strip_x[strip_x_keys[n]][...,:clip_y,:] 
                                        for n in xrange(0,len(strip_x_keys)-1)],axis=-2)

                loaded_field = np.concatenate((loaded_field,strip_x[strip_x_keys[n+1]][...]),axis=-2)
            else:
                loaded_field = strip_x[strip_x_keys[0]]


        return loaded_field


    def __add__(self,other):
        """!A method that allows model objects to be added together. It does element wise addition for each of the fields."""
        me = copy.deepcopy(self)
        for key, value in me.__dict__.iteritems():
            for key1, value1 in other.__dict__.iteritems():
                if key == key1:
                    setattr(me, key, value + value1)
        return me

    def __div__(self,other):
        """!A method that allows model objects to be divided by floating point numbers. Each field is divided by a floating point number."""
        me = copy.deepcopy(self)
        for key, value in me.__dict__.iteritems():
                setattr(me, key, value/float(other))
        return me

    def __mul__(self, other):
        """! A method that allows model objects to be multiplied by floating point numbers. Each field is multiplied by a floating point number."""
        me = copy.deepcopy(self)
        for key, value in me.__dict__.iteritems():
                setattr(me, key, value * float(other))
        return me

    def __rmul__(self, other):
        """! A method that allows model objects to be multiplied by floating point numbers. Each field is multiplied by a floating point number."""
        me = copy.deepcopy(self)
        for key, value in me.__dict__.iteritems():
                setattr(me, key, value * float(other))
        return me

class Upoint_field(MITgcm_Simulation):
    """! This is the class for all fields on zonal velocity points."""
    
    def __init__(self,model_instance,netcdf_filename,variable,time_level=-1,empty=False,field_name=None):
        if empty:
            pass
        else:
            self.load_field(model_instance,netcdf_filename,variable,time_level,field_name,grid_loc='U')

        return

    def load_field(self,model_instance, netcdf_filename,variable,time_level=-1,field_name=None,grid_loc='U'):
        """!Load a model field from NetCDF output. This function associates the field with the object it is called on.

        time_level can be an integer or an array of integers. If it's an array, then multiple time levels will be returned as a higher dimensional array.

        netcdf_filename can be a string with shell wildcards - they'll be expanded and all of the files loaded. BUT, this is intended as a way to take a quick look at the model. I would *strongly* recommend using something like "gluemncbig" to join the multiple tiles into one file before doing proper analysis."""
        if field_name == None:
            field_name = variable

        file_list = glob.glob(netcdf_filename)

        self[field_name] = self.load_from_file(model_instance,file_list,variable,time_level,grid_loc)
        return 
    
    ### Derivatives of model fields    
    def take_d_dx(self,model_instance,input_field = 'UVEL',output_field='dU_dx'):
        """! Take the x derivative of the field given on u-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            np.seterr(divide='ignore')
            UVEL = self[input_field]
            dU_dx = np.zeros((UVEL.shape))

            for i in xrange(1,UVEL.shape[2]-2):
                dU_dx[:,:,i] = np.nan_to_num(model_instance.grid['wet_mask_U'][:,:,i]*
                        (model_instance.grid['wet_mask_U'][:,:,i+1]*UVEL[:,:,i+1] + 
                        (1 - model_instance.grid['wet_mask_U'][:,:,i+1])*UVEL[:,:,i] - 
                        (1 - model_instance.grid['wet_mask_U'][:,:,i-1])*UVEL[:,:,i] - 
                        model_instance.grid['wet_mask_U'][:,:,i-1]*UVEL[:,:,i-1])/(
                        model_instance.grid['wet_mask_U'][:,:,i-1]*model_instance.grid['dxF'][:,i-1] + 
                        model_instance.grid['wet_mask_U'][:,:,i+1]*model_instance.grid['dxF'][:,i]))
            i = 0
            dU_dx[:,:,i] = (UVEL[:,:,i+1] - UVEL[:,:,i])/(model_instance.grid['dxF'][:,i])
            i = UVEL.shape[2]-1
            dU_dx[:,:,i] = (UVEL[:,:,i] - UVEL[:,:,i-1])/(model_instance.grid['dxF'][:,i-1])

            self[output_field] = dU_dx
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
            
        return 

    
    

    def take_d_dy(self,model_instance,input_field = 'UVEL',output_field='dU_dy'):
        """! Take the y derivative of the field on u points, using the spacings provided.


        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""
        
        if input_field in self:
            np.seterr(divide='ignore')
            UVEL = self[input_field]
            dU_dy = np.zeros((UVEL.shape))

            for j in xrange(1,UVEL.shape[1]-1):
                dU_dy[:,j,1:] = (model_instance.grid['wet_mask_U'][:,j,1:]*
                                (model_instance.grid['wet_mask_U'][:,j+1,1:]*UVEL[:,j+1,1:] + 
                                (1 - model_instance.grid['wet_mask_U'][:,j+1,1:])*UVEL[:,j,1:] - 
                                # if j+1 point is not fluid, use j point as the starting 
                                # location for the derivative
                                (1 - model_instance.grid['wet_mask_U'][:,j-1,1:])*UVEL[:,j,1:] - 
                                model_instance.grid['wet_mask_U'][:,j-1,1:]*UVEL[:,j-1,1:])/(
                                model_instance.grid['wet_mask_U'][:,j-1,1:]*model_instance.grid['dyC'][j,:] + 
                                model_instance.grid['wet_mask_U'][:,j+1,1:]*model_instance.grid['dyC'][j+1,:]))

            j = 0
            dU_dy[:,j,1:] = (model_instance.grid['wet_mask_U'][:,j,1:]*(UVEL[:,j+1,1:] -
                UVEL[:,j,1:])/(model_instance.grid['dyC'][j+1,:]))
            j = UVEL.shape[1]-1
            dU_dy[:,j,1:] = (model_instance.grid['wet_mask_U'][:,j,1:]*(UVEL[:,j,1:] -
                UVEL[:,j-1,1:])/(model_instance.grid['dyC'][j,:]))

            self[output_field] = np.nan_to_num(dU_dy)
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
            
        return 
    
   
            
    def take_d_dz(self,model_instance,input_field = 'UVEL',output_field='dU_dz'):
        """! Take the z derivative of the field given on u-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""
        
        if input_field in self:
            np.seterr(divide='ignore')
            UVEL = self[input_field]
            d_dz = np.zeros((UVEL.shape))

            for k in xrange(1,UVEL.shape[0]-1):
                # model doesn't have overhangs, so only need this to work with fluid above and bathymetry below.
                d_dz[k,:,:] = np.nan_to_num(model_instance.grid['wet_mask_U'][k,:,:]*(UVEL[k-1,:,:]  -
                (1-model_instance.grid['wet_mask_U'][k+1,:,:])*UVEL[k,:,:]-
                model_instance.grid['wet_mask_U'][k+1,:,:]*UVEL[k+1,:,:])/(model_instance.grid['drC'][k] +
                model_instance.grid['wet_mask_U'][k+1,:,:]*model_instance.grid['drC'][k+1]))

            k = 0
            d_dz[k,:,:] = (UVEL[k,:,:] - UVEL[k+1,:,:])/(model_instance.grid['drC'][k+1])
            k = UVEL.shape[0]-1
            d_dz[k,:,:] = (UVEL[k-1,:,:] - UVEL[k,:,:])/(model_instance.grid['drC'][k])

            self[output_field] = d_dz
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return 

class Vpoint_field(MITgcm_Simulation):
    """! This is the class for all fields on meridional velocity points."""

    def __init__(self,model_instance,netcdf_filename,variable,time_level=-1,empty=False,field_name=None):
        """!Instantiate a field on the meridional velocity points."""
        if empty:
            pass
        else:
            self.load_field(model_instance,netcdf_filename,variable,time_level,field_name,grid_loc='V')

        return
    
    def load_field(self,model_instance, netcdf_filename,variable,time_level=-1,field_name=None,grid_loc='V'):
        """!Load a model field from NetCDF output. This function associates the field with the object it is called on.

        time_level can be an integer or an array of integers. If it's an array, then multiple time levels will be returned as a higher dimensional array.

        netcdf_filename can be a string with shell wildcards - they'll be expanded and all of the files loaded. BUT, this is intended as a way to take a quick look at the model. I would *strongly* recommend using something like "gluemncbig" to join the multiple tiles into one file before doing proper analysis."""
        if field_name == None:
            field_name = variable

        file_list = glob.glob(netcdf_filename)

        self[field_name] = self.load_from_file(model_instance,file_list,variable,time_level,grid_loc)
        return 


    def take_d_dx(self,model_instance,input_field = 'VVEL',output_field='dV_dx'):
        """!Take the x derivative of the field on v points using the spacings in model_instance.grid object.
        
        This function can be daisy-chained to get higher order derivatives.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            np.seterr(divide='ignore')
            VVEL = self[input_field]
            dV_dx = np.zeros((VVEL.shape))

            for i in xrange(1,VVEL.shape[2]-1):
                dV_dx[:,1:,i] = np.nan_to_num(model_instance.grid['wet_mask_V'][:,1:,i]*
                        (model_instance.grid['wet_mask_V'][:,1:,i+1]*VVEL[:,1:,i+1] + 
                        (1 - model_instance.grid['wet_mask_V'][:,1:,i+1])*VVEL[:,1:,i] - 
                        (1 - model_instance.grid['wet_mask_V'][:,1:,i-1])*VVEL[:,1:,i] - 
                        model_instance.grid['wet_mask_V'][:,1:,i-1]*VVEL[:,1:,i-1])/(
                        model_instance.grid['wet_mask_V'][:,1:,i-1]*model_instance.grid['dxC'][:,i] + 
                        model_instance.grid['wet_mask_V'][:,1:,i+1]*model_instance.grid['dxC'][:,i+1]))
            i = 0
            dV_dx[:,1:,i] = (VVEL[:,1:,i+1] - VVEL[:,1:,i])/(model_instance.grid['dxC'][:,i+1])
            i = VVEL.shape[2]-1
            dV_dx[:,1:,i] = (VVEL[:,1:,i] - VVEL[:,1:,i-1])/(model_instance.grid['dxC'][:,i])

            self[output_field] = dV_dx
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return

    def take_d_dy(self,model_instance,input_field = 'VVEL',output_field='dV_dy'):
        """! Take the y derivative of the field given on v-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            np.seterr(divide='ignore')
            VVEL = self[input_field]
            dV_dy = np.zeros((VVEL.shape))

            for j in xrange(1,VVEL.shape[1]-2):
                dV_dy[:,j,:] = np.nan_to_num(model_instance.grid['wet_mask_V'][:,j,:]*(
                        model_instance.grid['wet_mask_V'][:,j+1,:]*VVEL[:,j+1,:] + 
                        (1 - model_instance.grid['wet_mask_V'][:,j+1,:])*VVEL[:,j,:] - 
                        (1 - model_instance.grid['wet_mask_V'][:,j-1,:])*VVEL[:,j,:] - 
                        model_instance.grid['wet_mask_V'][:,j-1,:]*VVEL[:,j-1,:])/(
                        model_instance.grid['wet_mask_V'][:,j-1,:]*model_instance.grid['dyF'][j-1,:] + 
                        model_instance.grid['wet_mask_V'][:,j+1,:]*model_instance.grid['dyF'][j,:]))
            j = 0
            dV_dy[:,j,:] = (VVEL[:,j+1,:] - VVEL[:,j,:])/(model_instance.grid['dyF'][j,:])
            j = VVEL.shape[1]-1
            dV_dy[:,j,:] = (VVEL[:,j,:] - VVEL[:,j-1,:])/(model_instance.grid['dyF'][j-1,:])

            self[output_field] = dV_dy
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return
    
    
    def take_d_dz(self,model_instance,input_field = 'VVEL',output_field='dV_dz'):
        """! Take the z derivative of the field given on v-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            np.seterr(divide='ignore')
            VVEL = self[input_field]
            d_dz = np.zeros((VVEL.shape))

            for k in xrange(1,VVEL.shape[0]-1):
                # model doesn't have overhangs, so only need this to work with fluid above and bathymetry below.
                d_dz[k,:,:] = np.nan_to_num(model_instance.grid['wet_mask_V'][k,:,:]*(VVEL[k-1,:,:]  -
                    (1-model_instance.grid['wet_mask_V'][k+1,:,:])*VVEL[k,:,:]-
                    model_instance.grid['wet_mask_V'][k+1,:,:]*VVEL[k+1,:,:])/(model_instance.grid['drC'][k] +
                    model_instance.grid['wet_mask_V'][k+1,:,:]*model_instance.grid['drC'][k+1]))

            k = 0
            d_dz[k,:,:] = (VVEL[k,:,:] - VVEL[k+1,:,:])/(model_instance.grid['drC'][k+1])
            k = VVEL.shape[0]-1
            d_dz[k,:,:] = (VVEL[k-1,:,:] - VVEL[k,:,:])/(model_instance.grid['drC'][k])

            self[output_field] = d_dz
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return
    
class Wpoint_field(MITgcm_Simulation):
    """! This is the class for all fields on vertical velocity points."""

    def __init__(self,model_instance,netcdf_filename,variable,time_level=-1,empty=False,field_name=None):
        if empty:
            pass
        else:
            self.load_field(model_instance,netcdf_filename,variable,time_level,field_name,grid_loc='W')

        return

    def load_field(self,model_instance, netcdf_filename,variable,time_level=-1,field_name=None,grid_loc='W'):
        """!Load a model field from NetCDF output. This function associates the field with the object it is called on.

        time_level can be an integer or an array of integers. If it's an array, then multiple time levels will be returned as a higher dimensional array."""
        if field_name == None:
            field_name = variable


        file_list = glob.glob(netcdf_filename)


        loaded_field = self.load_from_file(model_instance,file_list,variable,time_level,grid_loc)


        if len(loaded_field.shape) == 4:
                self[field_name] = np.append(loaded_field,np.zeros((loaded_field.shape[0],1,loaded_field.shape[-2],loaded_field.shape[-1])),axis=1)
        else:
                self[field_name] = np.append(loaded_field,np.zeros((1,loaded_field.shape[-2],loaded_field.shape[-1])),axis=0)


        return 


    def take_d_dx(self,model_instance,input_field = 'WVEL',output_field='dW_dx'):
        """!Take the x derivative of the field on w points, using spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            np.seterr(divide='ignore')
            WVEL = self[input_field]
            d_dx = np.zeros((WVEL.shape))

            for i in xrange(1,WVEL.shape[2]-1):
                d_dx[:,:,i] = np.nan_to_num(model_instance.grid['wet_mask_W'][:,:,i]*
                                (model_instance.grid['wet_mask_W'][:,:,i+1]*WVEL[:,:,i+1] + 
                                (1 - model_instance.grid['wet_mask_W'][:,:,i+1])*WVEL[:,:,i] - 
                                (1 - model_instance.grid['wet_mask_W'][:,:,i-1])*WVEL[:,:,i] - 
                                model_instance.grid['wet_mask_W'][:,:,i-1]*WVEL[:,:,i-1])/(
                                model_instance.grid['wet_mask_W'][:,:,i-1]*model_instance.grid['dxC'][:,i] + 
                                model_instance.grid['wet_mask_W'][:,:,i+1]*model_instance.grid['dxC'][:,i+1]))
            i = 0
            d_dx[:,:,i] = (WVEL[:,:,i+1] - WVEL[:,:,i])/(model_instance.grid['dxC'][:,i+1])
            i = WVEL.shape[2]-1
            d_dx[:,:,i] = (WVEL[:,:,i] - WVEL[:,:,i-1])/(model_instance.grid['dxC'][:,i])

            self[output_field] = d_dx

        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return  


    def take_d_dy(self,model_instance,input_field = 'WVEL',output_field='dW_dy'):
        """!Take the y derivative of the field on w points, using spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            np.seterr(divide='ignore')
            WVEL = self[input_field]
            dW_dy = np.zeros((WVEL.shape))

            for j in xrange(1,WVEL.shape[2]-1):
                dW_dy[:,j,:] = np.nan_to_num(model_instance.grid['wet_mask_W'][:,j,:]*
                                    (model_instance.grid['wet_mask_W'][:,j+1,:]*WVEL[:,j+1,:] + 
                                    (1 - model_instance.grid['wet_mask_W'][:,j+1,:])*WVEL[:,j,:] - 
                                    (1 - model_instance.grid['wet_mask_W'][:,j-1,:])*WVEL[:,j,:] - 
                                    model_instance.grid['wet_mask_W'][:,j-1,:]*WVEL[:,j-1,:])/(
                                    model_instance.grid['wet_mask_W'][:,j-1,:]*model_instance.grid['dyC'][j,:] + 
                                    model_instance.grid['wet_mask_W'][:,j+1,:]*model_instance.grid['dyC'][j+1,:]))
            j = 0
            dW_dy[:,j,:] = (WVEL[:,j+1,:] - WVEL[:,j,:])/(model_instance.grid['dyC'][j+1,:])
            j = WVEL.shape[1]-1
            dW_dy[:,j,:] = (WVEL[:,j,:] - WVEL[:,j-1,:])/(model_instance.grid['dyC'][j,:])

            self[output_field] = dW_dy
       
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return 

    def take_d_dz(self,model_instance,input_field = 'WVEL',output_field='dW_dz'):
        """! Take the z derivative of the field given on w-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""  
        
        if input_field in self:
            np.seterr(divide='ignore')
            WVEL = self[input_field]
            dWVEL_dz = np.zeros((WVEL.shape))

            for k in xrange(1,WVEL.shape[0]-2):
                dWVEL_dz[k,:,:] = np.nan_to_num(model_instance.grid['wet_mask_TH'][k,:,:]*(WVEL[k-1,:,:] -
                            (1-model_instance.grid['wet_mask_TH'][k+1,:,:])*WVEL[k,:,:]-
                            model_instance.grid['wet_mask_TH'][k+1,:,:]*WVEL[k+1,:,:])/
                            (model_instance.grid['drF'][k-1]+
                            model_instance.grid['wet_mask_TH'][k+1,:,:]*model_instance.grid['drF'][k]))

            k = 0
            dWVEL_dz[k,:,:] = (WVEL[k,:,:] - WVEL[k+1,:,:])/(model_instance.grid['drF'][k])
            k = WVEL.shape[0]-2
            dWVEL_dz[k,:,:] = (WVEL[k-1,:,:] - WVEL[k,:,:])/(model_instance.grid['drF'][k-1])        
            k = WVEL.shape[0]-1
            dWVEL_dz[k,:,:] = (WVEL[k-1,:,:] - WVEL[k,:,:])/(model_instance.grid['drF'][k-1])

            self[output_field] = dWVEL_dz
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return 



class Tracerpoint_field(MITgcm_Simulation):  
    """!This is the base class for all model fields on the tracer points. It includes definitions for taking derivatives."""
    def __init__(self,model_instance,netcdf_filename,variable,time_level=-1,empty=False,field_name=None):
        if empty:
            pass
        else:
            self.load_field(model_instance,netcdf_filename,variable,time_level,field_name,grid_loc='T')
        return


    def load_field(self,model_instance, netcdf_filename,variable,time_level=-1,field_name=None,grid_loc='T'):
        """!Load a model field from NetCDF output. This function associates the field with the object it is called on.

        time_level can be an integer or an array of integers. If it's an array, then multiple time levels will be returned as a higher dimensional array.

        netcdf_filename can be a string with shell wildcards - they'll be expanded and all of the files loaded. BUT, this is intended as a way to take a quick look at the model. I would *strongly* recommend using something like "gluemncbig" to join the multiple tiles into one file before doing proper analysis."""
        if field_name == None:
            field_name = variable

        file_list = glob.glob(netcdf_filename)

        self[field_name] = self.load_from_file(model_instance,file_list,variable,time_level,grid_loc)
        return 


    def take_d_dx(self,model_instance,input_field = 'RHO',output_field='dRHO_dx'):
        """!Take the x derivative of the field on tracer points, using spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            np.seterr(divide='ignore')
            rho = self[input_field]

            d_dx = (self.numerics_take_d_dx(rho[:],model_instance.grid['wet_mask_TH'][:],
                                            model_instance.grid['dxC'][:],))
            self[output_field] = np.nan_to_num(d_dx)
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return 
    
    def numerics_take_d_dx(self,rho,wet_mask_TH,dxC):

        d_dx = np.zeros((rho.shape))

        i = 0
        d_dx[:,:,i] = (rho[:,:,i+1] - rho[:,:,i])/(dxC[:,i+1])
        i = rho.shape[2]-1
        d_dx[:,:,i] = (rho[:,:,i] - rho[:,:,i-1])/(dxC[:,i])

        for i in xrange(1,rho.shape[2]-1):
            d_dx[:,:,i] = (wet_mask_TH[:,:,i]*
                            (wet_mask_TH[:,:,i+1]*rho[:,:,i+1] + 
                            (1 - wet_mask_TH[:,:,i+1])*rho[:,:,i] - 
                            (1 - wet_mask_TH[:,:,i-1])*rho[:,:,i] - 
                            wet_mask_TH[:,:,i-1]*rho[:,:,i-1])/(
                            wet_mask_TH[:,:,i-1]*dxC[:,i] + 
                            wet_mask_TH[:,:,i+1]*dxC[:,i+1]))
        return d_dx


    def take_d_dy(self,model_instance,input_field = 'RHO',output_field='dRHO_dy'):
        """!Take the y derivative of the field on tracer points, using spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
        #np.seterr(divide='ignore')

            self[output_field] = np.nan_to_num(self.numerics_take_d_dy(self[input_field][:],model_instance.grid['wet_mask_TH'][:],
                                                 model_instance.grid['dyC'][:]))
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return  
        
    def numerics_take_d_dy(self,rho,wet_mask_TH,dyC):
        """!The numerical bit of taking the y derivative. This has been separated out so that it can be accelerated with numba, but that isn't working yet."""

        d_dy = np.zeros((rho.shape))
            
        j = 0
        d_dy[:,j,:] = (rho[:,j+1,:] - rho[:,j,:])/(dyC[j+1,:])
        j = rho.shape[1]-1
        d_dy[:,j,:] = (rho[:,j,:] - rho[:,j-1,:])/(dyC[j,:])

        for j in xrange(1,rho.shape[1]-1):
           d_dy[:,j,:] = (wet_mask_TH[:,j,:]*
                            (wet_mask_TH[:,j+1,:]*rho[:,j+1,:] + 
                            (1 - wet_mask_TH[:,j+1,:])*rho[:,j,:] - 
                            (1 - wet_mask_TH[:,j-1,:])*rho[:,j,:] - 
                            wet_mask_TH[:,j-1,:]*rho[:,j-1,:])/(
                            wet_mask_TH[:,j-1,:]*dyC[j,:] + 
                            wet_mask_TH[:,j+1,:]*dyC[j+1,:]))
        return d_dy

    
    def take_d_dz(self,model_instance,input_field = 'RHO',output_field='dRHO_dz'):
        """! Take the z derivative of the field given on tracer-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            np.seterr(divide='ignore')
            rho = self[input_field]
            d_dz = np.zeros((rho.shape))

            for k in xrange(1,rho.shape[0]-1):
                # model doesn't have overhangs, so only need this to work with fluid above and bathymetry below.
                d_dz[k,:,:] = (model_instance.grid['wet_mask_TH'][k,:,:]*(rho[k-1,:,:]  -
                                    (1-model_instance.grid['wet_mask_TH'][k+1,:,:])*rho[k,:,:]-
                                    model_instance.grid['wet_mask_TH'][k+1,:,:]*rho[k+1,:,:])/(model_instance.grid['drC'][k] +
                                    model_instance.grid['wet_mask_TH'][k+1,:,:]*model_instance.grid['drC'][k+1]))

            k = 0
            d_dz[k,:,:] = (rho[k,:,:] - rho[k+1,:,:])/(model_instance.grid['drC'][k+1])
            k = rho.shape[0]-1
            d_dz[k,:,:] = (rho[k-1,:,:] - rho[k,:,:])/(model_instance.grid['drC'][k])

            self[output_field] = np.nan_to_num(d_dz)
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return 


    
class Vorticitypoint_field(MITgcm_Simulation):  
    """!A class for fields on vorticity points."""
    def __init__(self,model_instance,netcdf_filename,variable,time_level=-1,empty=False,field_name=None):
        if empty:
            pass
        else:
            self.load_field(model_instance,netcdf_filename,variable,time_level,field_name,grid_loc='Zeta')

        return

    def load_field(self,model_instance, netcdf_filename,variable,time_level=-1,field_name=None,grid_loc='Zeta'):
        """!Load a model field from NetCDF output. This function associates the field with the object it is called on.

        time_level can be an integer or an array of integers. If it's an array, then multiple time levels will be returned as a higher dimensional array.

        netcdf_filename can be a string with shell wildcards - they'll be expanded and all of the files loaded. BUT, this is intended as a way to take a quick look at the model. I would *strongly* recommend using something like "gluemncbig" to join the multiple tiles into one file before doing proper analysis."""
        if field_name == None:
            field_name = variable

        file_list = glob.glob(netcdf_filename)

        self[field_name] = self.load_from_file(model_instance,file_list,variable,time_level,grid_loc)
        return 


    def take_d_dx(self,model_instance,input_field = 'momVort3',output_field='dmomVort3_dx'):
        """! Take the x derivative of the field given on vorticity-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            np.seterr(divide='ignore')
            momVort3 = self[input_field]
            dmomVort3_dx = np.zeros((momVort3.shape))

            for i in xrange(1,momVort3.shape[2]-2):
                dmomVort3_dx[:,:,i] = np.nan_to_num(model_instance.grid['wet_mask_U'][:,:,i]*
                        (model_instance.grid['wet_mask_U'][:,:,i+1]*momVort3[:,:,i+1] + 
                        (1 - model_instance.grid['wet_mask_U'][:,:,i+1])*momVort3[:,:,i] - 
                        (1 - model_instance.grid['wet_mask_U'][:,:,i-1])*momVort3[:,:,i] - 
                        model_instance.grid['wet_mask_U'][:,:,i-1]*momVort3[:,:,i-1])/(
                        model_instance.grid['wet_mask_U'][:,:,i-1]*model_instance.grid['dxF'][:,i-1] + 
                        model_instance.grid['wet_mask_U'][:,:,i+1]*model_instance.grid['dxF'][:,i]))
            i = 0
            dmomVort3_dx[:,:,i] = (momVort3[:,:,i+1] - momVort3[:,:,i])/(model_instance.grid['dxF'][:,i])
            i = momVort3.shape[2]-1
            dmomVort3_dx[:,:,i] = (momVort3[:,:,i] - momVort3[:,:,i-1])/(model_instance.grid['dxF'][:,i-1])

            self[output_field] = dmomVort3_dx
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
            
        return 

    def take_d_dy(self,model_instance,input_field = 'momVort3',output_field='dmomVort3_dy'):
        """! Take the y derivative of the field given on vorticity-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            np.seterr(divide='ignore')
            momVort3 = self[input_field]
            dmomVort3_dy = np.zeros((momVort3.shape))

            for j in xrange(1,momVort3.shape[1]-2):
                dmomVort3_dy[:,j,:] = np.nan_to_num(model_instance.grid['wet_mask_V'][:,j,:]*(
                        model_instance.grid['wet_mask_V'][:,j+1,:]*momVort3[:,j+1,:] + 
                        (1 - model_instance.grid['wet_mask_V'][:,j+1,:])*momVort3[:,j,:] - 
                        (1 - model_instance.grid['wet_mask_V'][:,j-1,:])*momVort3[:,j,:] - 
                        model_instance.grid['wet_mask_V'][:,j-1,:]*momVort3[:,j-1,:])/(
                        model_instance.grid['wet_mask_V'][:,j-1,:]*model_instance.grid['dyF'][j-1,:] + 
                        model_instance.grid['wet_mask_V'][:,j+1,:]*model_instance.grid['dyF'][j,:]))
            j = 0
            dmomVort3_dy[:,j,:] = (momVort3[:,j+1,:] - momVort3[:,j,:])/(model_instance.grid['dyF'][j,:])
            j = momVort3.shape[1]-1
            dmomVort3_dy[:,j,:] = (momVort3[:,j,:] - momVort3[:,j-1,:])/(model_instance.grid['dyF'][j-1,:])

            self[output_field] = dmomVort3_dy
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return

    def take_d_dz(self,model_instance,input_field = 'momVort3',output_field='dmomVort3_dz'):
        """! Take the z derivative of the field given on vorticity-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            np.seterr(divide='ignore')
            momVort3 = self[input_field]
            d_dz = np.zeros((momVort3.shape))

            for k in xrange(1,momVort3.shape[0]-1):
            # model doesn't have overhangs, so only need this to work with fluid above and bathymetry below.
                d_dz[k,:,:] = np.nan_to_num(model_instance.grid['wet_mask_TH'][k,:,:]*(momVort3[k-1,:,:]  -
            		(1-model_instance.grid['wet_mask_TH'][k+1,:,:])*momVort3[k,:,:]-
            		model_instance.grid['wet_mask_TH'][k+1,:,:]*momVort3[k+1,:,:])/(model_instance.grid['drC'][k] +
            		model_instance.grid['wet_mask_TH'][k+1,:,:]*model_instance.grid['drC'][k+1]))

            k = 0
            d_dz[k,:,:] = (momVort3[k,:,:] - momVort3[k+1,:,:])/(model_instance.grid['drC'][k+1])
            k = momVort3.shape[0]-1
            d_dz[k,:,:] = (momVort3[k-1,:,:] - momVort3[k,:,:])/(model_instance.grid['drC'][k])

            self[output_field] = d_dz
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return 


class Grid(MITgcm_Simulation):
    """!This defines the class for the grid object. 

    Currently requires a single grid file.

    This object holds all the information about the grid on which the simulation was run. It also holds masks for selecting only the boundary values of fields on the tracer points. 
    """

    def __init__(self, grid_netcdf_filename):
        """!Define a single object that has all of the grid variables tucked away in it. 
        Each of the variables pulled directly from the netcdf file still has the 
        original description attached to it. The 2D and 3D arrays do not.

        Variables imported are:
        * rAw
        * rAs
        * rA
        * HFacW
        * HFacS
        * HFacC
        * X
        * Xp1
        * dxF
        * dxC
        * dxV
        * Y
        * Yp1
        * dyU
        * dyC
        * dyF
        * Z
        * Zl
        * Zu
        * drC
        * drF
        * fCoriG
        * fCori

        Variables computed and stored are:
        * (Z_y,Y_z)
        * (X_y,Y_x)
        * (Z_x,X_z) 
        * (Z_3d,Y_3d,X_3d) 

        * (DZF,DYF, DXF): a 3d array of dxF,dyF adn dzF 

        * wet_mask_V : a 3d array that is one if the point is in the fluid, zero otherwise.
        * wet_mask_U 
        * wet_mask_TH 
        * wet_mask_W 

        * cell_volume

        **These boundary masks can be computed with the compute_boundary_masks function**
        * west_mask : a 3d array that is one if the point has a boundary to the west
        * east_mask
        * south_mask
        * north_mask
        * bottom_mask

        -----
        Notes: uses glob to expand wildcards in the file name. BUT, it will only use the first file that matches.

        """
        grid_netcdf_file_list = glob.glob(grid_netcdf_filename)

        #print grid_netcdf_file_list
        grid_netcdf_file = netCDF4.Dataset(grid_netcdf_file_list[0])

        self['rAw'] = grid_netcdf_file.variables['rAw']
        self['rAs'] = grid_netcdf_file.variables['rAs']
        self['rA'] = grid_netcdf_file.variables['rA']
        self['HFacW'] = grid_netcdf_file.variables['HFacW']
        self['HFacS'] = grid_netcdf_file.variables['HFacS']
        self['HFacC'] = grid_netcdf_file.variables['HFacC']
        self['X']= grid_netcdf_file.variables['X']
        self['Xp1'] = grid_netcdf_file.variables['Xp1']
        self['dxF'] = grid_netcdf_file.variables['dxF']
        self['dxC'] = grid_netcdf_file.variables['dxC']
        self['dxV'] = grid_netcdf_file.variables['dxV']
        self['Y'] = grid_netcdf_file.variables['Y']
        self['Yp1'] = grid_netcdf_file.variables['Yp1']
        self['dyU'] = grid_netcdf_file.variables['dyU']
        self['dyC'] = grid_netcdf_file.variables['dyC']
        self['dyF'] = grid_netcdf_file.variables['dyF']
        self['Z'] = grid_netcdf_file.variables['Z']
        self['Zl'] = grid_netcdf_file.variables['Zl']
        self['Zu'] = grid_netcdf_file.variables['Zu']
        self['drC'] = grid_netcdf_file.variables['drC']
        self['drF'] = grid_netcdf_file.variables['drF']
        self['fCoriG'] = grid_netcdf_file.variables['fCoriG']
        self['fCori'] = grid_netcdf_file.variables['fCori']

        (self['Z_y'],self['Y_z']) = np.meshgrid(self['Z'][:],self['Y'][:],indexing='ij')
        (self['X_y'],self['Y_x']) = np.meshgrid(self['X'],self['Y'][:],indexing='ij')
        (self['Z_x'],self['X_z']) = np.meshgrid(self['Z'],self['X'][:],indexing='ij')
        (self['Z_3d'],self['Y_3d'],self['X_3d']) = np.meshgrid(self['Z'][:],self['Y'][:],self['X'][:],indexing='ij')

        (self['DZF'],self['DYF'], self['DXF']) = np.meshgrid(self['drF'][:],self['dyF'][0,:],self['dxF'][:,0],indexing='ij')

        self['wet_mask_V'] = np.ones((np.shape(self['HFacS'][:])))
        self['wet_mask_V'][self['HFacS'][:] == 0.] = 0.
        self['wet_mask_U'] = np.ones((np.shape(self['HFacW'][:])))
        self['wet_mask_U'][self['HFacW'][:] == 0.] = 0.
        self['wet_mask_TH'] = np.ones((np.shape(self['HFacC'][:])))
        self['wet_mask_TH'][self['HFacC'][:] == 0.] = 0.
        self['wet_mask_W'] = np.append(self['wet_mask_TH'],np.ones((1,len(self['Y'][:]),len(self['X'][:]))),axis=0)

        self['cell_volume'] = copy.deepcopy(self['dxF'][:]*self['dyF'][:]*
                                    self['drF'][:].reshape((len(self['drF'][:]),1,1)))

        #self.compute_masks(self['wet_mask_TH'][:])
        # this isn't automatically done, since it's expensive, and only needed when computing boundary fluxes.
                        
        return
        
    @numba.jit        
    def compute_boundary_masks(self):
        """!This function does the computationally heavy job of looping through each dimension and creating masks that are one if the boundary is next to the grid point in the specified direction. This function is accelerated by numba, making it about 100 times faster."""

        wet_mask_TH = self['wet_mask_TH'][:]

        west_mask = np.zeros((wet_mask_TH.shape))
        east_mask = np.zeros((wet_mask_TH.shape))
        south_mask = np.zeros((wet_mask_TH.shape))
        north_mask = np.zeros((wet_mask_TH.shape))
        bottom_mask = np.zeros((wet_mask_TH.shape))
        
        # Find the fluxes through the boundary of the domain
        for k in xrange(0,wet_mask_TH.shape[0]):
            for j in xrange(0,wet_mask_TH.shape[1]):
                for i in xrange(0,wet_mask_TH.shape[2]):
                    # find points with boundary to the west. In the simplest shelf configuration this is the only tricky boundary to find.
                    if wet_mask_TH[k,j,i] - wet_mask_TH[k,j,i-1] == 1:
                        west_mask[k,j,i] = 1


                    # find the eastern boundary points. Negative sign is to be consistent about fluxes into the domain.
                    if wet_mask_TH[k,j,i-1] - wet_mask_TH[k,j,i] == 1:
                        east_mask[k,j,i-1] = 1


                    # find the southern boundary points
                    if wet_mask_TH[k,j,i] - wet_mask_TH[k,j-1,i] == 1:
                        south_mask[k,j,i] = 1


                    # find the northern boundary points
                    if wet_mask_TH[k,j-1,i] - wet_mask_TH[k,j,i] == 1:
                        north_mask[k,j,i-1] = 1


                    # Fluxes through the bottom
                    if wet_mask_TH[k-1,j,i] - wet_mask_TH[k,j,i] == 1:
                        bottom_mask[k-1,j,i] = 1
        
        self['west_mask'] = west_mask
        self['east_mask'] = east_mask
        self['south_mask'] = south_mask
        self['north_mask'] = north_mask
        self['bottom_mask'] = bottom_mask
        return
        
        

class Density(Tracerpoint_field):
    """!A tracer point field that contains methods for density fields. Only linear equation of state with temperature variations is supported at the moment.

    The linear equation of state is given by

    \f[
    \rho = \rho_{nil} (-\alpha_{T} (\theta - \theta_{0}) + \beta_{S} (S - S_{0})) + \rho_{nil}
    \f]
    where \f$ \rho_{nil} \f$ is the reference density, \f$ \alpha_{T} \f$ is the thermal expansion coefficient, \f$ \beta_{S} \f$ is the haline contraction coefficient, \f$ \theta \f$ and \f$ \beta \f$ are the temperature and salinity fields, and subscript zeros denote reference values.
    """

    def __init__(self,model_instance,Talpha=2e-4,Sbeta=0,RhoNil=1035,cp=4000,
                    temp_field='THETA',salt_field='S',density_field='RHO',
                    Tref=20,Sref=30,empty=False):
        if empty:
            pass
        else:
            self['cp'] = cp
            self['Talpha'] = Talpha
            self['Sbeta'] = Sbeta
            self['RhoNil'] = RhoNil
            self.calculate_density(model_instance,Talpha,Sbeta,RhoNil,cp,
                                    temp_field,salt_field,density_field,Tref,Sref)


    def calculate_density(self,model_instance,Talpha=2e-4,Sbeta=0,RhoNil=1035,cp=4000,
          temp_field='THETA',salt_field='S',density_field='RHO',Tref=20,Sref=30):
        """!Cacluate Density field given temperature and salinity fields, using the linear equation of state."""
        if model_instance['EOS_type'] == 'linear':
            if Sbeta == 0:
                self[density_field] = (RhoNil*( -Talpha*(model_instance.temperature[temp_field] - Tref)) 
                + RhoNil)
            else:
                self[density_field] = (RhoNil*( Sbeta*(model_instance.salinity[salt_field] - Sref) - Talpha*(model_instance.temperature[temp_field] - Tref)) 
                + RhoNil)
                print 'Warning: Linear equation of state with salinity is currently untested. Proceed with caution.'
        else:
          raise ValueError('Only linear EOS supported at the moment. Sorry.')

    def calculate_TotRhoTend(self,model_instance):
        """!Calculate time rate of change of the Density field from the temperature tendency and the linear equation of state. Returned as 'TotRhoTend' associated with the density object.

        Differentiating the linear equation of state with respect to temperature, and assuming \f$ \beta_{S} \f$ equals zero, gives
        \f[
        \frac{\partial \rho}{\partial t} = - \rho_{nil} \alpha_{T} \frac{\partial \theta}{\partial t}
        \f]
        """
        if model_instance['EOS_type'] == 'linear':
            if self['Sbeta'] == 0:
                self['TotRhoTend'] = (-self['RhoNil']*self['Talpha']*model_instance.temperature['TOTTTEND'])
            else:
                raise ValueError('This operator only supports temperature variations at the moment. Sorry.') 
        else:    
            raise ValueError('Only liner EOS supported at the moment.')

            
    
class Bernoulli(Tracerpoint_field):
    """!The Bernoulli field, evaluated from velocity, Pressure and Density.
    \f[
    BP = \frac{P}{\rho_{0}} +  g z + \frac{u^{2} + v^{2}}{2}
    \f]
    """
    def __init__(self,model_instance,density_field='RHO',UVEL_field='UVEL',VVEL_field='VVEL'):
        BP_pressure = ((model_instance.pressure['P'][:,:,:]/model_instance.density['RhoNil'] + 
                 model_instance.grid['Z'][:].reshape((len(model_instance.grid['Z'][:]),1,1))*model_instance['g']))

        BP_u = (model_instance.zonal_velocity[UVEL_field][:,:,1:]*model_instance.zonal_velocity[UVEL_field][:,:,1:] + 
                 model_instance.zonal_velocity[UVEL_field][:,:,:-1]*model_instance.zonal_velocity[UVEL_field][:,:,:-1])/2

        BP_v = (model_instance.meridional_velocity[VVEL_field][:,1:,:]*model_instance.meridional_velocity[VVEL_field][:,1:,:] + 
                 model_instance.meridional_velocity[VVEL_field][:,:-1,:]*model_instance.meridional_velocity[VVEL_field][:,:-1,:])/2

        self['BP'] = BP_pressure + (BP_u + BP_v)/2
        # decomposition into three components suggested by Craig.
        
class Pressure(Tracerpoint_field):
    """!Calculates the pressure field from the density field using the hydrostatic approximation."""
    def __init__(self,model_instance,density_field='RHO',ETAN_field='ETAN'):

        # derive the hydrostatic pressure
        delta_P = np.zeros((np.shape(model_instance.density[density_field])))
        delta_P[:,:,:] = (model_instance['g']*model_instance.density[density_field][:,:,:]*
                            model_instance.grid['drF'][:].reshape(len(model_instance.grid['drF'][:]),1,1))
        
    
        # add free surface contribution
        delta_P[0,:,:] = (delta_P[0,:,:] + 
                          model_instance.free_surface[ETAN_field]*model_instance['g']*model_instance.density[density_field][0,:,:])
    
        self['delta_P'] = delta_P
        self['P'] = np.cumsum(delta_P,0)
        
        return
    

        
class Potential_vorticity(Tracerpoint_field):
    """!Evaluate the potential vorticity on the tracer points."""
    def __init__(self,model_instance,density_field='RhoNil',density_deriv_field='dRHO_dz',vort_field='omega_a'):
        self['Q'] = -model_instance.vorticity[vort_field]*model_instance.density[density_deriv_field]/model_instance.density[density_field]

   