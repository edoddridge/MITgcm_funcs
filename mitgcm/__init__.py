import numpy as np
import netCDF4
import copy
import os

    
class Simulation(dict):
    
    def __init__(self,output_dir,grid_netcdf_filename,EOS_type='linear',g=9.8):
        """Instantiate an MITgcm model instance."""
        
        os.chdir(output_dir)
        self['output_dir'] = output_dir

        self.grid = Grid(grid_netcdf_filename)
        

        self['g'] = g
        self['EOS_type'] = EOS_type
        if EOS_type != 'linear':
            raise ValueError('Only linear equation of state is currently supported')
            
    def load_field(self,netcdf_filename,variable,time_level):
        """ Load a model field from NetCDF output. This function associates the field with the object it is called on.

	time_level can be an integer or an array of integers. If it's an array, then multiple time levels will be returned as a higher dimensional array."""
        
        netcdf_file = netCDF4.MFDataset(netcdf_filename)
        loaded_field = netcdf_file.variables[variable][time_level,:,:,:]
        netcdf_file.close()
        
        self[variable] = loaded_field
        return
       
    def __add__(self,other):
        me = copy.deepcopy(self)
        for key, value in me.__dict__.iteritems():
                for key1, value1 in other.__dict__.iteritems():
                        if key == key1:
                            setattr(me, key, value + value1)
        return me

    def __div__(self,other):
        me = copy.deepcopy(self)
        for key, value in me.__dict__.iteritems():
                setattr(me, key, value/float(other))
        return me

    def __mul__(self, other):
        me = copy.deepcopy(self)
        for key, value in me.__dict__.iteritems():
                setattr(me, key, value * float(other))
        return me

    def __rmul__(self, other):
        me = copy.deepcopy(self)
        for key, value in me.__dict__.iteritems():
                setattr(me, key, value * float(other))
        return me

class Upoint_field(Simulation):
    
    def __init__(self,netcdf_filename,variable,time_level):
        netcdf_file = netCDF4.MFDataset(netcdf_filename)
        loaded_field = netcdf_file.variables[variable][time_level,:,:,:]
        netcdf_file.close()
        
        self[variable] = loaded_field
        return
    
    ### Derivatives of model fields    
    def take_d_dx(self,model_instance,input_field = 'UVEL',output_field='dU_dx'):
        """ Take the x derivative of the field given on u-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            UVEL = self[input_field]
            dU_dx = np.zeros((UVEL.shape))

            for i in xrange(1,UVEL.shape[2]-2):
                dU_dx[:,:,i] = np.nan_to_num(model_instance.grid.wet_mask_U[:,:,i]*
                        (model_instance.grid.wet_mask_U[:,:,i+1]*UVEL[:,:,i+1] + 
                        (1 - model_instance.grid.wet_mask_U[:,:,i+1])*UVEL[:,:,i] - 
                        (1 - model_instance.grid.wet_mask_U[:,:,i-1])*UVEL[:,:,i] - 
                        model_instance.grid.wet_mask_U[:,:,i-1]*UVEL[:,:,i-1])/(
                        model_instance.grid.wet_mask_U[:,:,i-1]*model_instance.grid.dxF[:,i-1] + 
                        model_instance.grid.wet_mask_U[:,:,i+1]*model_instance.grid.dxF[:,i]))
            i = 1
            dU_dx[:,:,i] = (UVEL[:,:,i+1] - UVEL[:,:,i])/(model_instance.grid.dxF[:,i])
            i = UVEL.shape[2]-1
            dU_dx[:,:,i] = (UVEL[:,:,i] - UVEL[:,:,i-1])/(model_instance.grid.dxF[:,i-1])

            self[output_field] = dU_dx
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
            
        return 

    
    

    def take_d_dy(self,model_instance,input_field = 'UVEL',output_field='dU_dy'):
        """ Take the y derivative of the field on u points, using the spacings provided.


        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""
        
        if input_field in self:
            UVEL = self[input_field]
            dU_dy = np.zeros((UVEL.shape))

            for j in xrange(1,UVEL.shape[1]-1):
                dU_dy[:,j,1:] = np.nan_to_num(model_instance.grid.wet_mask_U[:,j,1:]*
                                (model_instance.grid.wet_mask_U[:,j+1,1:]*UVEL[:,j+1,1:] + 
                                (1 - model_instance.grid.wet_mask_U[:,j+1,1:])*UVEL[:,j,1:] - 
                                # if j+1 point is not fluid, use j point as the starting 
                                # location for the derivative
                                (1 - model_instance.grid.wet_mask_U[:,j-1,1:])*UVEL[:,j,1:] - 
                                model_instance.grid.wet_mask_U[:,j-1,1:]*UVEL[:,j-1,1:])/(
                                model_instance.grid.wet_mask_U[:,j-1,1:]*model_instance.grid.dyC[j,:] + 
                                model_instance.grid.wet_mask_U[:,j+1,1:]*model_instance.grid.dyC[j+1,:]))

            j = 1
            dU_dy[:,j,1:] = (UVEL[:,j+1,1:] - UVEL[:,j,1:])/(model_instance.grid.dyC[j+1,:])
            j = UVEL.shape[1]-1
            dU_dy[:,j,1:] = (UVEL[:,j,1:] - UVEL[:,j-1,1:])/(model_instance.grid.dyC[j,:])

            self[output_field] = dU_dy
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
            
        return 
    
   
            
    def take_d_dz(self,model_instance,input_field = 'UVEL',output_field='dU_dz'):
        """ Take the z derivative of the field given on u-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""
        
        if input_field in self:
            UVEL = self[input_field]
            d_dz = np.zeros((UVEL.shape))

            for k in xrange(1,UVEL.shape[0]-1):
                # model doesn't have overhangs, so only need this to work with fluid above and bathymetry below.
                d_dz[k,:,:] = np.nan_to_num(model_instance.grid.wet_mask_U[k,:,:]*(UVEL[k-1,:,:]  -
                (1-model_instance.grid.wet_mask_U[k+1,:,:])*UVEL[k,:,:]-
                model_instance.grid.wet_mask_U[k+1,:,:]*UVEL[k+1,:,:])/(model_instance.grid.drC[k] +
                model_instance.grid.wet_mask_U[k+1,:,:]*model_instance.grid.drC[k+1]))

                k = 0
                d_dz[k,:,:] = (UVEL[k,:,:] - UVEL[k+1,:,:])/(model_instance.grid.drC[k+1])
                k = UVEL.shape[0]-1
                d_dz[k,:,:] = (UVEL[k-1,:,:] - UVEL[k,:,:])/(model_instance.grid.drC[k])

            self[output_field] = d_dz
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return 

class Vpoint_field(Simulation):
    
    def __init__(self,netcdf_filename,variable,time_level):
        netcdf_file = netCDF4.MFDataset(netcdf_filename)
        loaded_field = netcdf_file.variables[variable][time_level,:,:,:]
        netcdf_file.close()
        
        self[variable] = loaded_field
        return
    
    
    def take_d_dx(self,model_instance,input_field = 'VVEL',output_field='dV_dx'):
        """Take the x derivative of the field on v points using the spacings in model_instance.grid object.
        
        This function can be daisy-chained to get higher order derivatives.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            VVEL = self[input_field]
            dV_dx = np.zeros((VVEL.shape))

            for i in xrange(1,VVEL.shape[2]-1):
                dV_dx[:,1:,i] = np.nan_to_num(model_instance.grid.wet_mask_V[:,1:,i]*
                        (model_instance.grid.wet_mask_V[:,1:,i+1]*VVEL[:,1:,i+1] + 
                        (1 - model_instance.grid.wet_mask_V[:,1:,i+1])*VVEL[:,1:,i] - 
                        (1 - model_instance.grid.wet_mask_V[:,1:,i-1])*VVEL[:,1:,i] - 
                        model_instance.grid.wet_mask_V[:,1:,i-1]*VVEL[:,1:,i-1])/(
                        model_instance.grid.wet_mask_V[:,1:,i-1]*model_instance.grid.dxC[:,i] + 
                        model_instance.grid.wet_mask_V[:,1:,i+1]*model_instance.grid.dxC[:,i+1]))
            i = 1
            dV_dx[:,1:,i] = (VVEL[:,1:,i+1] - VVEL[:,1:,i])/(model_instance.grid.dxC[:,i+1])
            i = VVEL.shape[2]-1
            dV_dx[:,1:,i] = (VVEL[:,1:,i] - VVEL[:,1:,i-1])/(model_instance.grid.dxC[:,i])

            self[output_field] = dV_dx
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return

    def take_d_dy(self,model_instance,input_field = 'VVEL',output_field='dV_dy'):
        """ Take the y derivative of the field given on v-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            VVEL = self[input_field]
            dV_dy = np.zeros((VVEL.shape))

            for j in xrange(1,VVEL.shape[1]-2):
                dV_dy[:,j,:] = np.nan_to_num(model_instance.grid.wet_mask_V[:,j,:]*(
                        model_instance.grid.wet_mask_V[:,j+1,:]*VVEL[:,j+1,:] + 
                        (1 - model_instance.grid.wet_mask_V[:,j+1,:])*VVEL[:,j,:] - 
                        (1 - model_instance.grid.wet_mask_V[:,j-1,:])*VVEL[:,j,:] - 
                        model_instance.grid.wet_mask_V[:,j-1,:]*VVEL[:,j-1,:])/(
                        model_instance.grid.wet_mask_V[:,j-1,:]*model_instance.grid.dyF[j-1,:] + 
                        model_instance.grid.wet_mask_V[:,j+1,:]*model_instance.grid.dyF[j,:]))
            j = 1
            dV_dy[:,j,:] = (VVEL[:,j+1,:] - VVEL[:,j,:])/(model_instance.grid.dyF[j,:])
            j = VVEL.shape[1]-1
            dV_dy[:,j,:] = (VVEL[:,j,:] - VVEL[:,j-1,:])/(model_instance.grid.dyF[j-1,:])

            self[output_field] = dV_dy
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return
    
    
    def take_d_dz(self,model_instance,input_field = 'VVEL',output_field='dV_dz'):
        """ Take the z derivative of the field given on v-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            VVEL = self[input_field]
            d_dz = np.zeros((VVEL.shape))

            for k in xrange(1,VVEL.shape[0]-1):
                # model doesn't have overhangs, so only need this to work with fluid above and bathymetry below.
                d_dz[k,:,:] = np.nan_to_num(model_instance.grid.wet_mask_V[k,:,:]*(VVEL[k-1,:,:]  -
                    (1-model_instance.grid.wet_mask_V[k+1,:,:])*VVEL[k,:,:]-
                    model_instance.grid.wet_mask_V[k+1,:,:]*VVEL[k+1,:,:])/(model_instance.grid.drC[k] +
                    model_instance.grid.wet_mask_V[k+1,:,:]*model_instance.grid.drC[k+1]))

                k = 0
                d_dz[k,:,:] = (VVEL[k,:,:] - VVEL[k+1,:,:])/(model_instance.grid.drC[k+1])
                k = VVEL.shape[0]-1
                d_dz[k,:,:] = (VVEL[k-1,:,:] - VVEL[k,:,:])/(model_instance.grid.drC[k])

            self[output_field] = d_dz
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return
    
class Wpoint_field(Simulation,dict):

    def __init__(self,netcdf_filename,variable,time_level):
        netcdf_file = netCDF4.MFDataset(netcdf_filename)
        loaded_field = netcdf_file.variables[variable][time_level,:,:,:]
        netcdf_file.close()
        
        # Append a level of zeros on to the W point fields to represent no flow through the bottom of the domain.
        # It's a hack, but it helps with calculations later on.
	if hasattr(time_level, '__len__'):
        	self[variable] = np.append(loaded_field,np.zeros((len(time_level),1,loaded_field.shape[-2],loaded_field.shape[-1])),axis=1)
	else:
        	self[variable] = np.append(loaded_field,np.zeros((1,loaded_field.shape[1],loaded_field.shape[2])),axis=0)


        return
    

    
    def take_d_dx(self,model_instance,input_field = 'WVEL',output_field='dW_dx'):
        """Take the x derivative of the field on w points, using spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            WVEL = self[input_field]
            d_dx = np.zeros((WVEL.shape))

            for i in xrange(1,WVEL.shape[2]-1):
                d_dx[:,:,i] = np.nan_to_num(model_instance.grid.wet_mask_W[:,:,i]*
                                (model_instance.grid.wet_mask_W[:,:,i+1]*WVEL[:,:,i+1] + 
                                (1 - model_instance.grid.wet_mask_W[:,:,i+1])*WVEL[:,:,i] - 
                                (1 - model_instance.grid.wet_mask_W[:,:,i-1])*WVEL[:,:,i] - 
                                model_instance.grid.wet_mask_W[:,:,i-1]*WVEL[:,:,i-1])/(
                                model_instance.grid.wet_mask_W[:,:,i-1]*model_instance.grid.dxC[:,i] + 
                                model_instance.grid.wet_mask_W[:,:,i+1]*model_instance.grid.dxC[:,i+1]))
            i = 1
            d_dx[:,:,i] = (WVEL[:,:,i+1] - WVEL[:,:,i])/(model_instance.grid.dxC[:,i+1])
            i = WVEL.shape[2]-1
            d_dx[:,:,i] = (WVEL[:,:,i] - WVEL[:,:,i-1])/(model_instance.grid.dxC[:,i])

            self[output_field] = d_dx

        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return  

	
    def take_d_dy(self,model_instance,input_field = 'WVEL',output_field='dW_dy'):
        """Take the y derivative of the field on w points, using spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            WVEL = self[input_field]
            dW_dy = np.zeros((WVEL.shape))

            for j in xrange(1,WVEL.shape[2]-1):
                dW_dy[:,j,:] = np.nan_to_num(model_instance.grid.wet_mask_W[:,j,:]*
                                    (model_instance.grid.wet_mask_W[:,j+1,:]*WVEL[:,j+1,:] + 
                                    (1 - model_instance.grid.wet_mask_W[:,j+1,:])*WVEL[:,j,:] - 
                                    (1 - model_instance.grid.wet_mask_W[:,j-1,:])*WVEL[:,j,:] - 
                                    model_instance.grid.wet_mask_W[:,j-1,:]*WVEL[:,j-1,:])/(
                                    model_instance.grid.wet_mask_W[:,j-1,:]*model_instance.grid.dyC[j,:] + 
                                    model_instance.grid.wet_mask_W[:,j+1,:]*model_instance.grid.dyC[j+1,:]))
            j = 1
            dW_dy[:,j,:] = (WVEL[:,j+1,:] - WVEL[:,j,:])/(model_instance.grid.dyC[j+1,:])
            j = WVEL.shape[1]-1
            dW_dy[:,j,:] = (WVEL[:,j,:] - WVEL[:,j-1,:])/(model_instance.grid.dyC[j,:])

            self[output_field] = dW_dy
       
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return 

    def take_d_dz(self,model_instance,input_field = 'WVEL',output_field='dW_dz'):
        """ Take the z derivative of the field given on w-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""  
        
        if input_field in self:
            WVEL = self[input_field]
            dWVEL_dz = np.zeros((WVEL.shape))

            for k in xrange(1,WVEL.shape[0]-2):
                dWVEL_dz[k,:,:] = np.nan_to_num(model_instance.grid.wet_mask_TH[k,:,:]*(WVEL[k-1,:,:] -
                            (1-model_instance.grid.wet_mask_TH[k+1,:,:])*WVEL[k,:,:]-
                            model_instance.grid.wet_mask_TH[k+1,:,:]*WVEL[k+1,:,:])/
                            (model_instance.grid.drF[k-1]+
                            model_instance.grid.wet_mask_TH[k+1,:,:]*model_instance.grid.drF[k]))

                k = 0
                dWVEL_dz[k,:,:] = (WVEL[k,:,:] - WVEL[k+1,:,:])/(model_instance.grid.drF[k])
                k = WVEL.shape[0]-2
                dWVEL_dz[k,:,:] = (WVEL[k-1,:,:] - WVEL[k,:,:])/(model_instance.grid.drF[k-1])        
                k = WVEL.shape[0]-1
                dWVEL_dz[k,:,:] = (WVEL[k-1,:,:] - WVEL[k,:,:])/(model_instance.grid.drF[k-1])

            self[output_field] = dWVEL_dz
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return 


    
    
class Tracerpoint_field(Simulation,dict):  
    
    def take_d_dx(self,model_instance,input_field = 'RHO',output_field='dRHO_dx'):
        """Take the x derivative of the field on tracer points, using spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            rho = self[input_field]
            d_dx = np.zeros((rho.shape))

            for i in xrange(1,rho.shape[2]-1):
                d_dx[:,:,i] = np.nan_to_num(model_instance.grid.wet_mask_TH[:,:,i]*
                        (model_instance.grid.wet_mask_TH[:,:,i+1]*rho[:,:,i+1] + 
                        (1 - model_instance.grid.wet_mask_TH[:,:,i+1])*rho[:,:,i] - 
                        (1 - model_instance.grid.wet_mask_TH[:,:,i-1])*rho[:,:,i] - 
                        model_instance.grid.wet_mask_TH[:,:,i-1]*rho[:,:,i-1])/(
                        model_instance.grid.wet_mask_TH[:,:,i-1]*model_instance.grid.dxC[:,i] + 
                        model_instance.grid.wet_mask_TH[:,:,i+1]*model_instance.grid.dxC[:,i+1]))
            i = 1
            d_dx[:,:,i] = (rho[:,:,i+1] - rho[:,:,i])/(model_instance.grid.dxC[:,i+1])
            i = rho.shape[2]-1
            d_dx[:,:,i] = (rho[:,:,i] - rho[:,:,i-1])/(model_instance.grid.dxC[:,i])

            self[output_field] = d_dx
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return 


    def take_d_dy(self,model_instance,input_field = 'RHO',output_field='dRHO_dy'):
        """Take the y derivative of the field on tracer points, using spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            rho = self[input_field]
            d_dy = np.zeros((rho.shape))

            for j in xrange(1,rho.shape[2]-1):
                d_dy[:,j,:] = np.nan_to_num(model_instance.grid.wet_mask_TH[:,j,:]*
                                (model_instance.grid.wet_mask_TH[:,j+1,:]*rho[:,j+1,:] + 
                                (1 - model_instance.grid.wet_mask_TH[:,j+1,:])*rho[:,j,:] - 
                                (1 - model_instance.grid.wet_mask_TH[:,j-1,:])*rho[:,j,:] - 
                                model_instance.grid.wet_mask_TH[:,j-1,:]*rho[:,j-1,:])/(
                                model_instance.grid.wet_mask_TH[:,j-1,:]*model_instance.grid.dyC[j,:] + 
                                model_instance.grid.wet_mask_TH[:,j+1,:]*model_instance.grid.dyC[j+1,:]))
            j = 1
            d_dy[:,j,:] = (rho[:,j+1,:] - rho[:,j,:])/(model_instance.grid.dyC[j+1,:])
            j = rho.shape[1]-1
            d_dy[:,j,:] = (rho[:,j,:] - rho[:,j-1,:])/(model_instance.grid.dyC[j,:])

            self[output_field] = d_dy
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return  

    
    def take_d_dz(self,model_instance,input_field = 'RHO',output_field='dRHO_dz'):
        """ Take the z derivative of the field given on tracer-points, using the spacings in grid object.

        Performs centred second-order differencing everywhere except next to boundaries. First order is 
        used there (meaning the gradients at the boundary are evaluated half a grid point away from where 
        they should be)."""

        if input_field in self:
            rho = self[input_field]
            d_dz = np.zeros((rho.shape))

            for k in xrange(1,rho.shape[0]-1):
                # model doesn't have overhangs, so only need this to work with fluid above and bathymetry below.
                d_dz[k,:,:] = np.nan_to_num(model_instance.grid.wet_mask_TH[k,:,:]*(rho[k-1,:,:]  -
                                    (1-model_instance.grid.wet_mask_TH[k+1,:,:])*rho[k,:,:]-
                                    model_instance.grid.wet_mask_TH[k+1,:,:]*rho[k+1,:,:])/(model_instance.grid.drC[k] +
                                    model_instance.grid.wet_mask_TH[k+1,:,:]*model_instance.grid.drC[k+1]))

                k = 0
                d_dz[k,:,:] = (rho[k,:,:] - rho[k+1,:,:])/(model_instance.grid.drC[k+1])
                k = rho.shape[0]-1
                d_dz[k,:,:] = (rho[k-1,:,:] - rho[k,:,:])/(model_instance.grid.drC[k])

            self[output_field] = d_dz
        else:
            raise ValueError('Chosen input array ' + str(input_field) + ' is not defined')
        return 


    
class Vorticitypoint_field(Simulation,dict):  
    pass
    # Eventually I should put some derivative funcitons in here.



class Grid(Simulation):

    def __init__(self, grid_netcdf_filename):
        """Define a single object that has all of the grid variables tucked away in it. 
        Each of the variables pulled directly from the netcdf file still has the 
        original description attached to it. The 2D and 3D arrays do not."""
        grid_netcdf_file = netCDF4.Dataset(grid_netcdf_filename)
        self.rAw = grid_netcdf_file.variables['rAw']
        self.rAs = grid_netcdf_file.variables['rAs']
        self.rA = grid_netcdf_file.variables['rA']
        self.HFacW = grid_netcdf_file.variables['HFacW']
        self.HFacS = grid_netcdf_file.variables['HFacS']
        self.HFacC = grid_netcdf_file.variables['HFacC']
        self.X = grid_netcdf_file.variables['X']
        self.Xp1 = grid_netcdf_file.variables['Xp1']
        self.dxF = grid_netcdf_file.variables['dxF']
        self.dxC = grid_netcdf_file.variables['dxC']
        self.dxV = grid_netcdf_file.variables['dxV']
        self.Y = grid_netcdf_file.variables['Y']
        self.dyU = grid_netcdf_file.variables['dyU']
        self.dyC = grid_netcdf_file.variables['dyC']
        self.dyF = grid_netcdf_file.variables['dyF']
        self.Z = grid_netcdf_file.variables['Z']
        self.Zl = grid_netcdf_file.variables['Zl']
        self.Zu = grid_netcdf_file.variables['Zu']
        self.drC = grid_netcdf_file.variables['drC']
        self.drF = grid_netcdf_file.variables['drF']
        self.fCoriG = grid_netcdf_file.variables['fCoriG']

        (self.Z_y,self.Y_z) = np.meshgrid(self.Z[:],self.Y,indexing='ij')
        (self.X_y,self.Y_x) = np.meshgrid(self.X,self.Y,indexing='ij')
        (self.Z_x,self.X_z) = np.meshgrid(self.Z,self.X,indexing='ij')
        (self.Z_3d,self.Y_3d,self.X_3d) = np.meshgrid(self.Z[:],self.Y,self.X,indexing='ij')

        (self.DZF,self.DYF, self.DXF) = np.meshgrid(self.drF,self.dyF[0,:],self.dxF[:,0],indexing='ij')


        self.wet_mask_V = copy.deepcopy(np.ones((np.shape(self.HFacS))))
        self.wet_mask_V[self.HFacS[:] == 0.] = 0.
        self.wet_mask_U = copy.deepcopy(np.ones((np.shape(self.HFacW))))
        self.wet_mask_U[self.HFacW[:] == 0.] = 0.
        self.wet_mask_TH = copy.deepcopy(np.ones((np.shape(self.HFacC))))
        self.wet_mask_TH[self.HFacC[:] == 0.] = 0.
        self.wet_mask_W = np.append(self.wet_mask_TH,np.ones((1,480,480)),axis=0)

	self.west_mask = np.zeros(self.wet_mask_TH.shape)
	self.east_mask = np.zeros(self.wet_mask_TH.shape)
	self.south_mask = np.zeros(self.wet_mask_TH.shape)
	self.north_mask = np.zeros(self.wet_mask_TH.shape)
	self.bottom_mask = np.zeros(self.wet_mask_TH.shape)


	# Find the fluxes through the boundary of the domain
	for k in xrange(0,self.wet_mask_TH.shape[0]):
	    for j in xrange(0,self.wet_mask_TH.shape[1]):
		for i in xrange(0,self.wet_mask_TH.shape[2]):
		    # find points with boundary to the west. In the simplest shelf configuration this is the only tricky boundary to find.
		    if self.wet_mask_TH[k,j,i] - self.wet_mask_TH[k,j,i-1] == 1:
		        self.west_mask[k,j,i] = 1


		    # find the eastern boundary points. Negative sign is to be consistent about fluxes into the domain.
		    if self.wet_mask_TH[k,j,i-1] - self.wet_mask_TH[k,j,i] == 1:
		        self.east_mask[k,j,i] = 1


		    # find the southern boundary points
		    if self.wet_mask_TH[k,j,i] - self.wet_mask_TH[k,j-1,i] == 1:
		        self.south_mask[k,j,i] = 1


		    # find the northern boundary points
		    if self.wet_mask_TH[k,j-1,i] - self.wet_mask_TH[k,j,i] == 1:
		        self.north_mask[k,j,i] = 1


		    # Fluxes through the bottom
		    if self.wet_mask_TH[k-1,j,i] - self.wet_mask_TH[k,j,i] == 1:
		        self.bottom_mask[k,j,i] = 1

        return

    
class Temperature(Tracerpoint_field):
    def __init__(self,netcdf_filename,variable,time_level):
        netcdf_file = netCDF4.MFDataset(netcdf_filename)
        loaded_field = netcdf_file.variables[variable][time_level,:,:,:]
        netcdf_file.close()
        
        self[variable] = loaded_field
        return
            
            
class Density(Tracerpoint_field):
    def __init__(self,model_instance,Talpha=2e-4,Sbeta=0,RhoNil=1035,cp=4000):
        if model_instance['EOS_type'] == 'linear':
            self['cp'] = cp
            self['Talpha'] = Talpha
            self['Sbeta'] = Sbeta
            self['RhoNil'] = RhoNil
            if Sbeta == 0:
                self['RHO'] = (RhoNil*( -Talpha*(model_instance.temperature['THETA'] - 25)) 
                          + RhoNil)
                    # final term is to make density very high in the cells taht aren't fluid.
            else:
                raise ValueError('Linear EOS only supports temperature variations at the moment. Sorry.') 
        else:
            raise ValueError('Only linear EOS supported at the moment. Sorry.')
                


    def calculate_TotRhoTend(self,model_instance):
        if model_instance['EOS_type'] == 'linear':
            if self['Sbeta'] == 0:
                self['TotRhoTend'] = (-self['RhoNil']*self['Talpha']*model_instance.temperature['TOTTTEND'])
            else:
                raise ValueError('Linear EOS only supports temperature variations at the moment. Sorry.') 
        else:    
            raise ValueError('Only liner EOS supported at the moment.')

            
            
class Bernoulli(Tracerpoint_field):
    def __init__(self,model_instance):
        self['BP'] = model_instance.grid.wet_mask_TH*(((model_instance.pressure['P'][:,:,:] + 
                 model_instance.grid.Z[:].reshape((40,1,1))*
                                    model_instance.density['RHO'][:,:,:]*model_instance['g'])/
                 model_instance.density['RhoNil']) + 
                 ((model_instance.zonal_velocity['UVEL'][:,:,1:]*model_instance.zonal_velocity['UVEL'][:,:,1:] + 
                 model_instance.zonal_velocity['UVEL'][:,:,:-1]*model_instance.zonal_velocity['UVEL'][:,:,:-1])/2 + 
                 (model_instance.meridional_velocity['VVEL'][:,1:,:]*model_instance.meridional_velocity['VVEL'][:,1:,:] + 
                 model_instance.meridional_velocity['VVEL'][:,:-1,:]*model_instance.meridional_velocity['VVEL'][:,:-1,:])/2)/2)

        
class Free_surface(Tracerpoint_field):
    def __init__(self,netcdf_filename,variable,time_level):
        netcdf_file = netCDF4.MFDataset(netcdf_filename)
        loaded_field = netcdf_file.variables[variable][time_level,:,:,:]
        netcdf_file.close()
        
        self[variable] = loaded_field
        return
            
        
class Pressure(Tracerpoint_field):
    def __init__(self,model_instance):

        # derive the hydrostatic pressure
        delta_P = np.zeros((np.shape(model_instance.density['RHO'])))
        delta_P[:,:,:] = model_instance['g']*model_instance.density['RHO'][:,:,:]*model_instance.grid.drF[:].reshape(40,1,1);
        
    
        # add free surface contribution
        delta_P[0,:,:] = (delta_P[0,:,:] + 
                          model_instance.free_surface['ETAN']*model_instance['g']*model_instance.density['RHO'][0,:,:])
    
        self['delta_P'] = delta_P
        self['P'] = np.cumsum(delta_P,0)
        
        return
    
class Vorticity(Vorticitypoint_field):
    def __init__(self,netcdf_filename = '3D_fields.all.nc',variable='momVort3',time_level=0):
        netcdf_file = netCDF4.MFDataset(netcdf_filename)
        self[variable] = netcdf_file.variables[variable][time_level,:,:]
        netcdf_file.close()
        
class Potential_vorticity(Tracerpoint_field):
    def __init__(self,model_instance):
        self['Q'] = -model_instance.vorticity['omega_a']*model_instance.density['dRHO_dz']/model_instance.density['RHO']
