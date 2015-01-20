import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import netCDF4
import copy
import cPickle as pickle
import time
import mitgcm

plt.rcParams['figure.figsize'] = (12, 12) # set default figure size to 12x12 inches


path_to_output = '/media/Backup_1/Archive/MITgcm-simulations/4.165km_shelf/no-slip/simulation_011/'

# create model objects for 1 day and 5 day means
m = mitgcm.Simulation(path_to_output,'grid.all.nc')
m_5_day_means = mitgcm.Simulation(path_to_output,'grid.all.nc')

f = netCDF4.MFDataset('2D_fields.all.nc')
total_time_levels = len(f.variables['T'][:])
m['time'] = (f.variables['T'][:])
f.close()

counter_5_day_means = 0



os.chdir('input/')
m.grid.topog = np.fromfile('topog.box',dtype='>f8', count=-1, sep='')
m.grid.topog = np.reshape(m.grid.topog,(480,480))
m['heat_flux'] = np.fromfile('Qnet_p64.bin',dtype='>f8',count=-1,sep='')
m['heat_flux'] = np.reshape(m['heat_flux'],(480,480))
m['heat_flux'] = m['heat_flux'][::-1,:]

m_5_day_means.grid.topog = np.fromfile('topog.box',dtype='>f8', count=-1, sep='')
m_5_day_means.grid.topog = np.reshape(m_5_day_means.grid.topog,(480,480))
m_5_day_means['heat_flux'] = np.fromfile('Qnet_p64.bin',dtype='>f8',count=-1,sep='')
m_5_day_means['heat_flux'] = np.reshape(m_5_day_means['heat_flux'],(480,480))
m_5_day_means['heat_flux'] = m_5_day_means['heat_flux'][::-1,:]
os.chdir('../')



PV_flux_boundary = np.zeros((len(m.grid.drF),len(m.grid.dxF),len(m.grid.dyF)),dtype=float)
Q_in_domain = np.zeros((total_time_levels,1))
Q_5_day_means_in_domain = np.zeros((total_time_levels,1))

elapsed = np.zeros((total_time_levels+1,1))


#Make some pretty pics?
make_pictures = True



# NetCDF file for output
sim_number =  path_to_output[-4:-1]

# 5 day means
average_over = 5
PV_netcdf_5_day_means= netCDF4.Dataset('PV_5_day_means_from_simulation_{simulation_number}.nc'.format(simulation_number=sim_number),
                           'w', format='NETCDF4')

PV_depth_5_day_means = PV_netcdf_5_day_means.createDimension('depth', len(m.grid.Zl))
PV_lat_5_day_means = PV_netcdf_5_day_means.createDimension('lat', len(m.grid.Y))
PV_lon_5_day_means = PV_netcdf_5_day_means.createDimension('lon', len(m.grid.X))
PV_time_5_day_means = PV_netcdf_5_day_means.createDimension('time', None)


PV_times_5_day_means = PV_netcdf_5_day_means.createVariable('time','f4',('time',))
PV_depths_5_day_means = PV_netcdf_5_day_means.createVariable('depth','f4',('depth',))
PV_latitudes_5_day_means = PV_netcdf_5_day_means.createVariable('latitude','f4',('lat',))
PV_longitudes_5_day_means = PV_netcdf_5_day_means.createVariable('longitude','f4',('lon',))

PV_latitudes_5_day_means[:] = m.grid.Y
PV_longitudes_5_day_means[:] = m.grid.X
PV_depths_5_day_means[:] = m.grid.Z

J_x_5_day_means = PV_netcdf_5_day_means.createVariable('J_x','f8',('time','depth','lat','lon',))
J_x_5_day_means.units = 'kg m^-3 s^-2'

J_y_5_day_means = PV_netcdf_5_day_means.createVariable('J_y','f8',('time','depth','lat','lon',))
J_y_5_day_means.units = 'kg m^-3 s^-2'

J_z_5_day_means = PV_netcdf_5_day_means.createVariable('J_z','f8',('time','depth','lat','lon',))
J_z_5_day_means.units = 'kg m^-3 s^-2'

boundary_fluxes_5_day_means = PV_netcdf_5_day_means.createVariable('boundary_flux','f8',('time','depth','lat','lon',))
boundary_fluxes_5_day_means.units = 'kg m^-3 s^-2'



time.clock()
elapsed[0] = time.clock()
for time_level in xrange(0,5):#total_time_levels):
    # load in zonal velocity and tendency
    m.zonal_velocity = mitgcm.Upoint_field('3D_fields.all.nc','UVEL',time_level)
    m.zonal_velocity.load_field('U_mom_diags.all.nc','TOTUTEND',time_level)
    #take derivatives
    m.zonal_velocity.take_d_dx(m,input_field='UVEL',output_field='dU_dx')
    m.zonal_velocity.take_d_dy(m,input_field='UVEL',output_field='dU_dy')
    m.zonal_velocity.take_d_dz(m,input_field='UVEL',output_field='dU_dz')

    m.meridional_velocity = mitgcm.Vpoint_field('3D_fields.all.nc','VVEL',time_level)
    m.meridional_velocity.load_field('V_mom_diags.all.nc','TOTVTEND',time_level)
    m.meridional_velocity.take_d_dx(m,input_field='VVEL', output_field='dV_dx')
    m.meridional_velocity.take_d_dy(m,input_field='VVEL', output_field='dV_dy')
    m.meridional_velocity.take_d_dz(m,input_field='VVEL', output_field='dV_dz')

    m.vertical_velocity = mitgcm.Wpoint_field('3D_fields.all.nc', 'WVEL',time_level)
    m.vertical_velocity.take_d_dx(m, input_field='WVEL', output_field='dW_dx')
    m.vertical_velocity.take_d_dy(m, input_field='WVEL', output_field='dW_dy')
    m.vertical_velocity.take_d_dz(m, input_field='WVEL', output_field='dW_dz')

    m.temperature = mitgcm.Temperature('theta_diags.all.nc','THETA', time_level)
    m.temperature.load_field('theta_diags.all.nc','TOTTTEND',time_level)

    m.free_surface = mitgcm.Free_surface('2D_fields.all.nc', 'ETAN', time_level)
    

    m.density = mitgcm.Density(m,Talpha=0.0002, Sbeta=0, RhoNil=1035, cp=4000)
    m.density.take_d_dx(m, input_field='RHO', output_field='dRHO_dx')
    m.density.take_d_dy(m, input_field='RHO', output_field='dRHO_dy')
    m.density.take_d_dz(m, input_field='RHO', output_field='dRHO_dz')
    m.density.calculate_TotRhoTend(m)

    m.pressure = mitgcm.Pressure(m)

    m.bernoulli = mitgcm.Bernoulli(m)
    m.bernoulli.take_d_dx(m, input_field='BP', output_field='dBP_dx')
    m.bernoulli.take_d_dy(m, input_field='BP', output_field='dBP_dy')
    m.bernoulli.take_d_dz(m, input_field='BP', output_field='dBP_dz')

    m.vorticity = mitgcm.Vorticity(netcdf_filename='3D_fields.all.nc', variable='momVort3', time_level=time_level)

    m.vorticity['omega_a'] = m.grid.fCoriG + m.vorticity['momVort3']
    m.vorticity['omega_a'] = (m.vorticity['omega_a'][:,1:,:] + m.vorticity['omega_a'][:,:-1,:])/2
    m.vorticity['omega_a'] = (m.vorticity['omega_a'][:,:,1:] + m.vorticity['omega_a'][:,:,:-1])/2

    m.PV = mitgcm.Potential_vorticity(m)

    m.PV['J_x'] = (((m.vertical_velocity['dW_dy'][:-1,:,:] + m.vertical_velocity['dW_dy'][1:,:,:])/2 - 
             (m.meridional_velocity['dV_dz'][:,1:,:]+m.meridional_velocity['dV_dz'][:,:-1,:])/2)*
           m.density['TotRhoTend'] + 
           (m.meridional_velocity['TOTVTEND'][:,1:,:]+m.meridional_velocity['TOTVTEND'][:,:-1,:])*
           m.density['dRHO_dz']/2  + 
           (m.bernoulli['dBP_dy']*m.density['dRHO_dz'] - m.bernoulli['dBP_dz']*m.density['dRHO_dy'])
           )*(m.grid.DYF*m.grid.DZF*m.grid.HFacC)

    m.PV['J_y'] = (((m.zonal_velocity['dU_dz'][:,:,1:]+m.zonal_velocity['dU_dz'][:,:,:-1])/2 - 
              (m.vertical_velocity['dW_dx'][:-1,:,:] + m.vertical_velocity['dW_dx'][1:,:,:])/2)*
           m.density['TotRhoTend'] +
            -(m.zonal_velocity['TOTUTEND'][:,:,1:]+m.zonal_velocity['TOTUTEND'][:,:,:-1])*
            m.density['dRHO_dz']/2 + 
           (m.bernoulli['dBP_dz']*m.density['dRHO_dx'] - m.bernoulli['dBP_dx']*m.density['dRHO_dz'])
           )*m.grid.DXF*m.grid.DZF*m.grid.HFacC

    m.PV['J_z'] = (m.vorticity['omega_a']*m.density['TotRhoTend'] + 
           ((m.zonal_velocity['TOTUTEND'][:,:,1:]+m.zonal_velocity['TOTUTEND'][:,:,:-1])*m.density['dRHO_dy']/2 - 
            (m.meridional_velocity['TOTVTEND'][:,1:,:]+m.meridional_velocity['TOTVTEND'][:,:-1,:])*
            m.density['dRHO_dx']/2 ) + 
           (m.bernoulli['dBP_dx']*m.density['dRHO_dy'] - m.bernoulli['dBP_dy']*m.density['dRHO_dx'])
           )*m.grid.DXF*m.grid.DYF


    Q_in_domain[time_level] = np.sum(m.PV['Q'])
    
    PV_flux_boundary = np.zeros((len(m.grid.drF),len(m.grid.dxF),len(m.grid.dyF)),dtype=float)
    PV_flux_boundary[...] += m.PV['J_x'][:] * m.grid.west_mask
    PV_flux_boundary[...] += -m.PV['J_x'][:] * m.grid.east_mask
    PV_flux_boundary[...] += m.PV['J_y'][:] * m.grid.south_mask
    PV_flux_boundary[...] += -m.PV['J_y'][:] * m.grid.north_mask
    PV_flux_boundary[...] += m.PV['J_z'][:] * m.grid.bottom_mask
    PV_flux_boundary[0,:,:] += -m.PV['J_z'][0,:,:]
    
    J_x_1_day_means[time_level,...] = m.PV['J_x'][:]
    J_y_1_day_means[time_level,...] = m.PV['J_y'][:]
    J_z_1_day_means[time_level,...] = m.PV['J_z'][:]
    boundary_fluxes_1_day_means[time_level,...] = PV_flux_boundary[:]

    
    if np.mod(time_level+1,average_over) == 0:
        #load in the previous five time slices and then average them
        index = np.arange(time_level-average_over+1,time_level+1,1)
        
        m_5_day_means.zonal_velocity = mitgcm.Upoint_field('3D_fields.all.nc','UVEL',index)
        m_5_day_means.zonal_velocity['UVEL'] = m_5_day_means.zonal_velocity['UVEL'].mean(axis=0)
        m_5_day_means.zonal_velocity.load_field('U_mom_diags.all.nc','TOTUTEND',index)
        m_5_day_means.zonal_velocity['TOTUTEND'] = m_5_day_means.zonal_velocity['TOTUTEND'].mean(axis=0)
        m_5_day_means.zonal_velocity.take_d_dx(m,input_field='UVEL',output_field='dU_dx')
        m_5_day_means.zonal_velocity.take_d_dy(m,input_field='UVEL',output_field='dU_dy')
        m_5_day_means.zonal_velocity.take_d_dz(m,input_field='UVEL',output_field='dU_dz')

        m_5_day_means.meridional_velocity = mitgcm.Vpoint_field('3D_fields.all.nc','VVEL',index)
        m_5_day_means.meridional_velocity['VVEL'] = m_5_day_means.meridional_velocity['VVEL'].mean(axis=0)
        m_5_day_means.meridional_velocity.load_field('V_mom_diags.all.nc','TOTVTEND',index)
        m_5_day_means.meridional_velocity['TOTVTEND'] = m_5_day_means.meridional_velocity['TOTVTEND'].mean(axis=0)
        m_5_day_means.meridional_velocity.take_d_dx(m,input_field='VVEL', output_field='dV_dx')
        m_5_day_means.meridional_velocity.take_d_dy(m,input_field='VVEL', output_field='dV_dy')
        m_5_day_means.meridional_velocity.take_d_dz(m,input_field='VVEL', output_field='dV_dz')

        m_5_day_means.vertical_velocity = mitgcm.Wpoint_field('3D_fields.all.nc', 'WVEL',index)
        m_5_day_means.vertical_velocity['WVEL'] = m_5_day_means.vertical_velocity['WVEL'].mean(axis=0)
        m_5_day_means.vertical_velocity.take_d_dx(m, input_field='WVEL', output_field='dW_dx')
        m_5_day_means.vertical_velocity.take_d_dy(m, input_field='WVEL', output_field='dW_dy')
        m_5_day_means.vertical_velocity.take_d_dz(m, input_field='WVEL', output_field='dW_dz')

        m_5_day_means.temperature = mitgcm.Temperature('theta_diags.all.nc','THETA',index)
        m_5_day_means.temperature['THETA'] = m_5_day_means.temperature['THETA'].mean(axis=0)
        m_5_day_means.temperature.load_field('theta_diags.all.nc','TOTTTEND',time_level = index)
        m_5_day_means.temperature['TOTTTEND'] = m_5_day_means.temperature['TOTTTEND'].mean(axis=0)

        m_5_day_means.free_surface = mitgcm.Free_surface('2D_fields.all.nc','ETAN',index)
        m_5_day_means.free_surface['ETAN'] = m_5_day_means.free_surface['ETAN'].mean(axis=0)

        m_5_day_means.density = mitgcm.Density(m,Talpha=0.0002, Sbeta=0, RhoNil=1035, cp=4000)
        m_5_day_means.density.take_d_dx(m_5_day_means, input_field='RHO', output_field='dRHO_dx')
        m_5_day_means.density.take_d_dy(m_5_day_means, input_field='RHO', output_field='dRHO_dy')
        m_5_day_means.density.take_d_dz(m_5_day_means, input_field='RHO', output_field='dRHO_dz')
        m_5_day_means.density.calculate_TotRhoTend(m_5_day_means)

        m_5_day_means.pressure = mitgcm.Pressure(m_5_day_means)

        m_5_day_means.bernoulli = mitgcm.Bernoulli(m_5_day_means)
        m_5_day_means.bernoulli.take_d_dx(m_5_day_means, input_field='BP', output_field='dBP_dx')
        m_5_day_means.bernoulli.take_d_dy(m_5_day_means, input_field='BP', output_field='dBP_dy')
        m_5_day_means.bernoulli.take_d_dz(m_5_day_means, input_field='BP', output_field='dBP_dz')

        m_5_day_means.vorticity = mitgcm.Vorticity('3D_fields.all.nc','momVort3',index)
        m_5_day_means.vorticity['momVort3'] = m_5_day_means.vorticity['momVort3'].mean(axis=0)

        m_5_day_means.vorticity['omega_a'] = m_5_day_means.grid.fCoriG + m_5_day_means.vorticity['momVort3']
        # Average on to the traver points
        m_5_day_means.vorticity['omega_a'] = (m_5_day_means.vorticity['omega_a'][:,1:,:] + 
                                              m_5_day_means.vorticity['omega_a'][:,:-1,:])/2
        m_5_day_means.vorticity['omega_a'] = (m_5_day_means.vorticity['omega_a'][:,:,1:] + 
                                              m_5_day_means.vorticity['omega_a'][:,:,:-1])/2

        m_5_day_means.PV = mitgcm.Potential_vorticity(m_5_day_means)

        m_5_day_means.PV['J_x'] = (((m_5_day_means.vertical_velocity['dW_dy'][:-1,:,:] + 
                                     m_5_day_means.vertical_velocity['dW_dy'][1:,:,:])/2 - 
                 (m_5_day_means.meridional_velocity['dV_dz'][:,1:,:]+
                  m_5_day_means.meridional_velocity['dV_dz'][:,:-1,:])/2)*
               m_5_day_means.density['TotRhoTend'] + 
               (m_5_day_means.meridional_velocity['TOTVTEND'][:,1:,:]+
                m_5_day_means.meridional_velocity['TOTVTEND'][:,:-1,:])*
               m_5_day_means.density['dRHO_dz']/2  + 
               (m_5_day_means.bernoulli['dBP_dy']*m_5_day_means.density['dRHO_dz'] - 
                m_5_day_means.bernoulli['dBP_dz']*m_5_day_means.density['dRHO_dy'])
               )*(m_5_day_means.grid.DYF*m_5_day_means.grid.DZF*m_5_day_means.grid.HFacC)

        m_5_day_means.PV['J_y'] = (((m_5_day_means.zonal_velocity['dU_dz'][:,:,1:]+
                                     m_5_day_means.zonal_velocity['dU_dz'][:,:,:-1])/2 - 
                  (m_5_day_means.vertical_velocity['dW_dx'][:-1,:,:] + 
                   m_5_day_means.vertical_velocity['dW_dx'][1:,:,:])/2)*
               m_5_day_means.density['TotRhoTend'] +
                -(m_5_day_means.zonal_velocity['TOTUTEND'][:,:,1:]+
                  m_5_day_means.zonal_velocity['TOTUTEND'][:,:,:-1])*
                m_5_day_means.density['dRHO_dz']/2 + 
               (m_5_day_means.bernoulli['dBP_dz']*m_5_day_means.density['dRHO_dx'] - 
                m_5_day_means.bernoulli['dBP_dx']*m_5_day_means.density['dRHO_dz'])
               )*m_5_day_means.grid.DXF*m_5_day_means.grid.DZF*m_5_day_means.grid.HFacC

        m_5_day_means.PV['J_z'] = (m_5_day_means.vorticity['omega_a']*m_5_day_means.density['TotRhoTend'] + 
               ((m_5_day_means.zonal_velocity['TOTUTEND'][:,:,1:]+
                 m_5_day_means.zonal_velocity['TOTUTEND'][:,:,:-1])*m_5_day_means.density['dRHO_dy']/2 - 
                (m_5_day_means.meridional_velocity['TOTVTEND'][:,1:,:]+
                 m_5_day_means.meridional_velocity['TOTVTEND'][:,:-1,:])*
                m_5_day_means.density['dRHO_dx']/2 ) + 
               (m_5_day_means.bernoulli['dBP_dx']*m_5_day_means.density['dRHO_dy'] - 
                m_5_day_means.bernoulli['dBP_dy']*m_5_day_means.density['dRHO_dx'])
               )*m_5_day_means.grid.DXF*m_5_day_means.grid.DYF
        
        Q_5_day_means_in_domain[time_level] = np.sum(m_5_day_means.PV['Q'])
    
        PV_flux_boundary_5_day_means = np.zeros((len(m_5_day_means.grid.drF),
                                                 len(m_5_day_means.grid.dxF),
                                                 len(m_5_day_means.grid.dyF)),dtype=float)
        PV_flux_boundary_5_day_means[...] += m_5_day_means.PV['J_x'][:] * m_5_day_means.grid.west_mask
        PV_flux_boundary_5_day_means[...] += -m_5_day_means.PV['J_x'][:] * m_5_day_means.grid.east_mask
        PV_flux_boundary_5_day_means[...] += m_5_day_means.PV['J_y'][:] * m_5_day_means.grid.south_mask
        PV_flux_boundary_5_day_means[...] += -m_5_day_means.PV['J_y'][:] * m_5_day_means.grid.north_mask
        PV_flux_boundary_5_day_means[...] += m_5_day_means.PV['J_z'][:] * m_5_day_means.grid.bottom_mask
        PV_flux_boundary_5_day_means[0,:,:] += -m_5_day_means.PV['J_z'][0,:,:]

        J_x_5_day_means[counter_5_day_means,...] = m_5_day_means.PV['J_x'][:]
        J_y_5_day_means[counter_5_day_means,...] = m_5_day_means.PV['J_y'][:]
        J_z_5_day_means[counter_5_day_means,...] = m_5_day_means.PV['J_z'][:]
        boundary_fluxes_5_day_means[counter_5_day_means,...] = PV_flux_boundary_5_day_means[:]
        
        PV_times_5_day_means[counter_5_day_means] = m['time'][time_level]
        
        counter_5_day_means += 1

                    
    if make_pictures == True:
        # Plot WBC cross section
        j = 300
        i = 210
        k = 0
        q_space = 4

        slice_x0 = 0
        slice_length = 400
        slice_y0 = m.grid.Y[j]/1e3
        slice_dy = 0

        fig = plt.figure(figsize=(23,10))
        ax2 = fig.add_subplot(1,2,1)
        ax1 = fig.add_subplot(1,2,2)

        im = ax1.pcolor(m.grid.X[:]/1e3,m.grid.Y[:]/1e3,J_z[k,:,:],cmap='seismic',vmin = -500,vmax = 500)
        cax,kw = matplotlib.colorbar.make_axes(ax1)
        CB = plt.colorbar(im, cax=cax, **kw)
        CB.ax.tick_params(labelsize = 15)
        ax1.contour(m.grid.X[:]/1e3,m.grid.Y[:]/1e3,m.density['RHO'][k,:,:],contour_values,colors='k')#cmap='gray')
        ax1.plot([slice_x0,slice_x0+slice_length],[slice_y0-slice_dy/2,slice_y0+slice_dy/2],linewidth=2,color='b')
        ax1.quiver(m.grid.X[::q_space]/1e3,m.grid.Y[::q_space]/1e3,m.PV['J_x'][k,::q_space,::q_space],m.PV['J_y'][k,::q_space,::q_space],
                   color='k',scale=1500)
        ax1.quiver(m.grid.Y_x[::q_space,::q_space]/1e3,m.grid.X_y[::q_space,::q_space]/1e3,
                   m.zonal_velocity['UVEL'][k,::q_space,1::q_space], 
                   m.meridional_velocity['VVEL'][k,1::q_space,::q_space], color='g',scale=10)
        ax1.set_title('Vertical PV flux (Bernoulli formalism)')
        ax1.set_aspect('equal')
        ax1.axis([slice_x0,slice_x0+slice_length,slice_y0-3*slice_length/4,slice_y0+3*slice_length/4])


        sub = ax2.pcolor(m.grid.X[:]/1e3,m.grid.Zl[:],m.PV['Q'][:,j,:],cmap = 'RdBu_r',vmin=-3e-9,vmax=3e-9)
        cax,kw = matplotlib.colorbar.make_axes(ax2)
        CB = plt.colorbar(sub, cax=cax, **kw)
        CB.ax.tick_params(labelsize = 15)
        ax2.plot(m.grid.X[:]/1e3,m.grid.topog[j,:],'r',linewidth=3)
        ax2.axis([slice_x0,slice_x0+slice_length, -300,40])
        ax2.set_xlabel('Meridional distance (km)',fontsize=25)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.set_ylabel('Depth (m)',fontsize=25)

        ax2.contour(m.grid.X[:]/1e3,m.grid.Zl[:],m.density['RHO'][:,j,:],contour_values,colors='k')#cmap='gray')

        # Scalings are to fix the aspect ratio for the plot. Need to multiply the fields by what was done to the axes.
        ax2.quiver(m.grid.X[::q_space]/1e3,m.grid.Z[:], m.PV['J_x'][:,j,::q_space],m.PV['J_z'][:,j,::q_space],scale=1500)
        ax2.quiver(m.grid.X[::q_space]/1e3,m.grid.Z[:],m.zonal_velocity['UVEL'][:,j,1::q_space],
                   m.vertical_velocity['WVEL'][1:,j,::q_space]*1e3,
                   color='g',scale=1)

        ax2.set_title('WBC cross section at time = %03d days' % ((m['time'][time_level]-m['time'][0])/86400));
        fig.savefig('cross_sections/WBC %03d days' % ((m['time'][time_level]-m['time'][0])/86400),dpi = 200)
        plt.close()


    
    elapsed[time_level + 1] = time.clock()
    
    


# close the netcdf file
PV_netcdf_5_day_means.close()

