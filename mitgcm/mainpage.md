@author
Ed Doddridge



##Overview##
This package provides methods and classes for analysing the output of mitgcm simulations.



##To install##
Download and install python. I'd recommend [Annaconda](https://store.continuum.io/cshop/anaconda/), it's a nice self-contained python distribution.
You'll also need the NetCDF4 module. Run 'conda install netcdf4' in the command line.

You're now ready to install this package. Navigate to the directory it is in and run:

python setup.py install


##To use##
Import the module and instantiate a simulation object:

\code{.py}
import mitgcm

m = mitgcm.MITgcm_Simulation(path_to_output,'grid.all.nc')
\endcode

##Examples##
An example ipython notebook, and the data it relies on, are in the examples folder. Alternatively, the notebook can be viewed, but not edited, [here](http://nbviewer.ipython.org/urls/bitbucket.org/edoddridge/mitgcm/raw/master/examples/example%20notebook.ipynb/%3Fat%3Dmaster).

There is also an example for the interpolation functions [here](http://nbviewer.ipython.org/urls/bitbucket.org/edoddridge/mitgcm/raw/master/examples/interpolation%20example.ipynb)



##Revision History##
January 2015:
 * Added documentation

February 2015:
 * Added streamline and streakline algorithms
 * v0.1 tagged

