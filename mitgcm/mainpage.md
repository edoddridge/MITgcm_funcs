@author
Ed Doddridge



##Overview##
This package provides methods and classes for analysing the output of mitgcm simulations.



##To install##
Navigate to the main directory and run:

python setup.py install


##To read the documentation##
Open index.html with a web browser (it's in Docs/html), or compile the tex documentation. From there, you should be able to navigate it on your own.


##To use##
Import the module:

import mitgcm

Instantiate a simulation object:

m = mitgcm.MITgcm_Simulation(path_to_output,'grid.all.nc')



##Revision History##
January 2015:
Added documentation

February 2015:
Added streamline and streakline algorithms
