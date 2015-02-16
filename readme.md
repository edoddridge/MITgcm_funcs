#Read me#

##To install##
Download the code (either by downloading the repo or cloning it into a directory)

Navigate to the download directory and run:

python setup.py install


##To read the documentation##
Open index.html with a web browser (it's in Docs/html), or compile the tex documentation.


##To use##
Import the module:

import mitgcm

Instantiate a simulation object:

m = mitgcm.MITgcm_Simulation(path_to_output,'grid.all.nc')


The rest should be clear from the example.