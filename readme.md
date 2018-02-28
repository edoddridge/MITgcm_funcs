# Read me

## To install

The code runs in Python 2.7 and uses a number of dependencies that are all satisfied by the Anaconda distribution.

The code can be installed by downloading the code (either by downloading the repo or cloning it into a directory), navigating to the download directory and running:

```
python setup.py install
```

in the terminal


## To read the documentation
Open index.html with a web browser (it's in Docs/html), or click [here](https://edoddridge.bitbucket.io/MITgcm_py/index.html) to view the online documentation.


## To use
Import the module and instantiate a simulation object:

```
#!numpy
import mitgcm

m = mitgcm.MITgcm_Simulation(path_to_output,'grid.all.nc')
```

If the simulation ran on multiple tiles, then the grid files will need to be concatenated into one file with the extremely useful 'gluemncbig' script that comes with the MITgcm in the 'utils/python/MITgcmutils/scripts' folder. The model output files don't need to be combined - the module can do that on the fly, but you do need to specify the number of tiles in the x and y directions.

## An example!
The example notebook, which can be viewed online [here](http://nbviewer.ipython.org/urls/bitbucket.org/edoddridge/mitgcm/raw/master/examples/example%20notebook.ipynb/%3Fat%3Dmaster) or found in the examples/ folder, shows how to use some of the functions in this package. 

There is also an example for the interpolation functions [here](http://nbviewer.ipython.org/urls/bitbucket.org/edoddridge/mitgcm/raw/master/examples/interpolation%20example.ipynb)
