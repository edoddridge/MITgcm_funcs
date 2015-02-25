#Read me#

##To install##
Download the code (either by downloading the repo or cloning it into a directory)

Navigate to the download directory and run:

python setup.py install


##To read the documentation##
Open index.html with a web browser (it's in Docs/html), or click [here](http://edoddridge.bitbucket.org/MITgcm_py/index.html) to view the online documentation.


##To use##
Import the module and instantiate a simulation object:

```
#!numpy
import mitgcm

m = mitgcm.MITgcm_Simulation(path_to_output,'grid.all.nc')
```

##An example!##
The example notebook, which can be viewed online [here](http://nbviewer.ipython.org/urls/bitbucket.org/edoddridge/mitgcm/raw/master/examples/example%20notebook.ipynb/%3Fat%3Dmaster) or found in the examples/ folder, shows how to use some of the functions in this package. There are others, some of which aren't included because they need access to large datasets, and some which aren't included because it's not (yet) a comprehensive example.