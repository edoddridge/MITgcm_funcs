@author
Ed Doddridge



##Overview##
This package provides methods and classes for analysing the output of mitgcm simulations.



##To install##
Download and install python. I'd recommend [Annaconda](https://store.continuum.io/cshop/anaconda/), it's a nice self-contained python distribution.


The code can be installed through conda using

\code
conda install -c https://conda.anaconda.org/edoddridge mitgcm
\endcode

Or, it can be installed by:

Downloading the code (either by downloading the repo or cloning it into a directory)

Navigating to the download directory and running:

\code
python setup.py install
\endcode

in the terminal


###To install on ARCHER###
The way that python is setup on ARCHER makes this a little trickier.

Create a conda environment called ‘VENV’ with base python installed inside it

\code
conda create --  name VENV python
\endcode

activate this environment with

\code
source activate VENV
\endcode

remove the PYTHONPATH and PYTHONHOME variables

\code
unset PYTHONPATH
unset PYTHONHOME

conda install -c https://conda.anaconda.org/edoddridge mitgcm
\endcode

If you want to use ipython notebooks, you’ll also need to run

\code
conda install jupyter
\endcode


It's also worth adding 
\code 
source activate VENV
unset PYTHONPATH
unset PYTHONHOME
\endcode

to your .bashrc file, so that you don't have to manually do this everytime you log in.

##To use##
Import the module and instantiate a simulation object:

\code{.py}
import mitgcm

m = mitgcm.MITgcm_Simulation(path_to_output,'grid.all.nc')
\endcode

##Examples##
An example ipython notebook, and the data it relies on, are in the examples folder. Alternatively, the notebook can be viewed, but not edited, [here](http://nbviewer.ipython.org/urls/bitbucket.org/edoddridge/mitgcm/raw/master/examples/example%20notebook.ipynb/%3Fat%3Dmaster).

There is also an example for the interpolation functions [here](http://nbviewer.ipython.org/urls/bitbucket.org/edoddridge/mitgcm/raw/master/examples/interpolation%20example.ipynb)


