from setuptools import setup

setup(name='mitgcm',
      version='0.1',
      description='Load, explore and analyse MITgcm model output',
      url='http://bitbucket.com/edoddridge/mitgcm',
      author='Ed Doddridge',
      author_email='blank',
      license='MIT licence',
      packages=['mitgcm'],
      install_requires=[
          'numpy',
	  'netCDF4',
	  'numba',
        'scipy',
        'matplotlib',
              ],
      zip_safe=False)
