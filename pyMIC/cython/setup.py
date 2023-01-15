from setuptools import setup
from Cython.Build import cythonize
import numpy  #necessary for build up

setup(
    ext_modules = cythonize("Stokes_continuity_markers_cy_scipy.pyx",annotate=True),
    include_dirs=[numpy.get_include()]
)


