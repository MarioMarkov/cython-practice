from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("tensor", ["tensor.pyx"],include_dirs=[np.get_include()],  # Include NumPy headers
)
]

setup(
    ext_modules=cythonize(extensions, annotate = True )
)