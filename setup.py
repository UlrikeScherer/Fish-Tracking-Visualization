from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions=[
        Extension("methods", ["src/methods.pyx"],
                  include_dirs=[numpy.get_include()]),
    ]
setup(
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
