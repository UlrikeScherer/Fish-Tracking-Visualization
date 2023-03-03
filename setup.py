from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os, shutil

env = "fishproviz/config.env"
env_default = "scripts/env.default.sh"

if not os.path.exists(env):
    shutil.copyfile(env_default, env)


extensions = [
    Extension(
        "fishproviz.methods",
        ["fishproviz/methods.pyx"],
        include_dirs=[numpy.get_include()],
        # define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    )
]
setup(
    name="fishproviz",
    version="0.1",
    author="Luka Stärk",
    author_email="luka.staerk@mailbox.org",
    description="A package to analyze trajectories",
    ext_modules=cythonize(extensions),
    #install_requires=['requirement'],
    packages=[*find_packages()], #["fishproviz", "fishproviz.utils", "fishproviz.metrics","fishproviz.trajectory"],
    include_package_data=True
    )
