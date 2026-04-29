from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os
import shutil

env = "fishproviz/config.env"
env_default = "fishproviz/default_config.env"

if not os.path.exists(env):
    shutil.copyfile(env_default, env)


extensions = [
    Extension(
        "fishproviz.methods",
        ["fishproviz/methods.pyx"],
        include_dirs=[numpy.get_include()],
    )
]
setup(
    name="fishproviz",
    version="0.2",
    author="Luka Stärk",
    author_email="luka.staerk@mailbox.org",
    description="A package to analyze trajectories",
    ext_modules=cythonize(extensions),
    packages=find_packages(),
    include_package_data=True,
)
