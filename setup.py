"""
Minimal setup.py retained solely for:
  1. The config.env bootstrap (pyproject.toml has no hook for arbitrary file copies).
  2. The Cython/numpy extension — numpy.get_include() must be resolved at build time
     and cannot be expressed as a static path in pyproject.toml.

All project metadata (name, version, author, dependencies) lives in pyproject.toml.
"""

import os
import shutil

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

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

setup(ext_modules=cythonize(extensions))
