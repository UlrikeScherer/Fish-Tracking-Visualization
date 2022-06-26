from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os, shutil
env = 'scripts/env.sh'
env_default = 'scripts/env.default.sh'
extensions=[
        Extension("src.methods", ["src/methods.pyx"],
                  include_dirs=[numpy.get_include()],
                  #define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
                  )
    ]
setup(
    name='trajectory analysis',
    version='0.1',
    author="Luka Stärk",
    author_email="luka.staerk@mailbox.org",
    description="A package to analyze trajectories",
    ext_modules=cythonize(extensions),
    packages=['src'],
)

if not os.path.exists(env):
    shutil.copyfile(env_default, env)
