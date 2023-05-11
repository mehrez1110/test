from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("my_module.pyx", language="c", compiler="gcc")
)
