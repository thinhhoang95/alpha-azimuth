# setup.py
from setuptools import setup, Extension
import sys
import setuptools
import pybind11
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

import sys
import sysconfig

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path."""

    def __str__(self):
        return pybind11.get_include()

ext_modules = [
    Extension(
        'conflict_checker',
        ['conflict_checker.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
        ],
        language='c++',
        extra_compile_args=['-O3', '-std=c++17', '-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name='conflict_checker',
    version='0.1',
    author='Thinh Hoang',
    description='A C++ optimized conflict checker with Python bindings',
    ext_modules=ext_modules,
    install_requires=['pybind11'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
