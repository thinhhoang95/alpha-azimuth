# setup.py
from setuptools import setup, Extension
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

# to build, run: python setup.py build_ext --inplace

ext_modules = [
    Pybind11Extension(
        "conflict_checker",
        ["conflict_checker.cpp"],
        include_dirs=[pybind11.get_include()],
        cxx_std=17,
    ),
]

setup(
    name="conflict_checker",
    version="0.1",
    author="Thinh Hoang",
    description="A module to check conflicts between aircraft trajectories",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)