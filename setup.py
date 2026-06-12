"""
Build configuration for the C++ Monte Carlo simulation extension.
Compiles mc_engine.cpp and mc_bindings.cpp into a Python-importable module.

Usage:
    python setup.py build_ext --inplace
"""

import os
import sys
import platform
from setuptools import setup, Extension

# Try to import pybind11; if not available, provide a helpful message
try:
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print("ERROR: pybind11 is required to build the C++ extension.")
    print("Install it with: pip install pybind11")
    sys.exit(1)


def get_extra_compile_args():
    """Get platform-specific compiler flags for maximum performance."""
    if platform.system() == "Windows":
        return [
            "/O2",           # Maximum optimization
            "/std:c++17",    # C++17 standard
            "/openmp",       # OpenMP parallelization
            "/EHsc",         # Exception handling
            "/DNDEBUG",      # Disable debug assertions
        ]
    else:
        return [
            "-O3",               # Maximum optimization
            "-std=c++17",        # C++17 standard
            "-fopenmp",          # OpenMP parallelization
            "-march=native",     # CPU-specific optimizations
            "-DNDEBUG",          # Disable debug assertions
            "-ffast-math",       # Aggressive floating-point optimizations
        ]


def get_extra_link_args():
    """Get platform-specific linker flags."""
    if platform.system() == "Windows":
        return []
    else:
        return ["-fopenmp"]


# Define the C++ extension module
ext_modules = [
    Pybind11Extension(
        "mc_simulation",  # Module name (importable as `import mc_simulation`)
        sources=[
            os.path.join("src", "simulation", "mc_bindings.cpp"),
        ],
        include_dirs=[
            pybind11.get_include(),
            os.path.join("src", "simulation"),
        ],
        extra_compile_args=get_extra_compile_args(),
        extra_link_args=get_extra_link_args(),
        language="c++",
    ),
]

setup(
    name="wc-prediction-mc",
    version="1.0.0",
    author="WC Prediction",
    description="C++ Monte Carlo simulation engine for World Cup prediction",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)
