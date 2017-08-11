from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extra_compile_args = [
    '-Ofast',
    '-march=native',
    '-ffast-math',
    '-funroll-loops',
    '-g',
    '-fno-omit-frame-pointer',
]

extensions = [
    Extension(
        name="*",
        sources=["MrWSI/core/*.pyx"],
        extra_compile_args=extra_compile_args,
        libraries=["mrwsi"],
        library_dirs=["./MrWSI/core"],
        include_dirs=["./MrWSI/core/include"]),
]

setup(ext_modules=cythonize(
    extensions,
    compiler_directives={
        "profile": False,
        "cdivision": True,
        "boundscheck": False,
        "initializedcheck": False,
    }))
