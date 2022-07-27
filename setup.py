from setuptools import setup, Extension
import pybind11

functions_module = Extension(
    name='list2img',
    sources=['list2img.cpp'],
    include_dirs=[pybind11.get_include()],
)

setup(ext_modules=[functions_module])