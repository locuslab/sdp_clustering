import os
import sys
DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(DIR, "third_party", "pybind11"))
print(sys.path)

from glob import glob
from setuptools import setup
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext # noqa: E402

del sys.path[-1]

pkg_name = 'sdp_clustering'
ext_name = '_cpp'
__version__ = "0.0.3"

ext_modules = [
    Pybind11Extension(pkg_name+'.'+ext_name,
        include_dirs = ['./src'],
        sources = sorted(glob('src/*.c*')),
        define_macros = [('EXTENSION_NAME', ext_name)],
		extra_compile_args = ['-O3', '-Wall', '-g'],
        ),
]

setup(
    name=pkg_name,
    version=__version__,
    install_requires=['numpy', 'scipy'],
    author="Po-Wei Wang",
    author_email="poweiw@cs.cmu.edu",
    url="https://github.com/locuslab/sdp_clustering",
    description="SDP-based clustering by maximum modularity",
    long_description="",
    scripts=['bin/locale_alg'],
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    packages=[pkg_name],
    zip_safe=False,
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
)
