import torch.cuda

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

ext_modules = [
    CppExtension(
        name = 'cluster._cpp',
        include_dirs = ['./src'],
        sources = [
            'src/cluster.cpp',
            'src/cluster_cpu.cpp',
        ],
        extra_compile_args = ['-msse4.1', '-Wall', '-g']#, '-fsanitize=address']
    )
]

#with open("README.md", "r", encoding="utf-8") as fh:
#    long_description = fh.read()

# Python interface
setup(
    name='cluster',
    version='0.1.0',
    install_requires=['torch>=1.0'],
    packages=['cluster'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    author='anom',
    author_email='anom@abc.com',
    zip_safe=False,
    description='anom',
    #long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
)
