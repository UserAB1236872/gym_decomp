# pylint: disable=C0111,C0103

import os
import pathlib

from setuptools import setup, find_namespace_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

packages = find_namespace_packages(exclude=['tests'])
base_dir = pathlib.Path(__file__).parent

requirements = ["numpy>=1.0",
                "gym==0.12.5",
                "hip-mdp-public @ https://github.com/Zaerei/hip-mdp-public/archive/master.zip",
                "llvmlite==0.27.0",
                "numba==0.42.0",
                "pylzma>=0.5"]

with open(os.path.join(str(base_dir), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gym_decomp',
    use_scm_version=False,
    long_description='\n' + long_description,
    packages=packages,
    install_requires=requirements,
    setup_requires=['setuptools_scm'],
    extras_require={
        'dev': [
            'pytest>=4.0',
            'pylint>=2',
            'autopep8>=0'
        ]
    }
)
