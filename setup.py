# pylint: disable=C0111,C0103

import os
import pathlib
import subprocess

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

packages = find_packages(exclude=['tests'])
base_dir = pathlib.Path(__file__).parent

requirements = ["numpy>=1.0",
                "gym=*",
                "hip-mdp-public=https://github.com/Zaerei/hip-mdp-public",
                "llvmlite==0.27.0"
                "numba==0.42.0"]

with open(os.path.join(str(base_dir), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gym_decomp',
    use_scm_version=True,
    long_description='\n' + long_description,
    packages=packages,
    setup_requires=['setuptools_scm'],
)
