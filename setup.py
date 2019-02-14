# pylint: disable=C0111,C0103

import os
import pathlib
import subprocess

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

packages = find_packages(exclude=['tests'])
base_dir = pathlib.Path(__file__).parent

install_requires = ['pipenv']

requirements = convert_deps_to_pip(pfile['packages'], r=False)
requirements.append('pipenv')

test_requirements = convert_deps_to_pip(pfile['dev-packages'], r=False)
test_requirements.append('pipenv')

with open(os.path.join(str(base_dir), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gym_decomp',
    use_scm_version=True,
    long_description='\n' + long_description,
    packages=packages,
    setup_requires=['setuptools_scm'],
)
