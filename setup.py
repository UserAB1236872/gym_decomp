# pylint: disable=C0111,C0103

import os
import pathlib
import subprocess

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

__requires__ = ['pipenv']

packages = find_packages(exclude=['tests'])
base_dir = pathlib.Path(__file__).parent

pipenv_command = ['pipenv', 'install', '--deploy', '--system']
pipenv_command_dev = ['pipenv', 'install', '--dev', '--deploy', '--system']


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        subprocess.check_call(pipenv_command_dev)
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        subprocess.check_call(pipenv_command)
        install.run(self)


with open(os.path.join(str(base_dir), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gym_decomp',
    use_scm_version=True,
    long_description='\n' + long_description,
    packages=packages,
    setup_requires=['setuptools_scm'],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
