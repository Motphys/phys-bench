"""Setup script for isaacgym"""

import sys
import os

from setuptools import setup, find_packages, Distribution

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(self):
        return True

def collect_files(target_dir):
    file_list = []
    for (root, dirs, files) in os.walk(target_dir,followlinks=True):
        for filename in files:
            file_list.append(os.path.join('..', root, filename))
    return file_list

def _do_setup():
    root_dir = os.path.dirname(os.path.realpath(__file__))

    packages = find_packages(".")
    print(packages)

    #
    # TODO: do something more clever to collect only the bindings for the active versions of Python
    #

    package_files = []
    if sys.platform.startswith("win"):
        package_files = package_files + collect_files("isaacgym/_bindings/windows-x86_64")
    elif sys.platform.startswith("linux"):
        package_files = package_files + collect_files("isaacgym/_bindings/linux-x86_64")

    setup(packages=packages,
          package_data={
              "isaacgym": package_files
          },
          distclass=BinaryDistribution,
          python_requires='>=3.8,<3.9',
          install_requires = [
          ],
         )

_do_setup()
