#!/usr/bin/env python

import sys, os
from glob import glob

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from distutils.command.install import INSTALL_SCHEMES
from distutils.command.install_headers import install_headers
from setuptools import find_packages
from setuptools import setup

NAME =               'neurokernel'
VERSION =            '0.1'
AUTHOR =             'Neurokernel Development Team'
AUTHOR_EMAIL =       'neurokernel-dev@columbia.edu'
URL =                'https://github.com/neurokernel/neurokernel/'
MAINTAINER =         AUTHOR
MAINTAINER_EMAIL =   AUTHOR_EMAIL
DESCRIPTION =        'An open architecture for Drosophila brain emulation'
LONG_DESCRIPTION =   DESCRIPTION
DOWNLOAD_URL =       URL
LICENSE =            'BSD'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']
NAMESPACE_PACKAGES = ['neurokernel']
PACKAGES =           find_packages()

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    # This enables the installation of neurokernel/__init__.py as a data
    # file:
    for scheme in INSTALL_SCHEMES.values():
        scheme['data'] = scheme['purelib']

    setup(
        name = NAME,
        version = VERSION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        classifiers = CLASSIFIERS,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        url = URL,
        maintainer = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        namespace_packages = NAMESPACE_PACKAGES,
        packages = PACKAGES,

        # Force installation of __init__.py in namespace package:
        data_files = [('neurokernel', ['neurokernel/__init__.py'])],
        include_package_data = True,
        install_requires = [
            'bidict',
            'bottleneck',
            'futures',
            'h5py',
            'matplotlib',
            'msgpack_numpy',
            'networkx',
            'numexpr',
            'numpy',
            'pycuda >= 2012.1',
            'pyzmq',
            'scipy',
            'tables',
            'twiggy',
        ],
        )
