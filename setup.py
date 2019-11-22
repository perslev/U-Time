try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from utime import __version__

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

with open("requirements.txt") as req_file:
    requirements = list(filter(None, req_file.read().split("\n")))

setup(
    name='utime',
    version=__version__,
    description='A deep learning framework for automatic PSG sleep analysis.',
    long_description=readme + "\n\n" + history,
    author='Mathias Perslev',
    author_email='map@di.ku.dk',
    url='https://github.com/perslev/U-Time',
    license="LICENSE.txt",
    packages=["utime"],
    package_dir={'utime':
                 'utime'},
    entry_points={
       'console_scripts': [
           'ut=utime.bin.ut:entry_func',
       ],
    },
    install_requires=requirements,
    classifiers=['Environment :: Console',
                 'Operating System :: POSIX',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7'
                 'License :: OSI Approved :: MIT License']
)
