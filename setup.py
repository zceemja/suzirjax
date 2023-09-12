#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

with open('./requirements.txt', 'r') as f:
    install_requires = f.read().split()

with open('./README.md', 'r') as f:
    long_description = f.read()

setup(name='Suzirjax',
      version='1.0',
      description='UCL Optical Networks Group (ONG) Real-Time Constellation Shaper',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Mindaugas Jarmolovičius',
      author_email='zceemja@ucl.ac.uk',
      url='https://github.com/zceemja/suzirjax',
      install_requires=install_requires,
      python_requires='>=3.8',
      packages=find_packages(include=['suzirjax', 'suzirjax.*']),
      package_data={"suzirjax.resources": ["*"]},
      entry_points={
          'console_scripts': [
              'suzirjax=suzirjax:run_main_application',
              'suzirjax-anim=suzirjax:run_animator',
          ],
      },
)
