#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='Deepspeech ASR on Gaudi Processors',
    version='0.0.0',
    description='ASR made from deepspech framework for AWS Habana Gaudi Framework',
    author='',
    author_email='bencherif.research@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/PyTorchLightning/pytorch-lightning-conference-seed',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

