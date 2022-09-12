
from setuptools import find_packages, setup

setup(
    name='storch',
    version='0.2.2',
    license='MIT',
    description='PyTorch utilities for STomoya.',
    author='Tomoya Sawada (STomoya)',
    author_email='stomoya0110@gmail.com',
    url='https://github.com/STomoya/storch/',
    packages=find_packages(),
    install_requires=[
        'tqdm',
        'numpy',
        'opencv-python',
        'pillow',
        'matplotlib',
        'scikit-learn',
        'scikit-image',
        'hydra-core',
        'tensorboard'
    ]
)
