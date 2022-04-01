
from setuptools import setup, find_packages

setup(
    name='storch',
    version='0.0.3',
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
        'scikit-image'
    ]
)
