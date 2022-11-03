
from setuptools import find_packages, setup

with open('./storch/version.py', 'r') as fp:
    versionpy = fp.read().strip()
version = versionpy[versionpy.find('=')+1:].replace("'", '')

setup(
    name='storch',
    version=version,
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
        'tensorboard',
        'stutil@git+https://github.com/STomoya/stutil@v0.0.2'
    ]
)
