
import storch

__text_length = 62

ASCII_LOGO = r'''
         ______   ______  ______   ______   ______   __  __
        /\  ___\ /\__  _\/\  __ \ /\  == \ /\  ___\ /\ \_\ \
 )\_/(  \ \___  \\/_/\ \/\ \ \/\ \\ \  __< \ \ \____\ \  __ \
='o.o'=  \/\_____\  \ \_\ \ \_____\\ \_\ \_\\ \_____\\ \_\ \_\
 (_ _)    \/_____/   \/_/  \/_____/ \/_/ /_/ \/_____/ \/_/\/_/
    U                    version: {version}
'''.format(version=storch.__version__)

DESCRIPTION = 'pyTORCH utilities for STomoya (storch)'

HEADER = f'{ASCII_LOGO}\n{DESCRIPTION.center(__text_length)}'

def _collect_versions(num_cols=2):
    import cv2
    import hydra
    import matplotlib
    import numpy
    import PIL
    import skimage
    import sklearn
    import tensorboard
    import torch
    import torchvision
    import tqdm

    version_strings = storch.EasyDict()
    version_strings.torch = torch.__version__
    version_strings.torchvision = torchvision.__version__
    version_strings.cv2 = cv2.__version__
    version_strings.hydra = hydra.__version__
    version_strings.matplotlib = matplotlib.__version__
    version_strings.numpy = numpy.__version__
    version_strings.pillow = PIL.__version__
    version_strings.skimage = skimage.__version__
    version_strings.sklearn = sklearn.__version__
    version_strings.tensorboard = tensorboard.__version__
    version_strings.tqdm = tqdm.__version__

    dependency_versions = []
    temp = []
    for index, (module, version) in enumerate(version_strings.items(), 1):
        temp.append(f'|{module:>15} | {version:<10}|')
        if index % num_cols == 0:
            dependency_versions.append(' '.join(temp).center(__text_length))
            temp = []
    return '\n'.join(dependency_versions)


def get_detailed_header():
    dependency_versions = 'Dependency versions'.center(__text_length) + '\n' + _collect_versions(2)
    urls = 'GitHub: https://github.com/STomoya/storch'.center(__text_length)

    detailed_header = '\n\n'.join([HEADER, dependency_versions, urls])

    return detailed_header
