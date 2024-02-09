"""Transforms."""

from __future__ import annotations

from typing import Callable

import torchvision.transforms as T

from storch.utils.version import is_v2_transforms_available

# we do not try, except import because v0.15.x has v2 namespace, but we don't want
# to use this version, because of some breaking changes.
if is_v2_transforms_available():
    import torchvision.transforms.v2 as Tv2
else:
    Tv2 = None

import storch


def make_simple_transform(
    image_size: tuple[int] | int,
    crop: str = 'center',
    hflip: bool = True,
    mean: tuple[float] | float = 0.5,
    std: tuple[float] | float = 0.5,
) -> Callable:
    """Make a simple transform.

    Args:
    ----
        image_size (tuple[int] | int): size of the image to be resized to. if a single int value,
            (image_size, image_size) will be used.
        crop (str, optional): 'center': CenterCrop, 'random': RandomResizedCrop. Default: 'center'.
        hflip (bool, optional): If True, add RandomHorizontalFlip. Default: True.
        mean (tuple[float] | float, optional): mean used to normalize the image. Default: 0.5.
        std (tuple[float] | float, optional): std used to normalize the image. Default: 0.5.

    Returns:
    -------
        Callable: transforms

    """
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    if crop == 'random':
        transform = [T.RandomResizeCrop(image_size)]
    elif crop == 'center':
        transform = [T.Resize(max(image_size)), T.CenterCrop(image_size)]

    if hflip:
        transform.append(T.RandomHorizontalFlip())

    transform.extend([T.ToTensor(), T.Normalize(mean, std)])

    return T.Compose(transform)


def build_transform(name: str, **params) -> Callable:
    """Build transform given a name and parameters.

    Build a transform inside torchvision.transforms by their class name. If torchvision's v2 namespace is available,
    the v2 transforms are used preferentially. If you want a python object as the value, pass a string with
    `pyobj:` prefix (e.g., `'pyobj:torch.float32'` to pass `torch.float32`).

    Args:
    ----
        name (str): The name of the transform. ex) ToTenor, Normalize
        **params: Keyword arguments of passed to the transform.

    Returns:
    -------
        Callable: the built transform

    """
    # convert string to a python object if value starts with 'pyobj:' prefix.
    transform_kwargs = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith('pyobj:'):
            value = storch.get_obj_by_name(value.replace('pyobj:', ''))  # noqa: PLW2901
        transform_kwargs[key] = value

    if Tv2 is not None and hasattr(Tv2, name):
        return getattr(Tv2, name)(**transform_kwargs)
    if hasattr(T, name):
        return getattr(T, name)(**transform_kwargs)
    else:
        transform = storch.construct_class_by_name(class_name=name, **transform_kwargs)
        assert callable(transform), f'User defined tranform {name} is not callable'
        return transform


def make_transform_from_config(configs: list[dict]) -> Callable:
    """Make transform from list of TransformConfigs.

    Args:
    ----
        configs (list[dict]): List of dicts containing a least 'name' key.

    Returns:
    -------
        Callable: transforms

    Examples::
        >>> transforms = make_transform_from_config(
        >>>     [
        >>>         {'name': 'Resize', 'size': (224, 224)},
        >>>         {'name': 'ToTensor'},
        >>>         {'name': 'Normalize', 'mean': 0.5, 'std': 0.5}
        >>>     ]
        >>> )

    """
    transform = []
    for config in configs:
        if config.get('name') in ['RandomChoice', 'RandomOrder', 'RandomApply']:
            inner_transform = []
            for inner_config in config.get('transforms'):
                inner_transform.append(build_transform(**inner_config))
            params = dict(transforms=inner_transform)
            if config.get('p', None) is not None:
                params.update(dict(p=config.get('p')))
            transform.append(build_transform(config.get('name'), **params))
        else:
            transform.append(build_transform(**config))
    if is_v2_transforms_available():
        return Tv2.Compose(transform)
    return T.Compose(transform)


def make_cutmix_or_mixup(
    mixup_alpha: float = 1.0,
    cutmix_alpha: float = 0.0,
    prob: float = 1.0,
    switch_prob: float = 0.5,
    num_classes=1000,
    labels_getter='default',
) -> Callable:
    """Mixup or CutMix that uses the torchvision implementation.

    The arguments of this function acts equivalently to timm's mixup transform implementation.
    If `prob=0.0` or the alphas are both set to 0.0, returns a function that returns the input as-is.

    Args:
    ----
        mixup_alpha (float, optional): alpha for MixUp. usually 1.0 is used. disabled if 0.0. Default: 1.0.
        cutmix_alpha (float, optional): alpha for CutMix. usually 1.0 is used. disabled if 0.0. Default: 0.0.
        prob (float, optional): probability to apply cutmix or mixup. Default: 1.0.
        switch_prob (float, optional): probability to switch between cutmix or mixup. Default: 0.5.
        num_classes (int, optional): number of classes. Default: 1000.
        labels_getter (str, optional): See `torchvision.transforms.v2.{MixUp,CutMix}`. Default: 'default'.

    Raises:
    ------
        Exception: torchvision version does not suport CutMix and MixUp.

    Returns:
    -------
        Callable: _description_

    """
    if not is_v2_transforms_available():
        # mixup and cutmix was added at `0.16.0`
        raise Exception(
            'This function uses the `torchvision` implementation of MixUp and CutMix which was added'
            + 'in version `0.16.0`.'
        )

    def _noops(*args):
        return args

    mixup, cutmix = None, None

    if mixup_alpha > 0.0:  # noqa: PLR2004
        mixup = Tv2.MixUp(alpha=mixup_alpha, num_classes=num_classes, labels_getter=labels_getter)
    if cutmix_alpha > 0.0:  # noqa: PLR2004
        cutmix = Tv2.CutMix(alpha=cutmix_alpha, num_classes=num_classes, labels_getter=labels_getter)

    cutmix_or_mixup = None
    if cutmix is not None and mixup is not None:
        assert 0 < switch_prob < 1.0, '`switch_prob` should be in (0, 1).'  # noqa: PLR2004
        cutmix_or_mixup = Tv2.RandomChoice([cutmix, mixup], p=[switch_prob, 1 - switch_prob])
    elif cutmix is not None or mixup is not None:
        cutmix_or_mixup = cutmix if cutmix is not None else mixup
    else:
        return _noops

    if prob != 1.0:  # noqa: PLR2004
        cutmix_or_mixup = Tv2.RandomApply([cutmix_or_mixup], p=prob)
    elif prob == 0.0:  # noqa: PLR2004
        return _noops

    return cutmix_or_mixup
