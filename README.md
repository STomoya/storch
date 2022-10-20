
# pyTORCH utilities for STomoya (storch)

## Install

This module requires `torch` and `torchvision`, but are not included as dependencies because a specific version of them should be installed by the user, which fits their environment.

NOTE: PyTorch should be `>=1.6.0` to use codes around AMP.

```console
pip install git+https://github.com/STomoya/storch
```

## Design

🐈 It aims to:

- Integrate well with user-defined training loops.

Defining and configuration of training loops and models are fully up to the user. It should work well with existing user-defined training loops with some modifications.

👿 Conversely it doesn't:

- No pre-defined training loops.

No classes like `Trainer` will be implemented. (At least, until I find a way that makes these fully customizable + easily configured.)

## Version

Until `v0.2.13`, the version was bumped on every merge from a pull request. From `v0.3.0`, this module relies on [`stutil`](https://github.com/STomoya/stutil), and when any version updates on `stutil`, `storch` version will also be updated.

## License

[MIT License](./LICENSE)

## Author

[Tomoya Sawada](https://github.com/STomoya/)
