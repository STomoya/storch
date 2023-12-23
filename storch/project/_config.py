from omegaconf import OmegaConf


def get_default_config():
    config_dict = dict(
        defaults=['_self_'],
        env=dict(device='cuda', amp=False, compile=False),
        reproduce=dict(
            enabled=True, params=dict(seed=3407, use_deterministic_algorithm=True, warn_only=True, cudnn_benchmark=True)
        ),
        run=dict(folder='checkpoint', name='???', tag='date'),
        config=dict(
            data=dict(
                root='./data',
                image_size=224,
                transforms=dict(
                    train=[
                        dict(
                            name='RandomResizedCrop',
                            scale=[1.67, 1.0],
                            ratio=[3 / 4, 4 / 3],
                            size=['${config.data.image_size}', '${config.data.image_size}'],
                        ),
                        dict(name='RandomHorizontalFlip', p=0.5),
                        dict(name='ToTensor'),
                        dict(name='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ],
                    test=[
                        dict(name='Resize', size=['${config.data.image_size}', '${config.data.image_size}']),
                        dict(name='ToTensor'),
                        dict(name='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ],
                ),
                loader=dict(
                    train=dict(batch_size=32, drop_last=True, shuffle=True, num_workers=4, pin_memory=True),
                    val=dict(batch_size=32, drop_last=False, shuffle=False, num_workers=4, pin_memory=True),
                    test=dict(batch_size=1, drop_last=False, shuffle=False, num_workers=4, pin_memory=True),
                ),
            ),
            checkpoint=dict(log_file='log.log', log_interval=100, save_every=1, keep_last=1),
            model=dict(class_name='???'),
            train=dict(epochs=100, grad_accum_steps=2, clip_grad_norm=True, max_norm=5.0),
            optimizer=dict(class_name='torch.optim.AdamW', lr=0.001, betas=[0.9, 0.999], weight_decay=0.05),
            scheduler=dict(
                type='cosine', num_training_steps='${config.train.epochs}', num_iter_per_step=1, num_warmup_steps=5
            ),
        ),
    )
    return OmegaConf.create(config_dict)
