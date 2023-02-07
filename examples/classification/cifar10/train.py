
import datetime
import sys
import time

import numpy as np
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10

import storch
from storch.checkpoint import Checkpoint
from storch.dataset import make_transform_from_config
from storch.distributed import DistributedHelper
from storch.hydra_utils import get_hydra_config, save_hydra_config
from storch.metrics import BestStateKeeper, test_classification
from storch.path import Folder, Path
from storch.status import Status, ThinStatus
from storch.torchops import (inference_mode,
                             optimizer_step_with_gradient_accumulation)


def download_dataset(root='./cifar10'):
    CIFAR10(root, download=True)

def setup_run():
    cmdargs = sys.argv[1:]
    # input saved config.yaml for resuming.
    if len(cmdargs) == 1 and cmdargs[0].endswith('config.yaml'):
        config = OmegaConf.load(cmdargs[0])
        root_folder = Path(cmdargs[0]).dirname()
        assert root_folder.exists()
        folder = Folder(root_folder)
    # if not, load config.yaml.
    else:
        config = get_hydra_config('config', 'config.yaml')
        folder = Folder(Path(config.run.folder) / config.run.name)
        folder.mkdir()
        save_hydra_config(config, folder.root / 'config.yaml')
    return config, folder


def get_datasets(train_transform_config, test_transform_config, validation_size=5000, root='./cifar10'):
    train_transform = make_transform_from_config(train_transform_config)
    test_transform = make_transform_from_config(test_transform_config)
    train_dataset = CIFAR10(root, train=True, transform=train_transform)
    train_size = len(train_dataset) - validation_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, validation_size])
    test_dataset = CIFAR10(root, train=False, transform=test_transform)
    return train_dataset, val_dataset, test_dataset


def main():
    disthelper = DistributedHelper()
    start_time = time.perf_counter()

    cfg, folder = setup_run()

    if disthelper.is_primary():
        download_dataset(cfg.data.root)

    device = disthelper.device

    # datasets
    train_dataset, val_dataset, test_dataset = get_datasets(
        cfg.data.transforms.train, cfg.data.transforms.test,
        cfg.data.validation_size, cfg.data.root
    )
    train_dataset = disthelper.prepare_dataset(
        train_dataset, cfg.data.loader.batch_size, True, cfg.data.loader.drop_last,
        cfg.data.loader.num_workers, cfg.data.loader.pin_memory
    )
    val_dataset = disthelper.prepare_dataset(
        val_dataset, cfg.data.loader.batch_size, False, False
    )
    test_dataset = disthelper.prepare_dataset(
        test_dataset, 1, False, False
    )

    # model
    model = storch.construct_class_by_name(**cfg.model)
    model = disthelper.prepare_module(model, mode=cfg.distributed.mode, mixed_precision=cfg.amp)

    # optimizer
    optimizer = storch.construct_class_by_name(model.parameters(), **cfg.optimizer)

    # criterion
    criterion = storch.construct_class_by_name(**cfg.criterion)

    # gradscaler
    scaler = None
    if cfg.amp:
        scaler = ShardedGradScaler() if cfg.distributed.mode == 'fsdp' else GradScaler()

    # logging
    StatusCls = Status if disthelper.is_primary() else ThinStatus
    status = StatusCls(cfg.train.epochs*len(train_dataset), folder.root / cfg.run.log_file, False,
        cfg.run.log_interval, cfg.run.name)
    status.log_stuff(cfg, model, optimizer, train_dataset)

    model_ckpt, optim_ckpt = disthelper.prepare_for_checkpointing(optimizer, offload_to_cpu=cfg.train.offload_ckpt_to_cpu)

    # best model
    best_model = BestStateKeeper(
        'best-val-loss', 'min', model_ckpt, folder.root, disthelper=disthelper
    )

    # checkpointing
    checkpoint = Checkpoint(folder.root, cfg.train.keep_last_n_ckpt, disthelper=disthelper)
    checkpoint.register(model=model_ckpt, optimizer=optim_ckpt, status=status, best_model=best_model)
    if scaler is not None: checkpoint.register(scaler=scaler)
    checkpoint.load_latest()

    optimizer_step = optimizer_step_with_gradient_accumulation(
        cfg.train.grad_accum_steps, len(train_dataset), model.no_sync if hasattr(model, 'no_sync') else None
    )

    epoch = 0
    while not status.is_end():
        if disthelper.is_initialized():
            train_dataset.sampler.set_epoch(epoch)
            epoch += 1

        model.train()
        num_images, epoch_correct, epoch_loss = 0, 0, 0
        for images, labels in train_dataset:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            batch_loss = criterion(outputs, labels)

            optimizer_step(
                batch_loss, optimizer, scaler, module=model,
                zero_grad=True, set_to_none=True, clip_grad_norm=cfg.train.clip_grad_norm,
                max_norm=cfg.train.max_norm, update_scaler=True
            )

            batch_size = images.size(0)
            batch_correct = (outputs.max(1)[1] == labels).sum().item()
            num_images += batch_size
            epoch_correct += batch_correct
            epoch_loss += batch_loss * batch_size

            status.update(**{
                'Loss/CE/train/batch': batch_loss,
                'Metrics/Accuracy/train/batch': batch_correct / batch_size
            })

        num_images, epoch_correct, epoch_loss = disthelper.reduce([num_images, epoch_correct, epoch_loss]).tolist()
        epoch_train_accuracy = epoch_correct / num_images
        epoch_train_loss = epoch_loss / num_images

        model.eval()
        num_images, epoch_correct, epoch_loss = 0, 0, 0
        with inference_mode():
            for images, labels in val_dataset:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                batch_loss = criterion(outputs, labels)

                batch_size = images.size(0)
                batch_correct = (outputs.max(1)[1] == labels).sum().item()
                num_images += batch_size
                epoch_correct += batch_correct
                epoch_loss += batch_loss * batch_size

        num_images, epoch_correct, epoch_loss = disthelper.reduce([num_images, epoch_correct, epoch_loss]).tolist()
        val_accuracy = epoch_correct / num_images
        val_loss = epoch_loss / num_images

        status.dry_update(**{
            'Loss/CE/train': epoch_train_loss,
            'Loss/CE/val': val_loss,
            'Metrics/Accuracy/train': epoch_train_accuracy,
            'Metrics/Accuracy/val': val_accuracy
        })

        best_model.update(val_loss, step=status.batches_done)
        checkpoint.save()

    best_model.load()
    model.eval()
    model.requires_grad_(False)
    predictions, true_labels = [], []
    with inference_mode():
        for images, labels in test_dataset:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions.extend(outputs.max(1)[1].view(-1).tolist())
            true_labels.extend(labels.view(-1).tolist())
    predictions = disthelper.gather(predictions).cpu().numpy()
    true_labels = disthelper.gather(true_labels).cpu().numpy()

    if disthelper.is_primary():
        test_classification(
            predictions, true_labels, np.asarray(test_dataset.dataset.classes),
            folder.root / 'confmat', status.log
        )

    disthelper.wait_for_all_processes()
    duration = time.perf_counter() - start_time
    status.log(f'Training duration on main process: {datetime.timedelta(seconds=duration)}')

if __name__=='__main__':
    main()
