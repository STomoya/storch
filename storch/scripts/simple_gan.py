
from __future__ import annotations
from copy import deepcopy

from dataclasses import dataclass
import os
from typing import Callable
import torch
import torch.optim as optim

import storch
from storch.dataset.dataset import DatasetBase
from storch.status import Status
from storch.accelerate import MiniAccelerator
from storch.imageops import save_image
from storch.torchops import freeze, update_ema
import storch.loss as loss

def get_loss_fns(adv_type):
    gp_fn = loss.r1_regularizer()
    if adv_type == 'sn':
        adv_fn = loss.NonSaturatingLoss()
    if adv_type == 'gan':
        adv_fn = loss.GANLoss()
    if adv_type == 'ls':
        adv_fn = loss.LSGANLoss()
    if adv_type == 'hinge':
        adv_fn = loss.HingeLoss()
    return adv_fn, gp_fn

@dataclass
class Config:
    '''Config for training GAN'''
    G: torch.nn.Module                  # Generator
    D: torch.nn.Module                  # Discriminator
    dataset: DatasetBase                # dataset
    input_size: tuple                   # size of input tensor to G
    iterations: int=200000              # number of iterations to train
    adv_type: str='ns'                  # adversarial loss type
    g_lr: float=0.0002                  # learning rate for G
    d_lr: float=0.0002                  # learning rate for D
    betas: tuple=(0.5, 0.999)           # betas for Adam
    gp_every: int=1                     # calc gradient penalty every
    gp_lambda: float=0.                 # lambda for gradient penalty
    ema: bool=True                      # use EMA model
    ema_decay: float=0.999              # decay for EMA model
    augment: Callable=lambda x:x        # differentiable augmentation
    update_augment: Callable=None       # function to update augmentation which requires two arguments:
                                        #   the augmentation function, step_info class object
    amp: bool=True                      # use AMP
    num_test: int=16                    # number of samples for eval
    save: int=1000                      # save test and state dict every
    checkpoint_folder: str='checkpoint' # folder for saving stuff
    log_file: str='log.log'             # filename for logger
    log_interval: int=10                # interval for logging
    running: int=-999                   # if >0 save fake image very

@dataclass
class step_info:
    real_prob: torch.Tensor
    fake_prob: torch.Tensor

def train_fn(cfg: Config):

    storch.check_folder(cfg.checkpoint_folder)
    storch.check_folder(os.path.join(cfg.checkpoint_folder, 'images'))
    storch.check_folder(os.path.join(cfg.checkpoint_folder, 'models'))

    status = Status(cfg.iterations, False, os.path.join(cfg.checkpoint_folder, cfg.log_file),
        cfg.log_interval, __name__)
    status.log_training(cfg)

    G_ema = deepcopy(cfg.G) if cfg.ema else None
    if cfg.ema: freeze(G_ema)
    optim_G = optim.Adam(cfg.G.parameters(), lr=cfg.g_lr, betas=cfg.betas)
    optim_D = optim.Adam(cfg.D.parameters(), lr=cfg.g_lr, betas=cfg.betas)

    accelerator = MiniAccelerator(cfg.amp)
    G, G_ema, D, dataset, optim_G, optim_D, augment = accelerator.prepare(
        cfg.G, G_ema, cfg.D, cfg.dataset, optim_G, optim_D, cfg.augment)
    const_z = torch.randn((cfg.num_test, *cfg.input_size), device=accelerator.device)

    adv_fn, gp_fn = get_loss_fns(cfg.adv_type)

    while not status.is_end():
        for real in dataset:
            z = torch.randn((real.size(0), *cfg.input_size), device=accelerator.device)

            with accelerator.autocast():
                # G forward
                fake = G(z)

                # augment
                real_aug = augment(real)
                fake_aug = augment(fake)

                # D forward (SG)
                real_prob = D(real_aug)
                fake_prob = D(fake_aug.detach())

                # loss
                adv_loss = adv_fn.d_loss(real_prob, fake_prob)
                gp_loss = 0
                if cfg.gp_lambda > 0:
                    gp_loss = gp_fn(real, D, accelerator.scaler) * cfg.gp_lambda
                D_loss = adv_loss + gp_loss

            accelerator.backward(D_loss)
            optim_D.step()

            with accelerator.autocast():
                # D forward
                fake_prob = D(fake_aug)

                # loss
                G_loss = adv_fn.g_loss(fake_prob)

            accelerator.backward(G_loss)
            optim_G.step()
            if cfg.ema: update_ema(G, G_ema, cfg.ema_decay, True)

            if cfg.running > 0 and status.batches_done % cfg.running == 0:
                save_image(fake, os.path.join(cfg.checkpoint_folder, 'running.jpg'),
                    normalize=True, value_range=(-1, 1))

            if status.batches_done % cfg.save == 0:
                with torch.no_grad():
                    kbatches = status.batches_done / 1000
                    if cfg.ema:
                        images = G_ema(const_z)
                        state_dict = G_ema.state_dict()
                    else:
                        G.eval()
                        images = G(images)
                        state_dict = G.state_dict()
                        G.train()
                    save_image(images,
                        os.path.join(cfg.checkpoint_folder, 'images', f'{kbatches:.2f}.jpg'),
                        nrow=int(cfg.num_test**0.5), normalize=True, value_range=(-1, 1))
                    torch.save(state_dict,
                        os.path.join(cfg.checkpoint_folder, 'models', f'model_{kbatches}.pth'))

            accelerator.update()
            status.update(G=G_loss.item(), D=D_loss.item())
            if cfg.update_augment:
                cfg.update_augment(augment, step_info(real_prob, fake_prob))

            if status.is_end():
                break

    status.plot(os.path.join(cfg.checkpoint_folder, f'loss.png'))
