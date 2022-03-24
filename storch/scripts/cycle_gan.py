
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
import os
from typing import Callable

import torch
import toorch.optim as optim

from storch.dataset.dataset import DatasetBase
from storch.status import Status
from storch.accelerate import MiniAccelerator
from storch.imageops import save_images
import storch.loss as loss

def get_loss_fns(adv_type):
    if adv_type == 'sn':
        adv_fn = loss.NonSaturatingLoss()
    if adv_type == 'gan':
        adv_fn = loss.GANLoss()
    if adv_type == 'ls':
        adv_fn = loss.LSGANLoss()
    if adv_type == 'hinge':
        adv_fn = loss.HingeLoss()
    return adv_fn

@dataclass
class Config:
    GA: torch.nn.Module
    GB: torch.nn.Module
    DA: torch.nn.Module
    DB: torch.nn.Module
    dataset: DatasetBase
    input_size: tuple
    iterations: int=200000
    adv_type:str='ls'
    g_lr: float=0.0002
    d_lr: float=0.0002
    betas: tuple[float]=(0.5, 0.999)
    cycle_lambda: float=10.
    identity_lambda: float=0.
    vgg_lambda: float=0.
    augment: Callable=None
    update_augment: Callable=None
    amp: bool=True
    num_test: int=16
    save: int=1000
    checkpoint_folder: str='checkpoint'
    log_file: str='log.log'
    log_interval: int=10
    running: int=-1

@dataclass
class step_info:
    real_a_prob: torch.Tensor
    real_b_prob: torch.Tensor
    fake_a_prob: torch.Tensor
    fake_b_prob: torch.Tensor

def train_fn(cfg: Config):

    status = Status(cfg.iterations, False, os.path.join(cfg.checkpoint_folder, cfg.log_file),
        cfg.log_interval, __name__)
    status.log_training(cfg)

    optim_G = optim.Adam(chain(cfg.GA.parameters(), cfg.GB.parameters()), lr=cfg.g_lr, betas=cfg.betas)
    optim_D = optim.Adam(chain(cfg.DA.parameters(), cfg.DB.parameters()), lr=cfg.d_lr, betas=cfg.betas)

    accelerator = MiniAccelerator(cfg.amp)
    GA, GB, DA, DB, dataset, optim_G, optim_D, augment = accelerator.prepare(
        cfg.GA, cfg.GB, cfg.DA, cfg.DB, cfg.dataset, optim_D, optim_G, cfg.augment)

    adv_fn = get_loss_fns(cfg.adv_type)
    l1_fn  = torch.nn.L1Loss()

    while not status.is_end():
        for A, B in dataset:
            optim_D.zero_grad()
            optim_G.zero_grad()

            with accelerator.autocast():
                # G forward
                AB = GB(A)
                BA = GA(B)
                ABA = GA(AB)
                BAB = GB(BA)
                AA = GA(A)
                BB = GB(B)

                # D forward
                real_a_prob = DA(A)
                real_b_prob = DB(B)
                fake_a_prob = DA(BA.detach())
                fake_b_prob = DB(AB.detach())

                # loss
                adv_a_loss = adv_fn.d_loss(real_a_prob, fake_a_prob)
                adv_b_loss = adv_fn.d_loss(real_b_prob, fake_b_prob)
                D_loss = adv_a_loss + adv_b_loss

            accelerator.backward(D_loss)
            optim_D.step()

            with accelerator.autocast():
                # D forward
                fake_a_prob = DA(BA)
                fake_b_prob = DB(AB)

                # loss
                adv_a_loss = adv_fn.g_loss(fake_a_prob)
                adv_b_loss = adv_fn.g_loss(fake_b_prob)
                adv_loss = adv_a_loss + adv_b_loss
                cycle_loss, identity_loss = 0, 0
                if cfg.cycle_lambda > 0:
                    cycle_a_loss = l1_fn(ABA, A)
                    cycle_b_loss = l1_fn(BAB, B)
                    cycle_loss = cycle_a_loss + cycle_b_loss
                if cfg.identity_lambda > 0:
                    identity_a_loss = l1_fn(AA, A)
                    identity_b_loss = l1_fn(BB, A)
                    identity_loss = identity_a_loss + identity_b_loss
                G_loss = adv_loss + cycle_loss + identity_loss

            accelerator.backward(G_loss)
            optim_G.step()

            if cfg.running > 0 and status.batches_done % cfg.running == 0:
                save_images(A, AB, B, BA,
                    filename=os.path.join(cfg.checkpoint_folder, 'running.jpg'), nrow=4)

            if status.batches_done % cfg.save == 0:
                kbatches = status.batches_done / 1000
                save_images(A, AB, B, BA,
                    filename=os.path.join(cfg.checkpoint_folder, f'{kbatches:.2f}.jpg'), nrow=4)
                torch.save(dict(GA=GA.state_dict(), GB=GB.state_dict()),
                    os.path.join(cfg.checkpoint_folder, f'model-{kbatches:.2f}.pth'))

            accelerator.update()
            status.update({
                'Loss/adv/G': G_loss.item(), 'Loss/adv/D': D_loss.item(),
                'Loss/cycle': cycle_loss.item(), 'Loss/identity': identity_loss.item()})
            if cfg.update_augment:
                cfg.update_augment(augment,
                    step_info(real_a_prob, real_b_prob, fake_a_prob, fake_b_prob))

            if status.is_end():
                break

    status.plot(os.path.join(cfg.checkpoint_folder, f'loss.png'))
