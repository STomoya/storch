
# Advanced topics

- [Gradient Accumulation](#gradient-accumulation)
- [Distributed training](#distributed-training)

## Gradient Accumulation

- Use an instance of `storch.torchops.optimizer_step_with_gradient_accumulation` instead of `storch.torchops.optimizer_step`

```diff
- from storch.torchops import optimizer_step
+ from storch.torchops import optimizer_step_with_gradient_accumulation

...

+ optimizer_step = optimizer_step_with_gradient_accumulation(
+     cfg.grad_accum_steps, len(dataset)
+ )

...
```

## Distributed training

- Use `stroch.distributed.DistributedHelper`.

- Classes that save the model to a binary need an instance of `DistributedHelper` to be passed in on construction to work properly in distributed setting.

    - `storch.metrics.BestModelKeeper`
    - `storch.checkpoint.Checkpoint`

- Use `DistributedHelper.gather` or `DistributedHelper.reduce` function for communicating between processes. Currently it does not support `scatter` (v0.4.0).

- No need to use the `torch.cuda.amp.autocast` context manager for AMP.

- The `DataLoader` should be constructed via `DistributedHelper.prepare_dataset` to automatically setup a `DistrubutedSampler` sampler unless this is done by the user.

- The optimizers must be constructed after wrapping the model via `DistributedHelper.prepare_module`

- When data parallel mode is `FullyShardedDataParallel`

    - The checkpoint type is always `FULL_STATE_DICT`. `offload_to_cpu` option in `DistributedHelper.prepare_for_checkpoint` will configure both, `offload_to_cpu` and `rank0_only` option.

    - Passing the `no_sync` context manager to `optimizer_step` is optional (but preferred).

#### Best practices

- Lauch via `torchrun`

    - Using `DistributedHelper`, training codes launched via `torchrun` can also work on single process, single GPU environment without any code changes.

From [the basics](./Basics.md#keeping-the-best-model):

```diff
  import torch
  from storch.torchops import optimizer_step_with_gradient_accumulation
  from storch.checkpoint import Checkpoint
- from storch.status import Status
+ from storch.status import Status, ThinStatus
  from storch.metrics import BestModelKeeper
+ from storch.distributed import DistributedHelper

+ def main():
+     disthelper = DistributedHelper()

      cfg = get_config(...)

-     device = torch.device(cfg.device)
+     device = disthelper.device

-     dataset = get_dataset(cfg)
+     dataset = get_dataset(cfg, no_loader=True)
+     dataset = disthelper.prepare_dataset(dataset, **cfg.dataloader)

      model = get_model(cfg)
+     model = disthelper.prepare_module(model, cfg.dp_mode, cfg.amp_enabled)

      optimizer = get_optimizer(model.parameters(), cfg)
      criterion = get_criterion(cfg)

-     gradscaler = GradScaler() if cfg.amp_enabled else None
+     if cfg.amp_enabled:
+         gradscaler = ShardedGradScaler() if cfg.dp_mode == 'fsdp' else GradScaler()
+     else: gradscaler = None

      train_iters = cfg.epochs * len(dataset)
+     StatusCls = Status if dishelper.is_primary() else ThinStatus
-     status = Status(train_iters, cfg.log_file)
+     status = StatusCls(train_iters, cfg.log_file)
      status.log_stuff(cfg, model, optimizer, dataset)

+     model_ckpt, optim_ckpt = disthelper.prepare_for_checkpoint(optimizer, cfg.offload_to_cpu)

      best_model = BestModelKeeper(
-         'best-train-loss', 'min', model, folder=cfg.checkpoint_folder
+         'best-train-loss', 'min', model_ckpt, folder=cfg.checkpoint_folder, disthelper=disthelper
      )

-     checkpoint = Checkpoint(cfg.checkpoint_folder, cfg.keep_last)
+     checkpoint = Checkpoint(cfg.checkpoint_folder, cfg.keep_last, disthelper=disthelper)
      checkpoint.register(
-         model=model, optimizer=optimizer,
+         model=model_ckpt, optimizer=optim_ckpt,
          status=status, best_model=best_model)
      if cfg.amp_enabled:
          checkpoint.register(scaler=gradscaler)
      checkpoint.load_latest()

      optimizer_step = optimizer_step_with_gradient_accumulation(
-         cfg.grad_accum_steps, len(dataset)
+         cfg.grad_accum_steps, len(dataset), model.no_sync if cfg.dp_mode == 'fsdp' else None
      )

      while not status.is_end():
          for inputs, labels in dataset:
-             with autocast(enabled=cfg.amp_enabled):
              outputs = model(inputs)
              loss = criterion(outputs, labels)

              optimizer_step(
                  loss, optimizer, gradscaler, module=model,
                  zero_grad=True, set_to_none=True,
                  clip_grad_norm=cfg.clip_grad_norm, max_norm=cfg.max_norm,
                  update_scaler=True
              )

              batch_acc = accuracy(outputs, labels)
+             loss = disthelper.reduce_mean(loss)
+             batch_acc = disthelper.reduce_mean(batch_acc)
              status.update(**{
                  'Loss/CE': loss, 'Metrics/batch_accuarcy': batch_acc
              })

          epoch_loss_mean = ...
          best_model.update(epoch_loss_mean, step=status.batches_done)
          checkpoint.save()

+ if __name__=='__main__':
+     main()
```
