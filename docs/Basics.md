# Basics

## Original code sample

- Classification
- Optionally use AMP
- Optionally use `clip_grad_norm_`

```diff
  import torch

  cfg = get_config(...)

  device = torch.device(cfg.device)
  dataset = get_dataset(cfg)
  model = get_model(cfg)
  optimizer = get_optimizer(model.parameters(), cfg)
  criterion = get_criterion(cfg)

  gradscaler = GradScaler() if cfg.amp_enabled else None

  for epoch in range(cfg.epochs):
      for inputs, labels in dataset:
          with autocast(enabled=cfg.amp_enabled):
              outputs = model(inputs)
              loss = criterion(outputs, labels)

          optimizer.zero_grad(set_to_none=True)

          if cfg.amp_enabled:
              scaler.scaler(loss).backward()
              scaler.unscale_(optimizer)
              if cfg.clip_grad_norm:
                  torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
              scaler.step(optimizer)
          else:
              loss.backward()
              if cfg.clip_grad_norm:
                  torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
              optimizer.step()
```

## Optimization step

- Use `storch.torchops.optimizer_step`.

```diff
  import torch
+ from storch.torchops import optimizer_step

  cfg = get_config(...)

  device = torch.device(cfg.device)
  dataset = get_dataset(cfg)
  model = get_model(cfg)
  optimizer = get_optimizer(model.parameters(), cfg)
  criterion = get_criterion(cfg)

  gradscaler = GradScaler() if cfg.amp_enabled else None

  for epoch in range(cfg.epochs):
      for inputs, labels in dataset:
          with autocast(enabled=cfg.amp_enabled):
              outputs = model(inputs)
              loss = criterion(outputs, labels)

-        optimizer.zero_grad(set_to_none=True)
-        if cfg.amp_enabled:
-            scaler.scaler(loss).backward()
-            scaler.unscale_(optimizer)
-            if cfg.clip_grad_norm:
-                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
-            scaler.step(optimizer)
-        else:
-            loss.backward()
-            if cfg.clip_grad_norm:
-                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
-            optimizer.step()
+        optimizer_step(
+            loss, optimizer, scaler, module=model,
+            zero_grad=True, set_to_none=True,
+            clip_grad_norm=cfg.clip_grad_norm, max_norm=cfg.max_norm,
+            update_scaler=True
+        )
```

## Checkpointing

- Use `storch.checkpoint.Checkpoint`.

```diff
  import torch
  from storch.torchops import optimizer_step
+ from storch.checkpoint import Checkpoint

  cfg = get_config(...)

  device = torch.device(cfg.device)
  dataset = get_dataset(cfg)
  model = get_model(cfg)
  optimizer = get_optimizer(model.parameters(), cfg)
  criterion = get_criterion(cfg)

  gradscaler = GradScaler() if cfg.amp_enabled else None

+ checkpoint = Checkpoint(cfg.checkpoint_folder, cfg.keep_last)
+ checkpoint.register(model=model, optimizer=optimizer)
+ if cfg.amp_enabled:
+     checkpoint.register(scaler=gradscaler)
+ checkpoint.load_latest()

  for epoch in range(cfg.epochs):
      for inputs, labels in dataset:
          with autocast(enabled=cfg.amp_enabled):
              outputs = model(inputs)
              loss = criterion(outputs, labels)

          optimizer_step(
              loss, optimizer, gradscaler, module=model,
              zero_grad=True, set_to_none=True,
              clip_grad_norm=cfg.clip_grad_norm, max_norm=cfg.max_norm,
              update_scaler=True
          )
+     checkpoint.save()
```

## Logging

- Use `storch.status.Status`.

```diff
  import torch
  from storch.torchops import optimizer_step
  from storch.checkpoint import Checkpoint
+ from storch.status import Status

  cfg = get_config(...)

  device = torch.device(cfg.device)
  dataset = get_dataset(cfg)
  model = get_model(cfg)
  optimizer = get_optimizer(model.parameters(), cfg)
  criterion = get_criterion(cfg)

  gradscaler = GradScaler() if cfg.amp_enabled else None

+ train_iters = cfg.epochs * len(dataset)
+ status = Status(train_iters, cfg.log_file)
+ status.log_stuff(cfg, model, optimizer, dataset)

  checkpoint = Checkpoint(cfg.checkpoint_folder, cfg.keep_last)
- checkpoint.register(model=model, optimizer=optimizer)
+ checkpoint.register(model=model, optimizer=optimizer, status=status)
  if cfg.amp_enabled:
      checkpoint.register(scaler=gradscaler)
  checkpoint.load_latest()

- for epoch in range(cfg.epochs):
+ while not status.is_end():
      for inputs, labels in dataset:
          with autocast(enabled=cfg.amp_enabled):
              outputs = model(inputs)
              loss = criterion(outputs, labels)

          optimizer_step(
              loss, optimizer, gradscaler, module=model,
              zero_grad=True, set_to_none=True,
              clip_grad_norm=cfg.clip_grad_norm, max_norm=cfg.max_norm,
              update_scaler=True
          )

+         status.update()

      checkpoint.save()
```

To log loss, etc, pass in a dictionary with `**`.

```diff
  import torch
  from storch.torchops import optimizer_step
  from storch.checkpoint import Checkpoint
  from storch.status import Status

  cfg = get_config(...)

  device = torch.device(cfg.device)
  dataset = get_dataset(cfg)
  model = get_model(cfg)
  optimizer = get_optimizer(model.parameters(), cfg)
  criterion = get_criterion(cfg)

  gradscaler = GradScaler() if cfg.amp_enabled else None

  train_iters = cfg.epochs * len(dataset)
  status = Status(train_iters, cfg.log_file)
  status.log_stuff(cfg, model, optimizer, dataset)

  checkpoint = Checkpoint(cfg.checkpoint_folder, cfg.keep_last)
  checkpoint.register(model=model, optimizer=optimizer, status=status)
  if cfg.amp_enabled:
      checkpoint.register(scaler=gradscaler)
  checkpoint.load_latest()

  while not status.is_end():
      for inputs, labels in dataset:
          with autocast(enabled=cfg.amp_enabled):
              outputs = model(inputs)
              loss = criterion(outputs, labels)

          optimizer_step(
              loss, optimizer, gradscaler, module=model,
              zero_grad=True, set_to_none=True,
              clip_grad_norm=cfg.clip_grad_norm, max_norm=cfg.max_norm,
              update_scaler=True
          )

-         status.update()
+         batch_acc = accuracy(outputs, labels)
+         status.update(**{
+             'Loss/CE': loss, 'Metrics/batch_accuarcy': batch_acc
+         })

      checkpoint.save()
```

## Keeping the best model.

- Use `storch.metrics.BestModelKeeper`.

```diff
  import torch
  from storch.torchops import optimizer_step
  from storch.checkpoint import Checkpoint
  from storch.status import Status
+ from storch.metrics import BestModelKeeper

  cfg = get_config(...)

  device = torch.device(cfg.device)
  dataset = get_dataset(cfg)
  model = get_model(cfg)
  optimizer = get_optimizer(model.parameters(), cfg)
  criterion = get_criterion(cfg)

  gradscaler = GradScaler() if cfg.amp_enabled else None

  train_iters = cfg.epochs * len(dataset)
  status = Status(train_iters, cfg.log_file)
  status.log_stuff(cfg, model, optimizer, dataset)

+ best_model = BestModelKeeper(
+     'best-train-loss', 'min', model, folder=cfg.checkpoint_folder
+ )

  checkpoint = Checkpoint(cfg.checkpoint_folder, cfg.keep_last)
- checkpoint.register(model=model, optimizer=optimizer, status=status)
+ checkpoint.register(model=model, optimizer=optimizer,
+     status=status, best_model=best_model)
  if cfg.amp_enabled:
      checkpoint.register(scaler=gradscaler)
  checkpoint.load_latest()

  while not status.is_end():
      for inputs, labels in dataset:
          with autocast(enabled=cfg.amp_enabled):
              outputs = model(inputs)
              loss = criterion(outputs, labels)

          optimizer_step(
              loss, optimizer, gradscaler, module=model,
              zero_grad=True, set_to_none=True,
              clip_grad_norm=cfg.clip_grad_norm, max_norm=cfg.max_norm,
              update_scaler=True
          )

          batch_acc = accuracy(outputs, labels)
          status.update(**{
              'Loss/CE': loss, 'Metrics/batch_accuarcy': batch_acc
          })

+     epoch_loss_mean = ...
+     best_model.update(epoch_loss_mean, step=status.batches_done)
      checkpoint.save()
```

## Advanced topics

- See the [Advanced topics](./Advanced.md) page.
