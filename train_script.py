"""

`PyTorch <https://pytorch.org/>`_ implementation of a training script

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

"""
import re
from typing import Optional, Union, List, Literal, Callable, Dict, Iterator
import os
import math
from tqdm import tqdm
from decimal import Decimal

import torch
from torch.utils.data import DataLoader
import torchvision
import torch.backends.cudnn as cudnn
from torch.nn import Parameter

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from accelerate import Accelerator, DistributedDataParallelKwargs

from torchmetrics import MetricCollection
import wandb

import utils
from gaussian_ot import GaussianOT

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)
torch.autograd.set_detect_anomaly(True)

def train_model(
        model: torch.nn.Module,
        dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
        metric: MetricCollection,
        monitor: str,
        mode: Literal['max', 'min'] = 'max',
        early_stopping_patience: Optional[int] = None,
        out_dir: Optional[str] = None,
        loggers: Optional[List[str]] = "tensorboard",  # also supported: "wandb"
        denormalize_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        project_name: Optional[str] = "dummy",
        options: Optional[List[str]] = None,
        max_epoch: int = 10,
        max_step: int = 100000,
        train_ckpt_path: Optional[Union[str, Literal['best', 'last']]] = 'best',
        lr: float = 2e-4,
        batch_size: int = 16,
        log_interval: int = 1000,
        val_every: float = 1.,  # num_epoch / num_validation (can be < 1)
        debug: bool = False,
        train_params: Optional[Iterator[Parameter]] = None,
        **dataloader_kwargs
):
    # ******************************************************************************************************************
    # Logging Initialization
    # ******************************************************************************************************************
    if monitor not in metric.values():
        raise ValueError(f"monitored metric `{monitor}` expected to be found in the metric collection `{metric.keys()}`.")

    if out_dir is None:
        out_dir = 'logs'
        if options is not None:
            out_dir += '/' + '-'.join(options) + '-train'
        else:
            out_dir += '/train'

    BEST_CKPT = f'{out_dir}/best-ckpt'
    LAST_CKPT = f'{out_dir}/last-ckpt'

    config = {
        'max_epoch': max_epoch,
        'max_step': max_step,
        'train_ckpt_path': train_ckpt_path,
        'lr': lr,
        'batch_size': batch_size,
        'log_interval': log_interval,
        'val_every': val_every,
    }
    # if hasattr(model, 'config'):
    #     config['model'] = model.config

    options = [] if options is None else options

    # We use hugging face's `accelerator` library to handle ddp and device placement
    ddp = Accelerator(
        log_with=loggers,
        logging_dir=out_dir,
        # kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )

    if debug:
        ddp.print("RUNNING IN DEBUG MODE (1 training and 1 validation batch)")
        log_interval = 10
        max_epoch = 1
        max_step = 20
        options.append("debug")
        project_name += '-DEBUG'

    if loggers is not None:
        logger_init = {"wandb": {"tags": options, 'job_type': "train"},}
        ddp.init_trackers(project_name, config, init_kwargs=logger_init)

    if train_ckpt_path == 'best':
        train_ckpt_path = os.path.realpath(BEST_CKPT) if os.path.islink(BEST_CKPT) else None
    elif train_ckpt_path == 'last':
        train_ckpt_path = os.path.realpath(LAST_CKPT) if os.path.islink(LAST_CKPT) else None

    # ******************************************************************************************************************
    # Data
    # ******************************************************************************************************************
    train = dataset["train"]
    val_key = "val" if "val" in dataset else "validation"
    if val_key in dataset and "test" in dataset:
        val = dataset[val_key]
        test = dataset["test"]
    else:
        missing = "test" if "test" in dataset else val_key
        present = "test" if "test" not in dataset else val_key
        ddp.print(f"""
        Warning: `{missing}` partition is missing, saving `{present}` partition
        for test and 10% of `train` partiton for validation
        """)
        train, val = utils.random_split(train, split=(0.9, 0.1))
        test = dataset["test" if "test" in dataset else val_key]

    train = DataLoader(train, batch_size, True, **dataloader_kwargs)
    val = DataLoader(val, 4 * batch_size, False, **dataloader_kwargs)
    test = DataLoader(test, 4 * batch_size, False, **dataloader_kwargs)
    train, val, test = tuple(map(ddp.prepare_data_loader, (train, val, test)))

    # ******************************************************************************************************************
    # Model
    # ******************************************************************************************************************
    epoch_without_improvement = 0
    current_best_ckpt_metric = None
    global_step = 0

    if train_params is None: train_params = model.parameters()
    optimizer = torch.optim.Adam(train_params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch * len(train), 1e-6)
    if train_ckpt_path is not None:
        ddp.load_state(train_ckpt_path)
        numbers = re.findall(r"[-+]?(?:\d*\.*\d+)", train_ckpt_path)
        if len(numbers) == 2:
            global_step = int(numbers[0][1:])
            current_best_ckpt_metric = float(numbers[1][1:])
            ddp.print(f"continuing training from step {global_step} with metric={current_best_ckpt_metric}")

    model, optimizer, sched, metric = ddp.prepare(model, optimizer, sched, metric)
    # model, optimizer, metric = ddp.prepare(model, optimizer, metric)

    # ddp.register_for_checkpointing(sched)
    # ddp.register_for_checkpointing(metric)

    if val_every < 1:
        max_epoch = int(max_epoch / val_every)
        max_step = min(max_step, int(len(train) * max_epoch * val_every))
    else:
        max_step = min(max_step, len(train) * max_epoch)

    for partial_epoch in range(max_epoch):
        epoch = int(partial_epoch * val_every) if val_every < 1 else partial_epoch
        model.train()
        # **************************************************************************************************************
        # Training
        # **************************************************************************************************************
        for inputs in (pbar := tqdm(train, desc=f"epoch {epoch}", leave=False, disable=not ddp.is_local_main_process)):
            optimizer.zero_grad()
            outputs = model(**inputs)
            ddp.backward(outputs.loss)
            ddp.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
            log_dict = {
                key: val.detach().item()
                for key, val in outputs.items()
                if torch.is_tensor(val) and torch.numel(val) == 1
            }

            # **********************************************************************************************************
            # Logging
            # **********************************************************************************************************
            pbar.set_postfix(log_dict)
            ddp.log(log_dict, step=global_step)

            if global_step % log_interval == 0:
                if 'pixel_values' in inputs:
                    img_shape = inputs['pixel_values'].shape
                    columns = {
                        key: val
                        for key, val in {**inputs, **outputs}.items()
                        if torch.is_tensor(val) and img_shape == val.shape
                    }
                    collage = torch.cat(tuple(columns.values()), dim=-1).detach().cpu()[:8]
                    if denormalize_func is not None:
                        collage = denormalize_func(collage)
                    collage = torchvision.utils.make_grid(collage.clip(0., 1.), 1, 0)
                    caption = ' | '.join(columns.keys())
                    if "wandb" in loggers:
                        ddp.log({caption: wandb.Image(collage, caption=caption)}, step=global_step)
                    elif "tensorboard" in loggers:
                        if ddp.is_local_main_process:
                            ddp.get_tracker("tensorboard").add_image(caption, collage, global_step=global_step)

            global_step += 1
            if debug or global_step > max_step or (val_every < 1 and global_step % int(val_every * len(train)) == 0):
                break

        # **************************************************************************************************************
        # Validation
        # **************************************************************************************************************
        ddp.wait_for_everyone()
        model.eval()
        metric.reset()
        with torch.no_grad():
            for inputs in (pbar := tqdm(val, desc=f"validation", leave=False, disable=not ddp.is_local_main_process)):
                metric.update(**model(**inputs))
                if debug: break

            results = metric.compute()
            ddp.log({**results, 'epoch': epoch}, step=global_step)
            ddp.print(", ".join(f'{key.split("/")[-1]}: {val.item():.2f}' for key, val in results.items()))

        # **************************************************************************************************************
        # Checkpoint Saving
        # **************************************************************************************************************
        ddp.wait_for_everyone()
        res_str = '-'.join([f'{metric_name.split("/")[-1]}-{result.item():.2f}' for metric_name, result in results.items()])
        save_path = f'checkpoints/step-{str(global_step).zfill(int(math.log10(max_step)) + 1)}-{res_str}'
        ddp.save_state(f'{out_dir}/{save_path}')
        ddp.unwrap_model(model).save_pretrained(
            save_directory=f'{out_dir}/{save_path}/pretrained',
            is_main_process=ddp.is_local_main_process,
            save_function=ddp.save
        )

        if ddp.is_local_main_process:
            if os.path.islink(LAST_CKPT): os.remove(LAST_CKPT)
            os.symlink(save_path, LAST_CKPT)
            new_best = current_best_ckpt_metric is None or\
                       (results[monitor] > current_best_ckpt_metric and mode == 'max') or\
                       (results[monitor] < current_best_ckpt_metric and mode == 'min')
            if new_best:
                current_best_ckpt_metric = new_best
                if os.path.islink(BEST_CKPT): os.remove(BEST_CKPT)
                os.symlink(save_path, BEST_CKPT)
                epoch_without_improvement = 0
            else:
                epoch_without_improvement += 1

        if early_stopping_patience is not None and epoch_without_improvement * val_every > early_stopping_patience:
            ddp.print(f"""
            Validation metric {monitor} did not improve for the last {epoch_without_improvement * val_every} epochs.
            Early stopping patience ({early_stopping_patience}) exceeded. Terminating.
            """)
            break
        if global_step > max_step:
            break

    # ******************************************************************************************************************
    # Teardown
    # ******************************************************************************************************************
    ddp.print(results)
    ddp.print("Beginning teardown")
    ddp.end_training()
    ddp.clear()


class TransportOutput(utils.BaseOutput):
    pg_star: float
    w2: torch.FloatTensor
    transport: GaussianOT
    performance: Dict[str, torch.FloatTensor]
    samples: torch.FloatTensor


def transport(
        ddp: Accelerator,
        mmse: torch.nn.Module,
        gaussian_ots: Optional[Union[torch.nn.Sequential, GaussianOT]],
        dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
        metrics: MetricCollection,
        batch_size: int = 100,
        n_plot_samples: int = 4,
        desc: str = "degraded -> clean",
        pg_star: float = 0.,
        denormalize_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **dataloader_kwargs
):
    ddp.clear()
    train, val = dataset["train"], dataset["validation"]
    _, train = utils.random_split(dataset["train"], split=(0.9, 0.1))
    # _, val = utils.random_split(dataset["validation"], split=(49/50, 1/50))
    train = DataLoader(train, batch_size, True, **dataloader_kwargs)
    val = DataLoader(val, batch_size, True, **dataloader_kwargs)
    train, val = ddp.prepare(train, val)
    metrics = metrics.to(ddp.device)
    mmse = mmse.to(ddp.device)
    metrics.reset()

    if gaussian_ots is not None:
        gaussian_ots = gaussian_ots.to(ddp.device)
        for i in range(len(gaussian_ots)):
            if gaussian_ots[i]._computed is None:  # noqa
                for inputs in tqdm(train, desc=desc, leave=False, disable=not ddp.is_local_main_process):
                    # unpaired restoration: half the batch goes for source estimation, the other half for target
                    gaussian_ots[i].update(
                        source=gaussian_ots[:i](mmse(inputs["degraded"][:batch_size // 2])),
                        target=inputs["pixel_values"][batch_size // 2:]
                    )
            w2 = gaussian_ots[i].compute()  # might use cached computation
            ddp.print(f"{desc} transport cost stage {i}: {w2.item():.2f}")
            gaussian_ots[i].pg_star = pg_star

    for inputs in tqdm(val, desc=f"{desc} performance", leave=False, disable=not ddp.is_local_main_process):
        preds = mmse(inputs["degraded"])
        if gaussian_ots is not None:
            preds = gaussian_ots(preds)
        metrics.update(preds=preds, target=inputs["pixel_values"])

    perf = metrics.compute()
    ddp.print(f"{desc} performance: {', '.join([f'{metric}: {val.item():.2f}' for metric, val in perf.items()])}")

    transported_samples = preds[:n_plot_samples]
    clean_samples = inputs["pixel_values"][:n_plot_samples]
    if denormalize_func is not None:
        transported_samples, clean_samples = map(denormalize_func, (transported_samples, clean_samples))

    out_degraded = TransportOutput(w2=w2, transport=gaussian_ots, performance=perf, samples=transported_samples, pg_star=pg_star)
    out_clean = TransportOutput(w2=torch.zeros_like(w2), performance={"PSNR": float('inf'), "FID": 0.}, samples=clean_samples)
    return out_degraded, out_clean
