"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a training script

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Sequence, Union, List
import os
import datetime
import math
from tqdm import tqdm
from functools import partial, partialmethod
from itertools import accumulate

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision.datasets import ImageNet
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import Compose
import torch.backends.cudnn as cudnn

from torchsr.models import edsr
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, FrechetInceptionDistance
from accelerate import Accelerator
from vae import VAE
from ot import get_statistics, get_transport
import wandb

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


# **********************************************************************************************************************
# Constants
# **********************************************************************************************************************

IMG_SIZE = 256
N_EPOCH = 10
N_ANNEALING = 4
CKPT_PATH = None
BETA = 0.
DEBUG = False
SF = 4
DATA_PATH = os.path.expanduser("~/data/ImageNet")


def cosine_annealing(step, n_step, max_val):
    return max_val * (0.5 * math.cos(math.pi * (step / n_step + 1)) + 0.5)

def random_split(dataset: Union[Dataset, Sequence[Dataset]], split: Sequence[Union[int, float]], seed=0):
    is_data_list = not isinstance(dataset, Dataset)
    if is_data_list:
        length = len(dataset[0])  # noqa
        assert all(len(d) == length for d in dataset)
    else:
        length = len(dataset)

    if not all(isinstance(s, int) for s in split):
        assert all(0 < s < 1 for s in split) and sum(split) == 1
        split = list(map(lambda frac: int(frac * length), split))
        split[-1] = length - sum(split[:-1])

    if is_data_list: assert len(split) == len(dataset)
    assert sum(split) == length

    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    if is_data_list:
        return [Subset(d, indices[offset - length: offset]) for d, offset, length in zip(dataset, accumulate(split), split)]
    else:
        return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(accumulate(split), split)]


if __name__ == "__main__":
    # We use hugging face's `accelerator` library to handle ddp and device placement
    ddp = Accelerator()

    # ******************************************************************************************************************
    # Data
    # ******************************************************************************************************************
    out_dir = f'logs/{datetime.datetime.now().strftime("%d-%m-%Y-%H:%M")}'
    os.makedirs(out_dir, exist_ok=True)

    # Load the ImageNet dataset
    resize = [T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()]
    degraded = [T.Resize((IMG_SIZE // SF, IMG_SIZE // SF)), T.ToTensor()]
    normalize, denormalize = T.Normalize(0.5, 0.5), T.Compose([T.Normalize(0., 2.), T.Normalize(-0.5, 1.)])
    train = ImageNet(DATA_PATH, transform=Compose([*resize, normalize]))
    val = ImageNet(DATA_PATH, split="val", transform=Compose([*resize, normalize]))
    val_source = ImageNet(DATA_PATH, transform=Compose(degraded))
    train, val_source, val_target = random_split([train, val_source, train], (0.5, 0.25, 0.25))  # non-overlapping

    # Create data-loaders for the dataset partitions
    train = DataLoader(train, batch_size=10, shuffle=True, num_workers=10)
    val_source = DataLoader(val_source, batch_size=20, shuffle=True, num_workers=10)
    val_target = DataLoader(val_target, batch_size=20, shuffle=True, num_workers=10)
    val = DataLoader(val, batch_size=20, shuffle=True, num_workers=10)
    # Split evenly the data across devices
    train, val_source, val_target, val = tuple(map(ddp.prepare_data_loader, (train, val_source, val_target, val)))

    # ******************************************************************************************************************
    # Model & Optimizer
    # ******************************************************************************************************************
    model = VAE(
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=(64, 128, 256),  # noqa
        layers_per_block=2,
        latent_channels=256,
        sample_posterior=BETA > 0
    )
    latent_channels = 256
    mmse = edsr(scale=SF, pretrained=True).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCH * len(train), 1e-6)
    metric = MetricCollection({
        "psnr-y": PeakSignalNoiseRatio(), "fid-y": FrechetInceptionDistance(normalize=True),
        "psnr-x": PeakSignalNoiseRatio(), "fid-x": FrechetInceptionDistance(normalize=True),
        "psnr-x*": PeakSignalNoiseRatio(), "fid-x*": FrechetInceptionDistance(normalize=True),
        **{f"psnr-x0{i}": PeakSignalNoiseRatio() for i in range(10)},
        **{f"fid-x0{i}": FrechetInceptionDistance(normalize=True) for i in range(10)}
    })

    def update_metric(key, samples, recons):
        recons, samples = denormalize(recons), denormalize(samples)
        metric[f"psnr-{key}"].update(recons, samples)
        metric[f"fid-{key}"].update(samples, real=True)
        metric[f"fid-{key}"].update(recons, real=False)

    if CKPT_PATH is not None: ddp.load_state(CKPT_PATH)  # noqa
    model, mmse, optimizer, sched, metric = ddp.prepare(model, mmse, optimizer, sched, metric)

    if ddp.is_local_main_process and not DEBUG:
        run = wandb.init(project="icci", job_type="train")

    # ******************************************************************************************************************
    # Training
    # ******************************************************************************************************************
    global_step = 0
    annealed_beta = partial(cosine_annealing, n_step=N_ANNEALING * len(train), max_val=BETA)

    for epoch in range(N_EPOCH):
        model.train()
        for x, _ in (pbar := tqdm(train, desc=f"epoch {epoch}", leave=False, disable=not ddp.is_local_main_process)):
            optimizer.zero_grad()
            outputs = model(x, recon_loss="mse_loss", beta=annealed_beta(global_step))
            loss = outputs.total_loss
            if ~loss.isfinite().all(): loss = torch.zeros_like(loss)
            ddp.backward(loss)
            ddp.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
            global_step += 1
            if DEBUG: break
            # **********************************************************************************************************
            # Logging & Checkpoint Saving
            # **********************************************************************************************************
            if ddp.is_local_main_process:
                log_dict = {
                    "loss": outputs.total_loss.detach().item(),
                    "kl_unscaled": outputs.kl_unscaled.detach().item(),
                    "kl": outputs.kl.detach().item(),
                    "recon_loss": outputs.recon_loss.detach().item(),
                }
                pbar.set_postfix(log_dict)
                wandb.log({**log_dict, 'step': global_step})

            if global_step % 1000 == 0:
                with torch.no_grad():
                    ddp.wait_for_everyone()
                    save_path = f'{out_dir}/checkpoints/step-{str(global_step).zfill(5)}'
                    ddp.save_state(f'{save_path}/accelerator_full_state')
                    ddp.unwrap_model(model).save_pretrained(
                        save_directory=save_path,
                        save_function=ddp.save,
                        is_main_process=ddp.is_main_process
                    )
                    if ddp.is_local_main_process:
                        num_samples, num_decodings = 8, 4 if BETA > 0 else 1
                        images = x[:num_samples]
                        images = torch.cat([images, ] * num_decodings, dim=0)
                        outputs = model(images)
                        recons = torch.chunk(outputs.recon, num_decodings, 0)
                        samples = torch.chunk(outputs.sample, num_decodings, 0)

                        collage = torch.cat([samples[0], *recons], dim=-1).detach().cpu()
                        collage = torchvision.utils.make_grid(collage, 1, 0, True, (-1, 1))
                        caption = 'images' + ' | recons' * num_decodings
                        wandb.log({
                            'x | x_hat': wandb.Image(collage, caption='images' + ' | recons' * num_decodings),
                            'step': global_step,
                            'epoch': epoch,
                        })
                ddp.wait_for_everyone()
            if global_step > (5000 * (epoch + 1)): break

        # **************************************************************************************************************
        # Validation
        # **************************************************************************************************************
        ddp.wait_for_everyone()
        model.eval()
        metric.reset()
        mu_source, cov_source = get_statistics(model, val_source, ddp, "source statistics", latent_channels, normalize, mmse)
        mu_target, cov_target = get_statistics(model, val_target, ddp, "target statistics", latent_channels)
        transport = get_transport(cov_source, cov_target, mu_source, mu_target, ddp, model)
        with torch.inference_mode():
            for x, _ in (pbar := tqdm(val, desc=f"validation {epoch}", leave=False, disable=not ddp.is_local_main_process)):
                y = F.interpolate(denormalize(x), scale_factor=1./SF)
                bilinear = normalize(F.interpolate(y, scale_factor=SF, mode='bilinear')); update_metric("y", x, bilinear)
                x_star = normalize(mmse(y)); update_metric("x*", x, x_star)
                x_hat = model(x).recon; update_metric("x", x, x_hat)
                x_star_latent = model(x_star).z
                for i in range(10): update_metric(f"x0{i}", x, transport(x_star_latent, pg_star=i/10.))
                if DEBUG: break
            results = metric.compute()
            if ddp.is_local_main_process and not DEBUG:
                collage = torch.cat([bilinear, x_star, transport(x_star_latent, 0.5), transport(x_star_latent, 0.), x_hat, x], dim=-1).detach().cpu()
                collage = torchvision.utils.make_grid(collage[:num_samples], 1, 0, True, (-1, 1))
                res_list = [val for key, val in results.items() if any(['y' in key, 'x*' in key, 'x05' in key, 'x00' in key, 'psnr-x' == key, 'fid-x' == key])]
                caption = ' | '.join([f"{psnr.item():.1f} dB, {fid:.1f} FID" for psnr, fid in zip(res_list[::2], res_list[1::2])])
                wandb.log({
                    **results,
                    'y | x* | x05 | x00 | x': wandb.Image(collage, caption=caption),
                    'epoch': epoch
                })

        if DEBUG: break

    # ******************************************************************************************************************
    # Teardown
    # ******************************************************************************************************************
    ddp.print("run success")
    ddp.print("Beginning teardown")
    ddp.wait_for_everyone()
    ddp.save_state(f'{out_dir}/final_checkpoint.ckpt')
    ddp.clear()
    wandb.finish()
