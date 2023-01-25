"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a training script

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import os
import datetime
import math
from tqdm import tqdm
from functools import partial

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from torchmetrics import PeakSignalNoiseRatio
from accelerate import Accelerator
from vae import VAE
import wandb

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


# **********************************************************************************************************************
# Constants
# **********************************************************************************************************************

IMG_SIZE = 224
N_EPOCH = 10
N_ANNEALING = 4
CKPT_PATH = None
BETA = 1.
DEBUG = False


def cosine_annealing(step, n_step, max_val):
    return max_val * (0.5 * math.cos(math.pi * (step / n_step + 1)) + 0.5)


if __name__ == "__main__":
    # ******************************************************************************************************************
    # Data
    # ******************************************************************************************************************
    out_dir = f'logs/{datetime.datetime.now().strftime("%d-%m-%Y-%H:%M")}'
    os.makedirs(out_dir, exist_ok=True)

    # Load the ImageNet dataset
    transforms = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor()])
    imagenet_train = torchvision.datasets.ImageNet("~/data/ImageNet", transform=transforms)
    imagenet_val = torchvision.datasets.ImageNet("~/data/ImageNet", split="val", transform=transforms)

    # Create a data loader for the dataset
    train_dl = DataLoader(imagenet_train, batch_size=10, shuffle=True, num_workers=10, pin_memory=True)
    val_dl = DataLoader(imagenet_val, batch_size=32, shuffle=False, num_workers=10)

    # ******************************************************************************************************************
    # Model & Optimizer
    # ******************************************************************************************************************
    model = VAE(
        down_block_types=("DownEncoderBlock2D",) * 5,
        up_block_types=("UpDecoderBlock2D",) * 5,
        block_out_channels=(64, 64, 128, 128, 256, 256),  # noqa
        layers_per_block=2,
        latent_channels=256
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCH * len(train_dl), 1e-6)
    metric = PeakSignalNoiseRatio()

    # We use hugging face's `accelerator` library to handle ddp and device placement
    ddp = Accelerator()
    if CKPT_PATH is not None:
        ddp.load_state(CKPT_PATH)  # noqa
    model, optimizer, sched, train_dl, val_dl, metric = ddp.prepare(model, optimizer, sched, train_dl, val_dl, metric)

    if ddp.is_local_main_process and not DEBUG:
        run = wandb.init(project="icci", job_type="train")

    # ******************************************************************************************************************
    # Training
    # ******************************************************************************************************************
    global_step = 0
    annealed_beta = partial(cosine_annealing, n_step=N_ANNEALING * len(train_dl), max_val=BETA)

    for epoch in range(N_EPOCH):
        model.train()
        for x, _ in (pbar := tqdm(train_dl, desc=f"epoch {epoch}", leave=False, disable=not ddp.is_local_main_process)):
            optimizer.zero_grad()
            outputs = model(x, sample_posterior=BETA > 0, recon_loss="mse_loss", beta=annealed_beta(global_step))
            ddp.backward(outputs.total_loss)
            optimizer.step()
            sched.step()
            global_step += 1
            if DEBUG: break
            # **********************************************************************************************************
            # Logging & Checkpoint Saving
            # **********************************************************************************************************
            if ddp.is_local_main_process:
                log_dict = {
                    "loss": outputs.total_loss.item(),
                    "kl_unscaled": outputs.kl_unscaled.item(),
                    "kl": outputs.kl.item(),
                    "recon_loss": outputs.recon_loss.item(),
                }
                pbar.set_postfix(log_dict)
                wandb.log({**log_dict, 'step': global_step})

            if global_step % 1000 == 0:
                ddp.wait_for_everyone()
                ddp.save_state(f'{out_dir}/checkpoints/accelerator_full_state')
                ddp.unwrap_model(model).save_pretrained(
                    save_directory=f'{out_dir}/checkpoints',
                    save_function=ddp.save,
                    is_main_process=ddp.is_main_process
                )
                if ddp.is_local_main_process:
                    num_samples, num_decodings = 8, 4 if BETA > 0 else 1
                    images = x[:num_samples]
                    images = torch.cat([images, ] * num_decodings, dim=0)
                    recons = torch.chunk(model(images, sample_posterior=BETA > 0, return_dict=False), num_decodings, 0)

                    collage = torch.cat([x[:num_samples], *recons], dim=2).detach().cpu()
                    collage = torchvision.utils.make_grid(collage, num_samples, 0, True, (-1, 1))

                    wandb.log({
                        'collage': wandb.Image(collage, caption='images' + ' | recons' * num_decodings),
                        'step': global_step,
                        'epoch': epoch,
                    })
                ddp.wait_for_everyone()

        # **************************************************************************************************************
        # Validation
        # **************************************************************************************************************
        model.eval()
        with torch.no_grad():
            for x, _ in (pbar := tqdm(val_dl, desc=f"validation {epoch}", leave=False, disable=not ddp.is_local_main_process)):
                x_hat = model(x, sample_posterior=BETA > 0, return_dict=False)
                pbar.set_postfix({"metric": metric(x_hat, ddp.unwrap_model(model).norm(x)).item()})
                if DEBUG: break
            metric_result = metric.compute()
            metric.reset()
            if ddp.is_local_main_process and not DEBUG:
                wandb.log({'psnr': metric_result.item(), 'epoch': epoch})

        if DEBUG: break

    ddp.print("run success")
    ddp.wait_for_everyone()
    ddp.save_state(f'{out_dir}/final_checkpoint.ckpt')
    ddp.clear()
    wandb.finish()
