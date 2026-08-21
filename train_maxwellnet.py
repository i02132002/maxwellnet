# Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com

import torch
from Dataset import ShapeDataset
from MaxwellNet import MaxwellNet
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
import wandb
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import random
import logging
import argparse
import os
import json
from datetime import datetime


def main(load_ckpt, reset_lr=False, epochs_override=None, n_samples=None, skip_valid=False):
    directory = "test_run1"
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(
                            os.getcwd(), directory, f"maxwellnet_{datetime.now():%Y-%m-%d %H-%M-%S}.log"),
                        filemode='w')

    logging.info("training " + directory)

    specs_filename = os.path.join('specs_maxwell.json')

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs_maxwell.json"'
        )

    specs = json.load(open(specs_filename))

    seed_number = get_spec_with_default(specs, "Seed", None)
    if seed_number != None:
        fix_seed(seed_number, torch.cuda.is_available())

    rank = 0
    device = torch.device('cuda' if torch.cuda.is_available()
                           else 'mps' if torch.backends.mps.is_available()
                           else 'cpu')

    logging.info("Experiment description: \n" +
                 ' '.join([str(elem) for elem in specs["Description"]]))
    logging.info("Training with " + str(device))

    physical_specs = specs["PhysicalSpecs"]
    mode = physical_specs['mode']
    symmetry_x = physical_specs.get('symmetry_x', False)
    high_order = physical_specs.get('high_order', 'fourth')
    pml_thickness = physical_specs['pml_thickness']

    hf_config = get_spec_with_default(specs, "HFConfig", None)
    valid_fraction = get_spec_with_default(specs, "ValidFraction", 0.1)
    n_samples = n_samples if n_samples is not None else get_spec_with_default(specs, "NumTrainSamples", None)
    skip_valid = skip_valid or get_spec_with_default(specs, "SkipValid", False)
    train_dataset, valid_dataset = ShapeDataset.load_train_valid(
        hf_config, mode, valid_fraction, seed_number if seed_number is not None else 0, n_samples=n_samples)

    # MaxwellNet (unlike master's own fixed npz demo) needs its domain size
    # and wavelength/dpl fixed at construction time, but ShapeDataset draws
    # them from the actual sample rather than a static spec -- read them off
    # the first training sample (single/few-sample overfitting runs keep
    # this consistent for the whole run). wavelength=1 (dimensionless, in
    # units of the sample's own wavelength) matches master's own convention
    # of always normalizing to wavelength=1, so k=2*pi and grid spacing
    # (delta = wavelength/dpl) both land at the same scale master's own
    # demo uses, rather than an arbitrary absolute unit.
    probe = train_dataset[0]
    Nx, Nz = probe['scat_pot'].shape[-2:]
    wavelength_a = float(probe['wavelength_nm']) * 10.0
    delta_x_a = float(probe['delta_x_a'])
    dpl = wavelength_a / delta_x_a

    model = MaxwellNet(**specs["NetworkSpecs"], wavelength=1.0, dpl=dpl, Nx=Nx, Nz=Nz,
                       pml_thickness=pml_thickness, symmetry_x=symmetry_x, mode=mode, high_order=high_order)
    if torch.cuda.device_count() > 1:
        logging.info("Multiple GPUs: " + str(torch.cuda.device_count()))
    if load_ckpt is not None:
        load_path = os.path.join(os.getcwd(), directory, 'model', load_ckpt)
        ckpt_dict = torch.load(load_path + '.pt')
        ckpt_epoch = ckpt_dict['epoch']
        logging.info("Checkpoint loaded from {}-epoch".format(ckpt_epoch))
        model.load_state_dict(ckpt_dict['state_dict'])
        wandb_run_id = ckpt_dict.get('wandb_run_id')
    else:
        wandb_run_id = None

    model = torch.nn.DataParallel(model)
    model.train()
    model = model.to(device)

    logging.info("Number of network parameters: {}".format(
        sum(p.data.nelement() for p in model.parameters())))
    logging.debug(specs["NetworkSpecs"])
    logging.debug(specs["PhysicalSpecs"])

    optimizer = torch.optim.Adam(model.parameters(), lr=get_spec_with_default(
        specs, "LearningRate", 0.0001), weight_decay=0)
    scheduler = StepLR(optimizer, step_size=get_spec_with_default(
        specs, "LearningRateDecayStep", 10000), gamma=get_spec_with_default(specs, "LearningRateDecay", 1.0))

    batch_size = get_spec_with_default(specs, "BatchSize", 1)
    epochs = epochs_override if epochs_override is not None else get_spec_with_default(specs, "Epochs", 1)
    snapshot_freq = specs["SnapshotFrequency"]

    checkpoints = list(range(snapshot_freq, epochs + 1, snapshot_freq))

    filename = 'maxwellnet_' + mode
    wandb_specs = get_spec_with_default(specs, "WandB", {})
    log_freq = wandb_specs.get("LogFrequency", 1)
    wandb.init(
        project=wandb_specs.get("Project", "maxwellnet"),
        entity=wandb_specs.get("Entity"),
        mode=wandb_specs.get("Mode", "online"),
        dir=directory,
        name=filename,
        config=specs,
        id=wandb_run_id,
        resume="allow" if wandb_run_id is not None else None,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    wandb.define_metric("train_step")
    wandb.define_metric("train/loss_step", step_metric="train_step")

    # Fetching a sample means decoding an HF Arrow row, which is CPU-bound
    # and otherwise serializes with GPU/MPS compute; worker processes
    # overlap it instead.
    num_workers = get_spec_with_default(specs, "NumWorkers", min(8, os.cpu_count() or 0))
    # Cap below the dataset size so an overfitting run (n_samples small)
    # doesn't spin up worker processes it has nothing to hand them.
    train_num_workers = min(num_workers, len(train_dataset))
    train_loader_kwargs = dict(num_workers=train_num_workers, pin_memory=(device.type == "cuda"))
    if train_num_workers > 0:
        train_loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device.type == "cuda"))
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=(n_samples != 1), sampler=None, **train_loader_kwargs)
    logging.info("Train Dataset length: {}".format(len(train_dataset)))
    loss_train = torch.zeros(
        (int(epochs),), dtype=torch.float32, requires_grad=False)

    perform_valid = len(valid_dataset) > 0 and not skip_valid

    if perform_valid == True:
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                   shuffle=False, sampler=None, **loader_kwargs)
        logging.info("Valid Dataset length: {}".format(len(valid_dataset)))
        loss_valid = torch.zeros(
            (int(epochs),), dtype=torch.float32, requires_grad=False)

    if load_ckpt is not None:
        if reset_lr:
            new_lr = get_spec_with_default(specs, "LearningRate", 0.0001)
            for group in optimizer.param_groups:
                group['lr'] = new_lr
            logging.info("Resetting LR to {} and restarting decay schedule from {}-epoch".format(
                new_lr, ckpt_epoch))
        else:
            optimizer.load_state_dict(ckpt_dict['optimizer'])
            scheduler.load_state_dict(ckpt_dict['scheduler'])
        loss_train[:ckpt_epoch:] = ckpt_dict['loss_train'][:ckpt_epoch:]
        logging.info("Check point loaded from {}-epoch".format(ckpt_epoch))

        start_epoch = ckpt_epoch
    else:
        start_epoch = 0

    logging.info("Training start")

    for epoch in range(start_epoch + 1, epochs + 1):
        train(train_loader, model, optimizer, epoch, loss_train, device, mode, log_freq)
        logging.info("[Train] {} epoch. Loss: {:.5f}".format(
            epoch, loss_train[epoch-1].item())) if rank == 0 else None
        if perform_valid:
            valid(valid_loader, model, epoch, loss_valid, device, mode, log_freq)
            logging.info("[Valid] {} epoch. Loss: {:.5f}".format(
                epoch, loss_valid[epoch-1].item())) if rank == 0 else None

        if epoch in checkpoints:
            logging.info("Checkpoint saved at {} epoch.".format(
                epoch)) if rank == 0 else None
            if rank == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_train': loss_train,
                    'scheduler': scheduler.state_dict(),
                    'wandb_run_id': wandb.run.id,
                }, directory, str(epoch) + '_' + mode)

        if epoch % 200 == 0:
            logging.info("'latest' checkpoint saved at {} epoch.".format(
                epoch)) if rank == 0 else None
            if rank == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_train': loss_train,
                    'scheduler': scheduler.state_dict(),
                    'wandb_run_id': wandb.run.id,
                }, directory, 'latest')

        wandb.log({'train/lr': optimizer.param_groups[0]['lr'], 'epoch': epoch}) if rank == 0 else None

        scheduler.step()

    wandb.finish() if rank == 0 else None


def train(train_loader, model, optimizer, epoch, loss_train, device, mode, log_freq):
    model.train()
    n_batches = len(train_loader)
    log_total = log_diff = log_sample_ids = None
    with torch.set_grad_enabled(True):
        count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for batch_idx, data in enumerate(pbar):
            scat_pot = data['scat_pot'].to(device)
            ri_value = data['ri_value'].to(torch.float32).to(device)

            diff, total = model(scat_pot, ri_value)

            loss = torch.mean(diff.pow(2))
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
            optimizer.step()

            loss_train[epoch-1] += loss.item() * diff.size(0)
            count += diff.size(0)
            pbar.set_postfix(loss=loss.item(), grad_norm=grad_norm.item())

            train_step = (epoch - 1) * n_batches + batch_idx
            wandb.log({'train/loss_step': loss.item(), 'train/grad_norm': grad_norm.item(), 'train_step': train_step})

            if batch_idx == 0:
                log_total = total.detach().cpu()
                log_diff = diff.detach().cpu()
                log_sample_ids = list(data['sample_id'])

        loss_train[epoch-1] = loss_train[epoch-1] / count

    wandb.log({'train/loss': loss_train[epoch-1].item(), 'epoch': epoch})
    if log_freq and epoch % log_freq == 0:
        log_fields_to_wandb(log_total, log_diff, log_sample_ids, mode, 'train', epoch)


def valid(valid_loader, model, epoch, loss_valid, device, mode, log_freq):
    model.eval()
    log_total = log_diff = log_sample_ids = None
    with torch.set_grad_enabled(False):
        count = 0

        pbar = tqdm(valid_loader, desc=f"Epoch {epoch} [valid]", leave=False)
        for batch_idx, data in enumerate(pbar):
            scat_pot = data['scat_pot'].to(device)
            ri_value = data['ri_value'].to(torch.float32).to(device)

            diff, total = model(scat_pot, ri_value)

            loss = torch.mean(diff.pow(2))
            loss_valid[epoch-1] += loss.item() * diff.size(0)
            count += diff.size(0)
            pbar.set_postfix(loss=loss.item())

            if batch_idx == 0:
                log_total = total.detach().cpu()
                log_diff = diff.detach().cpu()
                log_sample_ids = list(data['sample_id'])

        loss_valid[epoch-1] = loss_valid[epoch-1] / count

    wandb.log({'valid/loss': loss_valid[epoch-1].item(), 'epoch': epoch})
    if log_freq and epoch % log_freq == 0:
        log_fields_to_wandb(log_total, log_diff, log_sample_ids, mode, 'valid', epoch)


def log_fields_to_wandb(total, diff, sample_ids, mode, train_valid, epoch):
    """total, diff: (B, 2, Nx, Nz) real/imag-channel-stacked tensors, as
    returned by MaxwellNet.forward for 'te' mode -- total is the predicted
    field envelope (real=offset+1, imag=offset), diff the PDE residual."""
    images = {}
    real, imag = total[0, 0], total[0, 1]
    amplitude = torch.sqrt(real**2 + imag**2)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    for ax, part, title in zip(axes, (real, imag, amplitude), ('real', 'imaginary', 'amplitude')):
        im = ax.imshow(part.numpy(), origin='lower', aspect='equal')
        ax.set_title(f'{sample_ids[0]}\n{title}(envelope)', fontsize=9)
        fig.colorbar(im, ax=ax)

    residual_energy = diff[0, 0].pow(2) + diff[0, 1].pow(2)
    im = axes[3].imshow(residual_energy.numpy(), origin='lower', aspect='equal')
    axes[3].set_title(f'{sample_ids[0]}\n|diff|^2', fontsize=9)
    fig.colorbar(im, ax=axes[3])

    images[f'{train_valid}/{mode}/field_grid'] = wandb.Image(fig)
    plt.close(fig)

    images['epoch'] = epoch
    wandb.log(images)


def save_checkpoint(state, directory, filename):
    model_directory = os.path.join(directory, 'model')
    if os.path.exists(model_directory) == False:
        os.makedirs(model_directory)
    torch.save(state, os.path.join(model_directory, filename + '.pt'))


def fix_seed(seed, is_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Train a MaxwellNet")
    arg_parser.add_argument(
        "--load_ckpt",
        "-l",
        default=None,
        help="This should specify a filename of your checkpoint within 'directory'/model if you want to continue your training from the checkpoint.",
    )
    arg_parser.add_argument(
        "--reset_lr",
        action="store_true",
        help="When resuming from --load_ckpt, ignore the checkpoint's saved optimizer/scheduler state and restart the optimizer at the LearningRate in specs_maxwell.json, with the decay schedule restarting from the checkpoint's epoch.",
    )
    arg_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override Epochs from specs_maxwell.json.",
    )
    arg_parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Limit training to this many rows (e.g. 1, for an overfitting sanity check). Overrides NumTrainSamples in specs_maxwell.json.",
    )
    arg_parser.add_argument(
        "--skip_valid",
        action="store_true",
        help="Skip validation entirely, regardless of ValidFraction, for faster overfitting runs. Also honored via SkipValid in specs_maxwell.json.",
    )

    args = arg_parser.parse_args()
    main(args.load_ckpt, args.reset_lr, args.epochs, args.n_samples, args.skip_valid)
